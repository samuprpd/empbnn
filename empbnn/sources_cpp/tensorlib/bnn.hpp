#pragma once

#include "tensor.hpp"

namespace Tensorlib {
namespace Bnn {

enum class Family {Gaussian, Uniform};
enum class Mode {Sample, MomentProp};

// Global Env vars

// Random generator
extern u32 rand_seed;
// Print ID 
// Avoid passing multiple parameters to mc_softmax
extern u32 result_id;

// https://en.wikipedia.org/wiki/Xorshift
constexpr u32 xorshift32(u32& x) {
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x;
}

// Requires parameter for the templates know what type to infer
// Default uses float typecasting
template<typename T>
constexpr T uniform(const T&) {
	return T(f32(xorshift32(rand_seed)) / f32(0xFFFFFFFF));
}

// Fixed point specific implementation
template<typename L, typename H, u32 S>
constexpr fxp<L,H,S> uniform(const fxp<L,H,S>&) {
	fxp<L,H,S> r;
	r.data = xorshift32(rand_seed) >> (32 - r.scale);
	return r;
}

template<typename T>
constexpr T clt_gaussian(const T& _v) {
	T acc = 0;
	for (u32 i = 0; i < 12; i++) acc += uniform(_v);
	return acc - T(6);
}

template<typename T, Mode mode, Family dist>
struct Data {
	T mean = 0;

	// if mode -> Sample, use stdev else var
	union {
		T var = 0;
		T std;
	};

	constexpr Data() {}

	template<typename K>
	constexpr Data(const K a) {mean = T(a); var = T(0);}

	template<typename K>
	constexpr Data(const K a, const K b) {mean = T(a); var = T(b);}

	template<typename K>
	constexpr operator K() { return sample(*this); }

	constexpr T stdev() const {return sqrt(var);}
};


// Data MomentProp Gaussian
template<typename T>
using DMPG = Data<T, Mode::MomentProp, Family::Gaussian>;
// Data Sample Gaussian
template<typename T>
using DSG = Data<T, Mode::Sample, Family::Gaussian>;
// Data Sample Uniform
template<typename T>
using DSU = Data<T, Mode::Sample, Family::Uniform>;


template<typename T>
constexpr void mp_to_sample(const DMPG<T>& v, DSG<T>& r) {
	r.mean = v.mean;
	r.var = v.stdev();
}

template<typename T>
constexpr void mp_to_sample(const DMPG<T>& v, DSU<T>& r) {
	r.var = sqrt(v.var * T(12));
	r.mean = v.mean - r.var / T(2);
}

// Operator for sampling mode

template<typename T>
constexpr T sample(const DSG<T>& w) {
	return T(w.var * clt_gaussian(w.mean) + w.mean);
}

template<typename T>
constexpr T sample(const DSU<T>& w) {
	return T(w.var * uniform(w.mean) + w.mean);
}

// Operators for moment propagation mode mode

template<typename T, typename K>
constexpr DMPG<T> operator*(const DMPG<T>& v, const DMPG<K>& w) {
	DMPG<T> aux;
	aux.mean = T(w.mean);
	aux.var = T(w.var);

	DMPG<T> r;
	// Output mean
	r.mean = v.mean * aux.mean;

	// Output var
	T vm2 = v.mean * v.mean;
	T wm2 = aux.mean * aux.mean;
	r.var = (vm2 + v.var) * (wm2 + aux.var) - vm2 * wm2; 

	return r;
}

template<typename T, typename K>
constexpr DMPG<T>& operator*=(DMPG<T>& v, const DMPG<K>& w) {
	auto r = v * w;
	v.mean = r.mean;
	v.var = r.var;
	return v;
}

template<typename T, typename K>
constexpr DMPG<T> operator/(const DMPG<T>& v, const DMPG<K>& w) {
	DMPG<T> invw;
	invw.mean = T(1) / T(w.mean);
	invw.var = w.var;

	return v * invw;
}

template<typename T, typename K>
constexpr DMPG<T> operator+(const DMPG<T>& v, const DMPG<K>& w) {
	DMPG<T> r;
	r.mean = v.mean + T(w.mean);
	r.var = v.var + T(w.var);
	return r;
}

template<typename T, typename K>
constexpr DMPG<T>& operator+=(DMPG<T>& v, const DMPG<K>& w) {
	v.mean += T(w.mean);
	v.var += T(w.var);
	return v;
}

template<typename T, typename K>
constexpr DMPG<T> operator+(const DMPG<T>& v, const K& w) {
	DMPG<T> r;
	r.mean = v.mean + T(w);
	return r;
}

template<typename T, typename K>
constexpr DMPG<T> operator-(const DMPG<T>& v,const DMPG<K>& w) {
	DMPG<T> r;
	r.mean = v.mean - T(w.mean);
	r.var = v.var + T(w.var);
	return r;
}

template<typename T, typename K>
constexpr DMPG<T> operator-(const DMPG<T>& v, const K& w) {
	DMPG<T> r;
	r.mean = v.mean - T(w);
	return r;
} 

// Online mean and variance struct algorithm
// Using shifted data by K algorithm
template<u32 nsamples, typename T>
struct OMV {
	T ex = 0, ex2 = 0, k = 0;
	bool init = true;

	constexpr OMV() {}

	template<typename K>
	constexpr void add_sample(K sample) {

		auto tsample = T(sample);

		if (init) {
			k = tsample;
			init = false;
		}

		auto sk = tsample - k;
		ex += sk;
		ex2 += sk * sk;
	}

	template<typename K>
	constexpr void get_mean_var(K& mean, K& var) {
		auto aux = K(ex) / K(nsamples);
		mean = K(k) + aux;
		var = (K(ex2) - aux * K(ex)) / K(nsamples - 1);
	}

	template<typename K>
	constexpr void get_mean_var(DMPG<K>& r) {
		get_mean_var(r.mean, r.var);
	}

};

template<u32 nsamples, typename T>
constexpr OMV<nsamples, T> OMV_init(const T&) {return OMV<nsamples, T>();}

template<u32 nsamples, typename L, typename H, u32 S>
constexpr OMV<nsamples, fxp<L,i64,S>> OMV_init(const fxp<L,H,S>&) {
	// Make sure use i64 as large data type to avoid overflows
	return OMV<nsamples, fxp<L,i64,S>>();
}

template<typename T, u32... DIMS>
constexpr DMPG<T> simple_max_pool(Tensor<DMPG<T>, DIMS...>& x) {
	DMPG<T> r = x.buffptr[0]; 
	for (u32 i = 1; i < x.total_size; i++) {
		if (x.buffptr[i].mean > r.mean) r = x.buffptr[i];
	}
	return r;
}

template<
	Family sample_family, u32 nsamples, 
	typename T, u32... DIMS, template <typename, u32...> class Tensor_X
>
constexpr DMPG<T> mc_max_pool(Tensor_X<DMPG<T>, DIMS...>& x) {

	TensorBuffer<Data<T, Mode::Sample, sample_family>, DIMS...> sample_mode_x;
	TensorBuffer<T, DIMS...> sample_x;

	for (u32 i = 0; i < x.total_size; i++) mp_to_sample(x.buffptr[i], sample_mode_x.buffptr[i]);

	auto omv = OMV_init<nsamples>(sample_x[{0}]);
	for(u32 s = 0; s < nsamples; s++) {
		for (u32 i = 0; i < x.total_size; i++) sample_x.buffptr[i] = sample(sample_mode_x.buffptr[i]);
		omv.add_sample(sample_x.max());
	}

	DMPG<T> r;
	omv.get_mean_var(r);
	return r;
}

template<typename T>
constexpr DMPG<T> simple_relu(const DMPG<T>& x) {
	DMPG<T> r = 0;
	if (x.mean > T(0)) r = x;
	return r;
}

template<Family sample_family, u32 nsamples, typename T>
constexpr DMPG<T> mc_relu(const DMPG<T>& x) {

	Data<T, Mode::Sample, sample_family> sample_x;
	mp_to_sample(x, sample_x);
	auto omv = OMV_init<nsamples>(x.mean);

	for(u32 s = 0; s < nsamples; s++) omv.add_sample(relu(sample(sample_x)));
	
	DMPG<T> r;
	omv.get_mean_var(r);
	return r;
}

template <typename T> 
constexpr void print_result(const T& r) {
	for (u32 i = 0; i < r.total_size; i++) {
		std::cout << f32(r.buffptr[i]);
		if (i < r.total_size - 1) std::cout << ", ";
	}
	std::cout << '\n';
}


template<
	Family sample_family, u32 nsamples, 
	typename T, u32... DIMS, template <typename, u32...> class Tensor_X
>
constexpr void mc_softmax(Tensor_X<DMPG<T>, DIMS...>& x) {
	
	TensorBuffer<Data<T, Mode::Sample, sample_family>, DIMS...> sample_mode_x;
	TensorBuffer<T, DIMS...> sample_x, dst;

	for (u32 i = 0; i < x.total_size; i++) mp_to_sample(x.buffptr[i], sample_mode_x.buffptr[i]);

	for(u32 s = 0; s < nsamples; s++) {
		for (u32 i = 0; i < x.total_size; i++) sample_x.buffptr[i] = sample(sample_mode_x.buffptr[i]);
		softmax(sample_x, dst);
		print_result(dst);
	}
}

template<typename T, Family sample_family, u32... DIMS>
constexpr void mp2s_layer(
	Tensor<DMPG<T>, DIMS...>& in, 
	Tensor<Data<T, Mode::Sample, sample_family>, DIMS...>& out
) {
	for (u32 i = 0; i < in.total_size; i++) mp_to_sample(in.buffptr[i], out.buffptr[i]);
}

template<typename T, Family sample_family, u32... DIMS>
constexpr void sample_layer(
	Tensor<Data<T, Mode::Sample, sample_family>, DIMS...>& in, 
	Tensor<T, DIMS...>& out
) {
	for (u32 i = 0; i < in.total_size; i++) out.buffptr[i] = sample(in.buffptr[i]);
}

}
}

template<typename T, Tensorlib::Bnn::Mode mode>
std::ostream& operator<<(
	std::ostream& os, 
	const Tensorlib::Bnn::Data<T, mode, Tensorlib::Bnn::Family::Gaussian>& d
) {
	os << "N(" << d.mean << " " << d.var << ")";
	return os;
}

template<typename T, Tensorlib::Bnn::Mode mode>
std::ostream& operator<<(
	std::ostream& os, 
	const Tensorlib::Bnn::Data<T, mode, Tensorlib::Bnn::Family::Uniform>& d
) {
	os << "U(" << d.mean << " " << d.var << ")";
	return os;
}

