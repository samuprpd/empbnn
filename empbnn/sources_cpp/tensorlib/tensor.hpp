#pragma once

/*
	Statically sized tensor library
*/

#include <array>
#include <cassert>
#include <fstream>
#include <initializer_list>
#include <tuple>
#include <iostream>

#include "fxp.hpp"

// Forward declaration of classes
template<typename T, u32 ...DIMS> class Tensor;
template<typename T, u32 ...DIMS> class TensorView;

// Buffer class
template<typename T, u32 ...DIMS>
class TensorBuffer : public Tensor<T, DIMS...> {

  public:
	using Tensor_t = Tensor<T,DIMS...>;

	// Actual data buffer
	std::array<T, Tensor_t::total_size> buff;

	// Constructor from values
	constexpr TensorBuffer(const std::initializer_list<T>& values = {}) {

		this->buffptr = buff.data();

		// Cant do this as constexpr in clang, yes in gcc
		// Some asserts could be uncommented to add more safety to the library
		// assert(Tensor_t::total_size >= values.size() && "Tensor is not big enough");

		u32 i = 0;
		for (auto& v : values) {
			buff[i] = v;
			i++;
		}

	}

	// Copy constructor
	template<typename K, u32 ...KDIMS>
	constexpr TensorBuffer(const Tensor<K, KDIMS...>& src) {
		
		static_assert(Tensor_t::total_size == (1 * ... * KDIMS), "Tensor shape missmatch");
		
		this->buffptr = buff.data();

		for (u32 i = 0; i < Tensor_t::total_size; i++) {
			buff[i] = T(src.buffptr[i]);
		}
	}


	// Stride 2D Copy constructor
	template<typename K, u32 KD0, u32 KD1, u32 ...KDIMS>
	constexpr TensorBuffer(
		const Tensor<K, KD0, KD1, KDIMS...>& src, 
		const std::tuple<u32,u32> r0, const std::tuple<u32,u32> r1
	) {

		this->buffptr = buff.data();
		auto& self = *this;
		
		auto i0_start = std::get<0>(r0);
		auto i0_end = std::get<1>(r0);

		auto i1_start = std::get<0>(r1);
		auto i1_end = std::get<1>(r1);

		/*
		// Check that the tensor size fits
		assert(
			(i0_end - i0_start + 1) * (i1_end - i1_start + 1) * (1 * ... * KDIMS) <= (1 * ... * DIMS) && 
			"Dst tensor is not big enough"
		);
		*/
		
		for (u32 i0 = i0_start; i0 <= i0_end; i0++) {
			for (u32 i1 = i1_start; i1 <= i1_end; i1++) {
				TensorView<K,KDIMS...,1> (self, {i0-i0_start, i1-i1_start}).copy(TensorView<K,KDIMS...,1>(src, {i0, i1}));
			}
		}
	}

	constexpr TensorView<T, DIMS...> view() {
		return TensorView<T, DIMS...>(this->buffptr);
	}

};

// Padded with 0s constructor
template<
	u32 VD0, u32 VD1,
	u32 DC0, u32 DC1, typename KC, u32... DC
>
constexpr TensorBuffer<KC,VD0,VD1,DC...> padded_lr_buffer_2d(
	Tensor<KC,DC0,DC1,DC...>& src, 
	const std::tuple<i32,i32> r0, const std::tuple<i32,i32> r1
) {
	
	auto r = TensorBuffer<KC,VD0,VD1,DC...>();

	auto i0_start = std::get<0>(r0);
	auto i0_end = std::get<1>(r0);

	auto i1_start = std::get<0>(r1);
	auto i1_end = std::get<1>(r1);

	// assert(i0_end - i0_start + 1 == VD0 && i1_end - i1_start + 1 == VD1 && "Shape missmatch");

	for (u32 i0 = 0; i0 < VD0; i0++) {
		for (u32 i1 = 0; i1 < VD1; i1++) {

			// Create a view of the rest of dimensions after 2d access
			auto rv = TensorView<KC,DC...,1> (r, {i0, i1});
	
			i32 ii0 = i0 + i0_start;
			i32 ii1 = i1 + i1_start;

			// Pad
			if (ii0 < 0 || ii1 < 0 || ii0 >= DC0 || ii1 >= DC1) rv.setval(0);
			// Copy rest of dimensions from a 2d access view of input
			else rv.copy(TensorView<KC,DC...,1>(src, {u32(ii0), u32(ii1)}));
		}
	}

	return r;
}

// View/Reference class

template<typename T, u32 ...DIMS>
class TensorView : public Tensor<T, DIMS...> {

  public:

	template<u32 ...KDIMS>
	constexpr TensorView(Tensor<T, KDIMS...>& t, std::initializer_list<u32> idxs = {}) {
		this->buffptr = t.buffptr + t.ptr_offset(idxs);
	}

	constexpr TensorView(T* buffptr) {
		this->buffptr = buffptr;
	}

};


// Abstract class
template<typename T, u32 ...DIMS>
class Tensor {

	friend class TensorView<T,DIMS...>;
	friend class TensorBuffer<T,DIMS...>;
	Tensor(){}

  public:

	// Raw pointer to data
	T* buffptr = nullptr;

	// Total size and memory buffer
	static constexpr u32 total_size = (1 * ... * DIMS);
	static constexpr u32 ndims = sizeof...(DIMS);
	static constexpr std::array<u32, ndims> shape = {DIMS...};

	static constexpr std::array<u32, ndims> compute_strides() {
		std::array<u32, ndims> r = {};
		
		u32 prod = 1;
		
		for(i32 i = ndims - 1; i >= 0; i--) {
			r[i] = prod;
			prod *= shape[i];
		}
	
		return r;
	}
	static constexpr auto strides = compute_strides();

	// Get pointer offset given a list of index
	constexpr u32 ptr_offset(const std::initializer_list<u32>& idxs) {

		// Cant do this as constexpr in clang
		// assert(idxs.size() <= ndims && "More index than dimensions");

		// This loop should always be unrolled and inlined
		u32 i = 0, ptr = 0;
		for (const auto idx : idxs) {
			ptr += idx * strides[i];
			i++;
		}

		// This changes the cfg graph a lot
		// assert(ptr < total_size && "Bound check failure");

		return ptr;
	}

	constexpr T& operator[](const std::initializer_list<u32> idxs) {
		return buffptr[ptr_offset(idxs)];
	}

	// Simple binary dump format

	void read_from_file(std::ifstream& input_file) {
		input_file.read((char*)(buffptr), sizeof(T) * total_size);
	}

	void write_to_file(std::ofstream& output_file) {
		output_file.write((char*)(buffptr), sizeof(T) * total_size);
	}

	template<typename KA, u32... DA>
	constexpr void copy(const Tensor<KA,DA...>& from) {

		static_assert(total_size >= (1 * ... * DA) && "Dst tensor smaller than src");

		for (u32 i = 0; i < from.total_size; i++) {
			buffptr[i] = T(from.buffptr[i]);
		}
	}

	constexpr std::pair<u32,T> argmax() {
		T m = buffptr[0];
		u32 r = 0;

		for (u32 i = 1; i < total_size; i++) {
			if (buffptr[i] > m) {
				m = buffptr[i];
				r = i;
			}
		}

		return std::pair(r, m);
	}

	constexpr T max() {
		return argmax().second;
	}

	constexpr T min() {
		T m = buffptr[0];
		for (u32 i = 1; i < total_size; i++) if (buffptr[i] < m) m = buffptr[i];
		return m;
	}

	constexpr T sum() {
		T acc = 0;
		for (u32 i = 0; i < total_size; i++) acc += buffptr[i];
		return acc;
	}

	constexpr T avg() {
		return sum() / T(total_size);
	}

	template<typename KA, u32... DA>
	constexpr T mac(const Tensor<KA,DA...>& a) {
		T r = 0;
		for (u32 i = 0; i < total_size; i++) r += buffptr[i] * T(a.buffptr[i]);
		return r;
	}

	// Tensor = fn(Tensor, value) Flat iteration
	template<typename K, typename FN>
	constexpr Tensor& flat_iter_val(const K& a, const FN& fn) {
		for (u32 i = 0; i < total_size; i++) {
			buffptr[i] = fn(buffptr[i], a);
		}
		return *this;
	}

	template<typename K>
	constexpr Tensor& setval(const K& v) {
		flat_iter_val(v, [](const T& _c, const K& _a) -> T {
			return T(_a);
		});
		return *this;
	}

	template<typename K>
	constexpr Tensor& add(const K& a) {
		return flat_iter_val(a, [](const T& _c, const K& _a) -> T {
			return _c + T(_a);
		});
	}

	template<typename K>
	constexpr Tensor& mul(const K& a) {
		return flat_iter_val(a, [](const T& _c, const K& _a) -> T {
			return _c * T(_a);
		});
	}

	template<typename K>
	constexpr Tensor& sub(const K& a) {
		return flat_iter_val(a, [](const T& _c, const K& _a) -> T {
			return _c - T(_a);
		});
	}

	template<typename K>
	constexpr Tensor& div(const K& a) {
		return flat_iter_val(a, [](const T& _c, const K& _a) -> T {
			return _c / T(_a);
		});
	}

	// t[i] = fn(t[i], a[i]) Flat iteration
	template<typename KA, u32... DA, typename FN>
	constexpr Tensor& flat_iter_tensor(const Tensor<KA,DA...>& a, const FN& fn) {
		
		// Static assert size checking
		constexpr auto s1 = (1 * ... * DA);
		static_assert(total_size == s1, "Tensor size missmatch");

		for (u32 i = 0; i < total_size; i++) buffptr[i] = fn(buffptr[i], a.buffptr[i]);

		return *this;
	}

	/*
		This functions require complex templating so they are not overriden with their
		simple version
	*/

	template<typename KA, u32... DA, template <typename, u32...> class TA>
	constexpr Tensor& add(const TA<KA,DA...>& a) {
		return flat_iter_tensor(a, [](const T& _c, const KA& _a) -> T {
			return _c + T(_a);
		});
	}

	template<typename KA, u32... DA, template <typename, u32...> class TA>
	constexpr Tensor& mul(const TA<KA,DA...>& a) {
		return flat_iter_tensor(a, [](const T& _c, const KA& _a) -> T {
			return _c * T(_a);
		});
	}

	template<typename KA, u32... DA, template <typename, u32...> class TA>
	constexpr Tensor& sub(const TA<KA,DA...>& a) {
		return flat_iter_tensor(a, [](const T& _c, const KA& _a) -> T {
			return _c - T(_a);
		});
	}

	template<typename KA, u32... DA, template <typename, u32...> class TA>
	constexpr Tensor& div(const TA<KA,DA...>& a) {
		return flat_iter_tensor(a, [](const T& _c, const KA& _a) -> T {
			return _c / T(_a);
		});
	}


	// t[i] = fn(t[i], a[i], b[i]) Flat iteration
	template<typename KA, u32... DA, typename KB, u32... DB, typename FN>
	constexpr Tensor& flat_iter_2_tensor(const Tensor<KA,DA...>& a, const Tensor<KB,DB...>& b, const FN& fn) {
		
		// Static assert size checking
		constexpr auto s1 = (1 * ... * DA);
		constexpr auto s2 = (1 * ... * DB);
		static_assert(total_size == s1 && total_size == s2, "Tensor size missmatch");

		for (u32 i = 0; i < total_size; i++) {
			buffptr[i] = fn(buffptr[i], a.buffptr[i], b.buffptr[i]);
		}
		return *this;
	}

	template<typename KA, u32... DA, typename KB, u32... DB>
	constexpr Tensor& add(const Tensor<KA,DA...>& a, const Tensor<KB,DB...>& b) {
		return flat_iter_2_tensor(a, b, [](const T&, const KA& _a, const KB& _b) -> T {
			return T(_a) + T(_b);
		});
	}

	template<typename KA, u32... DA, typename KB, u32... DB>
	constexpr Tensor& mul(const Tensor<KA,DA...>& a, const Tensor<KB,DB...>& b) {
		return flat_iter_2_tensor(a, b, [](const T&, const KA& _a, const KB& _b) -> T {
			return T(_a) * T(_b);
		});
	}

	template<typename KA, u32... DA, typename KB, u32... DB>
	constexpr Tensor& sub(const Tensor<KA,DA...>& a, const Tensor<KB,DB...>& b) {
		return flat_iter_2_tensor(a, b, [](const T&, const KA& _a, const KB& _b) -> T {
			return T(_a) - T(_b);
		});
	}

	template<typename KA, u32... DA, typename KB, u32... DB>
	constexpr Tensor& div(const Tensor<KA,DA...>& a, const Tensor<KB,DB...>& b) {
		return flat_iter_2_tensor(a, b, [](const T&, const KA& _a, const KB& _b) -> T {
			return T(_a) / T(_b);
		});
	}
};


namespace Tensorlib {

template <typename A, typename B, typename C>
constexpr A add_kernel(B& b, C& c) {return A(b + c);}

template <typename A, typename B, typename C>
constexpr A mac_kernel(B& w, C& x) {return A(x.mac(w));}

template<typename T>
constexpr T relu(const T& x) {return x > T(0) ? x : T(0);}

template<typename T>
constexpr T identity(const T& x) {return x;}

template<typename R, typename T>
constexpr R max_pool(T& x) {return x.max();}

template<typename R, typename T>
constexpr R avg_pool(T& x) {return x.avg();}

template<typename A_T, typename B_T, typename D_T, u32... DIMS>
constexpr void nn_flat_iter(
	Tensor<A_T, DIMS...>& a, Tensor<B_T, DIMS...>& b, Tensor<D_T, DIMS...>& dst,
	D_T (fkernel) (A_T&, B_T&), 
	D_T (fact) (const D_T&)
) {
	for (u32 i = 0; i < dst.total_size; i++) {
		dst.buffptr[i] = fact(fkernel(a.buffptr[i], b.buffptr[i]));
	}
}

// Matrix D0 x D1, Vector D1, Vector D1
template<u32 D0, u32 D1, typename D_T, typename X_T, typename W_T, typename B_T, u32... DIMS>
constexpr void nn_linear_iter(
	Tensor<D_T, D0>& dst,
	Tensor<X_T, D1>& x, Tensor<W_T, D0, D1>& w, Tensor<B_T, D0>& b,
	D_T (fkernel) (Tensor<W_T, D1>&, Tensor<X_T, D1>&),
	D_T (fact) (const D_T&)
) {
	for (u32 i = 0; i < D0; i++) {
		auto weight_row = TensorView<W_T,D1>(w, {i});
		dst[{i}] = fact(D_T(b[{i}]) + fkernel(weight_row, x));
	}
}

constexpr u32 nn_conv2d_size(u32 di, u32 p, u32 k, u32 s, u32 dl) {
	return (di + 2 * p - dl * (k - 1) - 1) / s + 1;
}

// Tensor XD0 x XD1 x KD2, Kernels NK x KD0 x KD1 x KD2 -> Tensor ? x ? x NK
template<
	u32 NK, u32 KD0, u32 KD1, u32 KD2, u32 XD0, u32 XD1, u32... DIMS,
	typename D_T, typename X_T, typename K_T, typename B_T
>
constexpr void nn_conv2d_iter(
	Tensor<D_T, DIMS...>& dst,
	Tensor<X_T, XD0, XD1, KD2>& x, Tensor<K_T, NK, KD0, KD1, KD2>& k, Tensor<B_T, NK>& b,
	const std::tuple<u32,u32>& stride, const std::tuple<u32,u32>& pad,
	D_T (fkernel) (Tensor<K_T, KD0, KD1, KD2>&, Tensor<X_T, KD0, KD1, KD2>&),
	D_T (fact) (const D_T&)
) {
	u32 pad_i = 0, pad_j = 0;
	std::tie(pad_i, pad_j) = pad;
	u32 stride_i = 0, stride_j = 0;
	std::tie(stride_i, stride_j) = stride;

	const u32 h_out = nn_conv2d_size(XD0, pad_i, KD0, stride_i, 1);
	const u32 w_out = nn_conv2d_size(XD1, pad_j, KD1, stride_j, 1);
	/*
	Using this asserts changes a lot the code generated by clang
	assert(h_out <= dst.shape[0] && "Dst shape[0] too small");
	assert(w_out <= dst.shape[1] && "Dst shape[1] too small");
	assert(NK <= dst.shape[2] && "Dst shape[2] too small");
	*/

	for (u32 t = 0; t < NK; t++) {
		// Create a view of the kernel
		auto kernel_weight = TensorView<K_T, KD0, KD1, KD2>(k, {t});
		
		for (u32 i = 0; i < h_out; i++) {
			for (u32 j = 0; j < w_out; j++) {

				// Gather a local flat copy of input
				i32 ii = i * stride_i - pad_i;
				i32 jj = j * stride_j - pad_j;
				auto aux = padded_lr_buffer_2d<KD0, KD1>(x, {ii,ii+KD0-1}, {jj,jj+KD1-1});
				
				dst[{i,j,t}] = fact(D_T(b[{t}]) + fkernel(kernel_weight, aux));
			}
		}
	}
}

template<u32 PD0, u32 PD1, u32 XD0, u32 XD1, u32 NCH, typename D_T, typename X_T>
constexpr void nn_pooling2d_iter(
	Tensor<D_T, XD0 / PD0, XD1 / PD1, NCH>& dst,
	Tensor<X_T, XD0, XD1, NCH>& x,
	D_T (fpool) (Tensor<X_T, PD0, PD1>&)
) {

	TensorBuffer<X_T, PD0, PD1> pool_buff;
	
	for (u32 t = 0; t < NCH; t++) {
		for (u32 i = 0; i < XD0 / PD0; i++) {
			for (u32 j = 0; j < XD1 / PD1; j++) {

				// Gaher the values for the pool function
				for(u32 ki = 0; ki < PD0; ki++) {
					for(u32 kj = 0; kj < PD1; kj++) {
						u32 ii = i * PD0 + ki;
						u32 jj = j * PD1 + kj;
						pool_buff[{ki,kj}] = x[{ii,jj,t}];
					}
				}

				dst[{i,j,t}] = fpool(pool_buff);
			}
		}
	}
}

template <typename X_T, u32... DIMS>
constexpr void softmax(Tensor<X_T, DIMS...>& x, Tensor<X_T, DIMS...>& dst) {
	X_T max = x.max();
	X_T denom = 0;

	for (u32 i = 0; i < x.total_size; i++) {
		dst.buffptr[i] = Tensorlib::exp(x.buffptr[i] - max);
		denom += dst.buffptr[i];
	}
	if (denom != X_T(0)) {
		denom = X_T(1) / denom;
		for (u32 i = 0; i < dst.total_size; i++) dst.buffptr[i] *= denom;
	}
}

}

template<typename T, u32 ...DIMS>
constexpr std::ostream& operator<<(std::ostream& os, const Tensor<T,DIMS...>& t) {
	os << "Tensor<";
	for (u32 i = 0; i < t.shape.size(); i++) {
		os << t.shape[i];
		if (i != t.shape.size() - 1) os << ",";
	}
	os << ">[";
	for (u32 i = 0; i < t.total_size; i++) {
		os << t.buffptr[i];
		if (i != t.total_size - 1) os << ", ";
	}
	os << "]";
	return os;
}

