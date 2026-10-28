#pragma once

/*
	Fixed point datatype with low precision and high precision template datatypes
*/

// Check explicit constructor

#include <cassert>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <ostream>

using u32 = uint32_t;
using i32 = int32_t;
using u64 = uint64_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;

template<typename L, typename H, u32 S>
class fxp {
  private:

	enum FP {fp};

	template<typename F>
	constexpr fxp(FP, const F& fval) {
		this->data = L(fval * (L(1) << S));
	}

	template<typename F>
	constexpr F fp_cast() const {
		return F(data) / (L(1) << S); 
	}

  public:

	static constexpr u32 scale = S;
	L data = 0;

	constexpr fxp() {}

	template<typename T>
	constexpr fxp(const T& data) {
		this->data = data << S;
	}

	// Floating point casting and constructors
	constexpr fxp(const f32& fval): fxp(fp, fval) {}
	constexpr fxp(const f64& fval): fxp(fp, fval) {}

	template<typename L2, typename H2, u32 S2>
	constexpr fxp(const fxp<L2,H2,S2>& v) {
		if (S2 >= S) {
			this->data = L(v.data >> (S2 - S));
		}
		else {
			this->data = L(v.data << (S - S2));
		}
	}

	template<typename T>
	constexpr operator T() const {
		return T(data >> S);
	}

	constexpr operator f32() const {
		return fp_cast<f32>();
	}
	constexpr operator f64() const {
		return fp_cast<f64>();
	}

	constexpr fxp operator*(const fxp& b) const {
		fxp r;
		r.data = (H(this->data) * H(b.data)) >> S;
		return r;
	}

	constexpr fxp operator/(const fxp& b) const {
		fxp r;
		r.data = (H(this->data) << S) / H(b.data);
		return r;
	}

	constexpr fxp operator+(const fxp& b) const {
		fxp r;
		r.data = this->data + b.data;
		return r;
	}

	constexpr fxp operator-(const fxp& b) const {
		fxp r;
		r.data = this->data - b.data;
		return r;
	}

	constexpr fxp& operator-=(const fxp& b) {
		*this = *this - b;
		return *this;
	}

	constexpr fxp& operator+=(const fxp& b) {
		*this = *this + b;
		return *this;
	}

	constexpr fxp& operator*=(const fxp& b) {
		*this = *this * b;
		return *this;
	}

	constexpr bool operator<(const fxp& b) const {
		return this->data < b.data;
	}

	constexpr bool operator>(const fxp& b) const {
		return !(*this < b);
	}

	constexpr bool operator==(const fxp& b) const {
		return (this->data == b.data);
	}

	constexpr bool operator!=(const fxp& b) const {
		return !(*this == b);
	}
};

namespace Tensorlib {

using _texp_dt = fxp<i32,i64,28>;

// https://en.wikipedia.org/wiki/Taylor_series
// Calculates exp function using the Taylor series approximation
constexpr _texp_dt _taylor_exp(const _texp_dt& v) {
	
	// Lookup table fxp_28 for (1/i!) i in [2,7]
	constexpr u32 lut_len = 6;
	constexpr _texp_dt lut [lut_len] = {
		_texp_dt(0.5), _texp_dt(0.16666666666666666), _texp_dt(0.041666666666666664), _texp_dt(0.008333333333333333), _texp_dt(0.001388888888888889), _texp_dt(0.0001984126984126984)
	};
	
	_texp_dt numerator = v;
	_texp_dt result = v + _texp_dt(1);
	
	for (u32 i = 0; i < lut_len; i++) {
		numerator *= v;
		result += numerator * lut[i];
	}
	return result;
}

template<typename L, typename H, u32 S>
constexpr fxp<L,H,S> fxp_exp(const fxp<L,H,S>& v) {
	
	// Lookup table fxp_28 for exp(i), i integer in [-19,0]
	constexpr u32 lut_len = 20;
	constexpr _texp_dt lut [lut_len] = {
		_texp_dt(1.0), _texp_dt(0.36787944117144233), _texp_dt(0.1353352832366127), _texp_dt(0.049787068367863944), _texp_dt(0.01831563888873418), _texp_dt(0.006737946999085467), _texp_dt(0.0024787521766663585), _texp_dt(0.0009118819655545162), _texp_dt(0.00033546262790251185), _texp_dt(0.00012340980408667956), _texp_dt(4.5399929762484854e-05), _texp_dt(1.670170079024566e-05), _texp_dt(6.14421235332821e-06), _texp_dt(2.2603294069810542e-06), _texp_dt(8.315287191035679e-07), _texp_dt(3.059023205018258e-07), _texp_dt(1.1253517471925912e-07), _texp_dt(4.139937718785167e-08), _texp_dt(1.522997974471263e-08), _texp_dt(5.602796437537268e-09)
	};

	// Integer part
	i32 ipart = (-v.data) >> v.scale;
	_texp_dt iexp = 0;

	if (ipart >= i32(lut_len) || ipart < 0) return 0;
	else iexp = lut[ipart];

	// Decimal part
	_texp_dt dexp = _taylor_exp(_texp_dt(v + fxp<L,H,S>(ipart)));
	
	return fxp<L,H,S>(iexp * dexp);
}

using _gtable_dt = fxp<i32,i32,20>;

// Lookup tables for the gaussian pdf and cdf in the range [-4, 4]
template <typename T>
constexpr void fxp_gauss_pdf_cdf(T k, T& pdf, T& cdf) {

	constexpr T min_lim = T(-4.0);
	constexpr T max_lim = T(4);
	constexpr u32 step = 42154;
	constexpr u32 offset = 4194304;
	constexpr u32 lut_len = 200;

	constexpr _gtable_dt lut_pdf [lut_len] = {
		_gtable_dt(0.00013), _gtable_dt(0.00016), _gtable_dt(0.00018), _gtable_dt(0.00022), _gtable_dt(0.00025), _gtable_dt(0.00029), _gtable_dt(0.00034), _gtable_dt(0.00040), _gtable_dt(0.00046), _gtable_dt(0.00053), _gtable_dt(0.00062), _gtable_dt(0.00071), _gtable_dt(0.00082), _gtable_dt(0.00094), _gtable_dt(0.00109), _gtable_dt(0.00124), _gtable_dt(0.00143), _gtable_dt(0.00163), _gtable_dt(0.00186), _gtable_dt(0.00212), _gtable_dt(0.00241), _gtable_dt(0.00274), _gtable_dt(0.00311), _gtable_dt(0.00352), _gtable_dt(0.00399), _gtable_dt(0.00450), _gtable_dt(0.00507), _gtable_dt(0.00571), _gtable_dt(0.00641), _gtable_dt(0.00719), _gtable_dt(0.00805), _gtable_dt(0.00900), _gtable_dt(0.01005), _gtable_dt(0.01119), _gtable_dt(0.01245), _gtable_dt(0.01383), _gtable_dt(0.01534), _gtable_dt(0.01699), _gtable_dt(0.01878), _gtable_dt(0.02072), _gtable_dt(0.02283), _gtable_dt(0.02511), _gtable_dt(0.02758), _gtable_dt(0.03024), _gtable_dt(0.03311), _gtable_dt(0.03619), _gtable_dt(0.03949), _gtable_dt(0.04302), _gtable_dt(0.04679), _gtable_dt(0.05081), _gtable_dt(0.05508), _gtable_dt(0.05962), _gtable_dt(0.06443), _gtable_dt(0.06952), _gtable_dt(0.07488), _gtable_dt(0.08053), _gtable_dt(0.08647), _gtable_dt(0.09269), _gtable_dt(0.09920), _gtable_dt(0.10600), _gtable_dt(0.11307), _gtable_dt(0.12043), _gtable_dt(0.12806), _gtable_dt(0.13595), _gtable_dt(0.14409), _gtable_dt(0.15248), _gtable_dt(0.16109), _gtable_dt(0.16992), _gtable_dt(0.17893), _gtable_dt(0.18813), _gtable_dt(0.19747), _gtable_dt(0.20695), _gtable_dt(0.21653), _gtable_dt(0.22619), _gtable_dt(0.23589), _gtable_dt(0.24562), _gtable_dt(0.25533), _gtable_dt(0.26500), _gtable_dt(0.27459), _gtable_dt(0.28407), _gtable_dt(0.29340), _gtable_dt(0.30255), _gtable_dt(0.31148), _gtable_dt(0.32016), _gtable_dt(0.32855), _gtable_dt(0.33661), _gtable_dt(0.34431), _gtable_dt(0.35162), _gtable_dt(0.35851), _gtable_dt(0.36494), _gtable_dt(0.37088), _gtable_dt(0.37632), _gtable_dt(0.38121), _gtable_dt(0.38555), _gtable_dt(0.38931), _gtable_dt(0.39247), _gtable_dt(0.39501), _gtable_dt(0.39693), _gtable_dt(0.39822), _gtable_dt(0.39886), _gtable_dt(0.39886), _gtable_dt(0.39822), _gtable_dt(0.39693), _gtable_dt(0.39501), _gtable_dt(0.39247), _gtable_dt(0.38931), _gtable_dt(0.38555), _gtable_dt(0.38121), _gtable_dt(0.37632), _gtable_dt(0.37088), _gtable_dt(0.36494), _gtable_dt(0.35851), _gtable_dt(0.35162), _gtable_dt(0.34431), _gtable_dt(0.33661), _gtable_dt(0.32855), _gtable_dt(0.32016), _gtable_dt(0.31148), _gtable_dt(0.30255), _gtable_dt(0.29340), _gtable_dt(0.28407), _gtable_dt(0.27459), _gtable_dt(0.26500), _gtable_dt(0.25533), _gtable_dt(0.24562), _gtable_dt(0.23589), _gtable_dt(0.22619), _gtable_dt(0.21653), _gtable_dt(0.20695), _gtable_dt(0.19747), _gtable_dt(0.18813), _gtable_dt(0.17893), _gtable_dt(0.16992), _gtable_dt(0.16109), _gtable_dt(0.15248), _gtable_dt(0.14409), _gtable_dt(0.13595), _gtable_dt(0.12806), _gtable_dt(0.12043), _gtable_dt(0.11307), _gtable_dt(0.10600), _gtable_dt(0.09920), _gtable_dt(0.09269), _gtable_dt(0.08647), _gtable_dt(0.08053), _gtable_dt(0.07488), _gtable_dt(0.06952), _gtable_dt(0.06443), _gtable_dt(0.05962), _gtable_dt(0.05508), _gtable_dt(0.05081), _gtable_dt(0.04679), _gtable_dt(0.04302), _gtable_dt(0.03949), _gtable_dt(0.03619), _gtable_dt(0.03311), _gtable_dt(0.03024), _gtable_dt(0.02758), _gtable_dt(0.02511), _gtable_dt(0.02283), _gtable_dt(0.02072), _gtable_dt(0.01878), _gtable_dt(0.01699), _gtable_dt(0.01534), _gtable_dt(0.01383), _gtable_dt(0.01245), _gtable_dt(0.01119), _gtable_dt(0.01005), _gtable_dt(0.00900), _gtable_dt(0.00805), _gtable_dt(0.00719), _gtable_dt(0.00641), _gtable_dt(0.00571), _gtable_dt(0.00507), _gtable_dt(0.00450), _gtable_dt(0.00399), _gtable_dt(0.00352), _gtable_dt(0.00311), _gtable_dt(0.00274), _gtable_dt(0.00241), _gtable_dt(0.00212), _gtable_dt(0.00186), _gtable_dt(0.00163), _gtable_dt(0.00143), _gtable_dt(0.00124), _gtable_dt(0.00109), _gtable_dt(0.00094), _gtable_dt(0.00082), _gtable_dt(0.00071), _gtable_dt(0.00062), _gtable_dt(0.00053), _gtable_dt(0.00046), _gtable_dt(0.00040), _gtable_dt(0.00034), _gtable_dt(0.00029), _gtable_dt(0.00025), _gtable_dt(0.00022), _gtable_dt(0.00018), _gtable_dt(0.00016), _gtable_dt(0.00013)
	};
	constexpr _gtable_dt lut_cdf [lut_len] = {
		_gtable_dt(0.00003), _gtable_dt(0.00004), _gtable_dt(0.00004), _gtable_dt(0.00005), _gtable_dt(0.00006), _gtable_dt(0.00007), _gtable_dt(0.00009), _gtable_dt(0.00010), _gtable_dt(0.00012), _gtable_dt(0.00014), _gtable_dt(0.00016), _gtable_dt(0.00019), _gtable_dt(0.00022), _gtable_dt(0.00025), _gtable_dt(0.00029), _gtable_dt(0.00034), _gtable_dt(0.00039), _gtable_dt(0.00046), _gtable_dt(0.00053), _gtable_dt(0.00061), _gtable_dt(0.00070), _gtable_dt(0.00080), _gtable_dt(0.00092), _gtable_dt(0.00105), _gtable_dt(0.00120), _gtable_dt(0.00137), _gtable_dt(0.00156), _gtable_dt(0.00178), _gtable_dt(0.00202), _gtable_dt(0.00230), _gtable_dt(0.00260), _gtable_dt(0.00295), _gtable_dt(0.00333), _gtable_dt(0.00375), _gtable_dt(0.00423), _gtable_dt(0.00476), _gtable_dt(0.00534), _gtable_dt(0.00599), _gtable_dt(0.00671), _gtable_dt(0.00750), _gtable_dt(0.00838), _gtable_dt(0.00934), _gtable_dt(0.01040), _gtable_dt(0.01156), _gtable_dt(0.01284), _gtable_dt(0.01423), _gtable_dt(0.01575), _gtable_dt(0.01741), _gtable_dt(0.01921), _gtable_dt(0.02117), _gtable_dt(0.02330), _gtable_dt(0.02560), _gtable_dt(0.02810), _gtable_dt(0.03079), _gtable_dt(0.03369), _gtable_dt(0.03681), _gtable_dt(0.04017), _gtable_dt(0.04377), _gtable_dt(0.04762), _gtable_dt(0.05175), _gtable_dt(0.05615), _gtable_dt(0.06084), _gtable_dt(0.06584), _gtable_dt(0.07114), _gtable_dt(0.07677), _gtable_dt(0.08273), _gtable_dt(0.08903), _gtable_dt(0.09569), _gtable_dt(0.10270), _gtable_dt(0.11007), _gtable_dt(0.11783), _gtable_dt(0.12595), _gtable_dt(0.13447), _gtable_dt(0.14336), _gtable_dt(0.15265), _gtable_dt(0.16233), _gtable_dt(0.17240), _gtable_dt(0.18286), _gtable_dt(0.19371), _gtable_dt(0.20494), _gtable_dt(0.21654), _gtable_dt(0.22852), _gtable_dt(0.24087), _gtable_dt(0.25356), _gtable_dt(0.26660), _gtable_dt(0.27998), _gtable_dt(0.29366), _gtable_dt(0.30765), _gtable_dt(0.32193), _gtable_dt(0.33647), _gtable_dt(0.35126), _gtable_dt(0.36629), _gtable_dt(0.38151), _gtable_dt(0.39693), _gtable_dt(0.41251), _gtable_dt(0.42822), _gtable_dt(0.44405), _gtable_dt(0.45997), _gtable_dt(0.47596), _gtable_dt(0.49198), _gtable_dt(0.50802), _gtable_dt(0.52404), _gtable_dt(0.54003), _gtable_dt(0.55595), _gtable_dt(0.57178), _gtable_dt(0.58749), _gtable_dt(0.60307), _gtable_dt(0.61849), _gtable_dt(0.63371), _gtable_dt(0.64874), _gtable_dt(0.66353), _gtable_dt(0.67807), _gtable_dt(0.69235), _gtable_dt(0.70634), _gtable_dt(0.72002), _gtable_dt(0.73340), _gtable_dt(0.74644), _gtable_dt(0.75913), _gtable_dt(0.77148), _gtable_dt(0.78346), _gtable_dt(0.79506), _gtable_dt(0.80629), _gtable_dt(0.81714), _gtable_dt(0.82760), _gtable_dt(0.83767), _gtable_dt(0.84735), _gtable_dt(0.85664), _gtable_dt(0.86553), _gtable_dt(0.87405), _gtable_dt(0.88217), _gtable_dt(0.88993), _gtable_dt(0.89730), _gtable_dt(0.90431), _gtable_dt(0.91097), _gtable_dt(0.91727), _gtable_dt(0.92323), _gtable_dt(0.92886), _gtable_dt(0.93416), _gtable_dt(0.93916), _gtable_dt(0.94385), _gtable_dt(0.94825), _gtable_dt(0.95238), _gtable_dt(0.95623), _gtable_dt(0.95983), _gtable_dt(0.96319), _gtable_dt(0.96631), _gtable_dt(0.96921), _gtable_dt(0.97190), _gtable_dt(0.97440), _gtable_dt(0.97670), _gtable_dt(0.97883), _gtable_dt(0.98079), _gtable_dt(0.98259), _gtable_dt(0.98425), _gtable_dt(0.98577), _gtable_dt(0.98716), _gtable_dt(0.98844), _gtable_dt(0.98960), _gtable_dt(0.99066), _gtable_dt(0.99162), _gtable_dt(0.99250), _gtable_dt(0.99329), _gtable_dt(0.99401), _gtable_dt(0.99466), _gtable_dt(0.99524), _gtable_dt(0.99577), _gtable_dt(0.99625), _gtable_dt(0.99667), _gtable_dt(0.99705), _gtable_dt(0.99740), _gtable_dt(0.99770), _gtable_dt(0.99798), _gtable_dt(0.99822), _gtable_dt(0.99844), _gtable_dt(0.99863), _gtable_dt(0.99880), _gtable_dt(0.99895), _gtable_dt(0.99908), _gtable_dt(0.99920), _gtable_dt(0.99930), _gtable_dt(0.99939), _gtable_dt(0.99947), _gtable_dt(0.99954), _gtable_dt(0.99961), _gtable_dt(0.99966), _gtable_dt(0.99971), _gtable_dt(0.99975), _gtable_dt(0.99978), _gtable_dt(0.99981), _gtable_dt(0.99984), _gtable_dt(0.99986), _gtable_dt(0.99988), _gtable_dt(0.99990), _gtable_dt(0.99991), _gtable_dt(0.99993), _gtable_dt(0.99994), _gtable_dt(0.99995), _gtable_dt(0.99996), _gtable_dt(0.99996), _gtable_dt(0.99997)
	};

	if (k < min_lim) {
		pdf = T(lut_pdf[0]);
		cdf = T(lut_pdf[0]);
	}
	else if (k > max_lim) {
		pdf = T(lut_pdf[lut_len - 1]);
		cdf = T(lut_pdf[lut_len - 1]);
	}
	else {
		// Get fixed point raw data
		u32 idx = _gtable_dt(k).data;
		idx += offset;
		idx /= step;
		pdf = T(lut_pdf[idx]);
		cdf = T(lut_pdf[idx]);
	}
}

/*
integer sqrt, bit by bit calculation
d << 30 or d << 60 determines data u32 or u64
From: https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Binary_numeral_system_(base_2)
*/
template<typename T>
constexpr T _isqrt(const T& n, T d) {
	T x = n, c = 0;

	while(d > n) d >>= 2;

	while (d != 0) {
		if (x >= c + d) {
			x -= c + d;
			c = (c >> 1) + d;
		}
		else c >>= 1;
		d >>= 2;
	}

	return c;
}

template<typename T>
constexpr T isqrt(const T&) {
	static_assert(false, "Invalid isqrt call");
	return 0;
}

template<>
constexpr i32 isqrt(const i32& n) {
	return _isqrt(u32(n), u32(1) << 30);
}

template<>
constexpr i64 isqrt(const i64& n) {
	return _isqrt(u64(n), u64(1) << 60);
}

template<typename L, typename H, u32 S>
constexpr fxp<L,H,S> fxp_sqrt(const fxp<L,H,S>& v) {
	fxp<L,H,S> r = 0;
	r.data = isqrt(H(v.data) << S);
	return  r;
}

// Generic float implementation
template <typename F>
F exp(const F& f) {return std::exp(f);}

// Fixed point implementation
template<typename L, typename H, u32 S>
fxp<L,H,S> exp(const fxp<L,H,S>& f) {return fxp_exp(f);}

// Generic float implementation
template<typename F>
constexpr F sqrt(const F& v) {return std::sqrt(v);}

// Fixed point implementation
template<typename L, typename H, u32 S>
constexpr fxp<L,H,S> sqrt(const fxp<L,H,S>& v) {return fxp<L,H,S>(fxp_sqrt(v));}

}

template<typename L, typename H, u32 S>
constexpr std::ostream& operator<<(std::ostream& os, const fxp<L,H,S> fx) {
	os << "fxp<" << S << ">(" << f32(fx) << ")";
	return os;
}

