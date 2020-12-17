#include "Int.h"
#include "gcd.h"

namespace asimd {

template<class T,int a,int o>
Int<T,a,o>::Int( T value ) : value( value ) {
}

template<class T,int a,int o>
Int<T,a,o>::operator T() const {
    return value;
}

template<class T,int a,int o>
T Int<T,a,o>::get() const {
    return value;
}

//-------------------------------------------------------
template<class T,int a,int o,class U,int b,int p>
auto operator+( const Int<T,a,o> &f, const Int<U,b,p> &that ) {
    constexpr int na = gcd( a, b );
    constexpr int no = ( ( o + p ) % na + na ) % na;
    return Int<T,na,no>( f.value + that.get() );
}

template<class T,int a,int o,class U,int b,int p>
auto operator-( const Int<T,a,o> &f, const Int<U,b,p> &that ) {
    constexpr int na = gcd( a, b );
    constexpr int no = ( ( o - p ) % na + na ) % na;
    return Int<T,na,no>( f.value - that.get() );
}

template<class T,int a,int o,class U,int b,int p>
auto operator*( const Int<T,a,o> &f, const Int<U,b,p> &that ) {
    constexpr int na = gcd( gcd( a * b, o * b ), a * p );
    constexpr int no = o * p;
    return Int<T,na,no>( f.value * that.get() );
}

//-------------------------------------------------------
template<class T,int a,int o,int d>
auto operator+( const Int<T,a,o> &f, N<d> ) {
    constexpr int no = ( ( o + d ) % a + a ) % a;
    return Int<T,a,no>( f.value + d );
}

template<class T,int a,int o,int d>
auto operator-( const Int<T,a,o> &f, N<d> ) {
    constexpr int no = ( ( o - d ) % a + a ) % a;
    return Int<T,a,no>( f.value - d );
}

template<class T,int a,int o,int d>
auto operator*( const Int<T,a,o> &f, N<d> ) {
    return Int<T,a*d,o*d>( f.value * d );
}

template<class T,int a,int o>
auto operator*( const Int<T,a,o> &, N<0> ) {
    return N<0>();
}

//-------------------------------------------------------
template<int d,class T,int a,int o>
auto operator+( N<d>, const Int<T,a,o> &f ) {
    constexpr int no = ( ( o + d ) % a + a ) % a;
    return Int<T,a,no>( f.value + d );
}

template<int d,class T,int a,int o>
auto operator-( N<d>,const Int<T,a,o> &f ) {
    constexpr int no = ( ( d - o ) % a + a ) % a;
    return Int<T,a,no>( d - f.value );
}

template<int d,class T,int a,int o>
auto operator*( N<d>, const Int<T,a,o> &f ) {
    return Int<T,a*d,o*d>( f.value * d );
}

template<class T,int a,int o>
auto operator*( N<0>, const Int<T,a,o> & ) {
    return N<0>();
}

//-------------------------------------------------------
template<class T,int a,int o,class D>
auto operator+( const Int<T,a,o> &f, const D &that ) {
    using R = decltype( f.value + that );
    return Int<R,1,0>( f.value + that );
}

template<class T,int a,int o,class D>
auto operator-( const Int<T,a,o> &f, const D &that ) {
    using R = decltype( f.value - that );
    return Int<R,1,0>( f.value - that );
}

template<class T,int a,int o,class D>
auto operator*( const Int<T,a,o> &f, const D &that ) {
    using R = decltype( f.value * that );
    constexpr int na = gcd( a, o );
    return Int<R,na,0>( f.value * that );
}

//-------------------------------------------------------
template<class D,class T,int a,int o>
auto operator+( const D &that, const Int<T,a,o> &f ) {
    using R = decltype( f.value + that );
    return Int<R,1,0>( f.value + that );
}

template<class D,class T,int a,int o>
auto operator-( const D &that, const Int<T,a,o> &f ) {
    using R = decltype( that - f.value );
    return Int<R,1,0>( that - f.value );
}

template<class D,class T,int a,int o>
auto operator*( const D &that, const Int<T,a,o> &f ) {
    using R = decltype( f.value * that );
    constexpr int na = gcd( a, o );
    return Int<R,na,0>( f.value * that );
}

} // namespace asimd
