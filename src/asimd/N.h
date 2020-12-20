#pragma once

#include <cstdint>

namespace asimd {

/**
  like std::integral_constant
*/
template<int n>
struct N {
    enum {                  value          = n };

    operator                std::size_t    () const { return n; }
    operator                int            () const { return n; }

    template<class OS> void write_to_stream( OS &os ) const { os << n; }

    int                     get            () const { return n; }

    template<class T> N     operator=      ( const T & ) const { return *this; }
    template<int m> N       operator=      ( N<m> ) const { static_assert( n == m, "" ); return *this; }

    N<-n>                   operator-      () const { return {}; }
};

template<class T> struct IsN { enum { value = false }; };
template<int n> struct IsN<N<n>> { enum { value = true }; };

template<class T> constexpr bool isN( const T & ) { return false; }
template<int n> constexpr bool isN( N<n> ) { return true; }

template<int n,int m> N<n+m   > operator+ ( N<n>, N<m> ) { return {}; }
template<int n,int m> N<n-m   > operator- ( N<n>, N<m> ) { return {}; }
template<int n,int m> N<n*m   > operator* ( N<n>, N<m> ) { return {}; }
template<int n,int m> N<n/m   > operator/ ( N<n>, N<m> ) { return {}; }
template<int n,int m> N<(n< m)> operator< ( N<n>, N<m> ) { return {}; }
template<int n,int m> N<(n<=m)> operator<=( N<n>, N<m> ) { return {}; }
template<int n,int m> N<(n> m)> operator> ( N<n>, N<m> ) { return {}; }
template<int n,int m> N<(n>=m)> operator>=( N<n>, N<m> ) { return {}; }
template<int n,int m> N<(n==m)> operator==( N<n>, N<m> ) { return {}; }

template<int n,class T> auto operator+ ( N<n>, const T &val ) { return n +  val; }
template<int n,class T> auto operator- ( N<n>, const T &val ) { return n -  val; }
template<int n,class T> auto operator* ( N<n>, const T &val ) { return n *  val; }
template<int n,class T> auto operator/ ( N<n>, const T &val ) { return n /  val; }
template<int n,class T> bool operator< ( N<n>, const T &val ) { return n <  val; }
template<int n,class T> bool operator<=( N<n>, const T &val ) { return n <= val; }
template<int n,class T> bool operator> ( N<n>, const T &val ) { return n >  val; }
template<int n,class T> bool operator>=( N<n>, const T &val ) { return n >= val; }
template<int n,class T> bool operator==( N<n>, const T &val ) { return n == val; }

template<class T,int m> auto operator+ ( const T &val, N<m> ) { return val +  m; }
template<class T,int m> auto operator- ( const T &val, N<m> ) { return val -  m; }
template<class T,int m> auto operator* ( const T &val, N<m> ) { return val *  m; }
template<class T,int m> auto operator/ ( const T &val, N<m> ) { return val /  m; }
template<class T,int m> bool operator< ( const T &val, N<m> ) { return val <  m; }
template<class T,int m> bool operator<=( const T &val, N<m> ) { return val <= m; }
template<class T,int m> bool operator> ( const T &val, N<m> ) { return val >  m; }
template<class T,int m> bool operator>=( const T &val, N<m> ) { return val >= m; }
template<class T,int m> bool operator==( const T &val, N<m> ) { return val == m; }

template<class T,int m> auto &operator+=( T &val, N<m> ) { val += m; return val; }
template<class T,int m> auto &operator-=( T &val, N<m> ) { val -= m; return val; }
template<class T,int m> auto &operator*=( T &val, N<m> ) { val *= m; return val; }
template<class T,int m> auto &operator/=( T &val, N<m> ) { val /= m; return val; }

} // namespace asimd
