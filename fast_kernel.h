#pragma once
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

// Exact same evolution as main.cpp; x86-64 AVX2 kernel, no extra state buffer.
// Compilers emit vpsllq/vpsrlq/vpor/vpxor and vectorized 64-bit multiply.
// 64-bit multiplication modulo 2^64, assembled from 32-bit partial products.
static inline __m256i mul64(__m256i a, __m256i b) {
    const __m256i lo=_mm256_mul_epu32(a,b);
    const __m256i ah=_mm256_srli_epi64(a,32), bh=_mm256_srli_epi64(b,32);
    const __m256i cross=_mm256_add_epi64(_mm256_mul_epu32(ah,b),_mm256_mul_epu32(a,bh));
    return _mm256_add_epi64(lo,_mm256_slli_epi64(cross,32));
}
static inline __m256i hash4(__m256i x) {
    x=_mm256_xor_si256(x,_mm256_srli_epi64(x,30));
    x=mul64(x,_mm256_set1_epi64x(0xbf58476d1ce4e5b9ULL));
    x=_mm256_xor_si256(x,_mm256_srli_epi64(x,27));
    x=mul64(x,_mm256_set1_epi64x(0x94d049bb133111ebULL));
    return _mm256_xor_si256(x,_mm256_srli_epi64(x,31));
}
static inline void evolve_avx2(uint64_t* state,size_t begin,size_t end,uint64_t seed,uint64_t boundary){
    constexpr uint64_t C=0x9e3779b97f4a7c15ULL;
    size_t i=begin;
    uint64_t carry=boundary;
    for(;i+4<=end;i+=4){
        const __m256i cur=_mm256_loadu_si256((const __m256i*)(state+i));
        const __m256i prev=_mm256_set_epi64x((long long)state[i+2],(long long)state[i+1],(long long)state[i],(long long)(carry<<63));
        const __m256i idx=_mm256_set_epi64x((long long)(seed^((i+3)*C)),(long long)(seed^((i+2)*C)),(long long)(seed^((i+1)*C)),(long long)(seed^(i*C)));
        const __m256i rotated=_mm256_or_si256(_mm256_slli_epi64(cur,1),_mm256_srli_epi64(prev,63));
        const __m256i next=_mm256_xor_si256(rotated,hash4(idx));
        carry=(state[i+3]>>63);
        _mm256_storeu_si256((__m256i*)(state+i),next);
    }
    for(;i<end;i++){
        uint64_t old=state[i];uint64_t x=seed^(uint64_t(i)*C);
        x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;x^=x>>31;
        state[i]=((old<<1)|carry)^x;carry=old>>63;
    }
}
