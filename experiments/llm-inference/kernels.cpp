// SPDX-License-Identifier: MIT
#include "kernels.h"
#include "llama-model.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
extern "C" { extern int (*integer_llm_hook)(const ggml_compute_params *, ggml_tensor *); }
namespace experiment {
// Signed two's complement planes: q4 = sum(p_i*2^i, i<3)-8*p_3.
Block pack(const int8_t * q, float scale) {
    Block b{}; b.scale = scale;
    float ab = 0;
    for (int j=0;j<32;++j) {
        b.u[j] = q[j]+8; ab += std::abs(int(q[j]));
        for(int k=0;k<4;++k) b.planes[k] |= ((uint8_t(q[j])>>k)&1u)<<j;
    }
    b.binary_scale = ab/32*scale;
    // Magnitude threshold fixed in advance at half mean absolute weight.
    const float threshold=ab/64;
    int count=0; float selected=0;
    for(int j=0;j<32;++j) {
        if(q[j]>=0) b.positive |= 1u<<j;
        if(std::abs(int(q[j]))>threshold) {
            b.negative |= 1u<<j; // nonzero mask, despite historical field name
            ++count; selected += std::abs(int(q[j]));
        }
    }
    b.ternary_scale = count ? scale*selected/count : 0;
    return b;
}
Activation pack_activation(const int8_t * q, float scale) {
    Activation a{}; a.scale=scale; std::memcpy(a.q,q,32);
    __m256i v = _mm256_loadu_si256((const __m256i*)q);
    for(int k=7;k>=0;--k) {
        a.planes[k] = (uint32_t)_mm256_movemask_epi8(v);
        v = _mm256_add_epi8(v,v);
    }
    // The Q4 zero point is subtracted once per activation, reused by every row.
    __m256i corr = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(8),
            _mm256_loadu_si256((const __m256i*)q)), _mm256_set1_epi16(1));
    _mm256_storeu_si256((__m256i*)a.correction,corr);
    return a;
}
int bit_dot(const Block & b,const Activation & a) {
    int sum=0;
    for(int i=0;i<4;++i) for(int j=0;j<8;++j)
        sum += (i==3?-8:1<<i)*(j==7?-128:1<<j)*__builtin_popcount(b.planes[i]&a.planes[j]);
    return sum;
}
int binary_dot(const Block & b,const Activation & a) {
    int sum=0;
    for(int j=0;j<8;++j) sum += (j==7?-128:1<<j)*
        (2*__builtin_popcount(b.positive&a.planes[j])-__builtin_popcount(a.planes[j]));
    return sum;
}
int ternary_dot(const Block & b,const Activation & a) {
    int sum=0;
    for(int j=0;j<8;++j) { auto v=a.planes[j]&b.negative;
        sum += (j==7?-128:1<<j)*(2*__builtin_popcount(b.positive&v)-__builtin_popcount(v)); }
    return sum;
}
template<class B> static float simd_impl(const B * b,const Activation * a,int n) {
    __m256 acc=_mm256_setzero_ps();
    for(int i=0;i<n;++i) {
        auto x=_mm256_loadu_si256((const __m256i*)b[i].u);
        auto y=_mm256_loadu_si256((const __m256i*)a[i].q);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        auto dot=_mm256_dpbusd_epi32(_mm256_setzero_si256(),x,y);
#else
        auto dot=_mm256_madd_epi16(_mm256_maddubs_epi16(x,y),_mm256_set1_epi16(1));
#endif
        dot=_mm256_sub_epi32(dot,_mm256_loadu_si256((const __m256i*)a[i].correction));
        acc=_mm256_fmadd_ps(_mm256_set1_ps(b[i].scale*a[i].scale),_mm256_cvtepi32_ps(dot),acc);
    }
    __m128 s=_mm_add_ps(_mm256_castps256_ps128(acc),_mm256_extractf128_ps(acc,1));
    s=_mm_add_ps(s,_mm_movehl_ps(s,s)); s=_mm_add_ss(s,_mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
float simd_dot(const Block * b,const Activation * a,int n){return simd_impl(b,a,n);}
float simd_dot(const Expanded * b,const Activation * a,int n){return simd_impl(b,a,n);}
template<bool INLINE_FP16> static float packed_impl(const void * raw,const Activation * a,int n) {
    const auto * b=(const block_q4_0*)raw;__m256 acc=_mm256_setzero_ps();
    for(int i=0;i<n;++i) {
        auto q=_mm_loadu_si128((const __m128i*)b[i].qs);
        auto x=_mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(q),_mm_srli_epi16(q,4),1),_mm256_set1_epi8(15));
        auto y=_mm256_loadu_si256((const __m256i*)a[i].q);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        auto dot=_mm256_dpbusd_epi32(_mm256_setzero_si256(),x,y);
#else
        auto dot=_mm256_madd_epi16(_mm256_maddubs_epi16(x,y),_mm256_set1_epi16(1));
#endif
        dot=_mm256_sub_epi32(dot,_mm256_loadu_si256((const __m256i*)a[i].correction));
        float scale;
#if defined(__F16C__)
        if constexpr(INLINE_FP16)scale=_cvtsh_ss(b[i].d);else scale=ggml_fp16_to_fp32(b[i].d);
#else
        scale=ggml_fp16_to_fp32(b[i].d);
#endif
        acc=_mm256_fmadd_ps(_mm256_set1_ps(scale*a[i].scale),_mm256_cvtepi32_ps(dot),acc);
    }
    __m128 s=_mm_add_ps(_mm256_castps256_ps128(acc),_mm256_extractf128_ps(acc,1));
    s=_mm_add_ps(s,_mm_movehl_ps(s,s));s=_mm_add_ss(s,_mm_movehdup_ps(s));return _mm_cvtss_f32(s);
}
float packed_dot(const void * b,const Activation * a,int n){return packed_impl<true>(b,a,n);}
float packed_dot_call(const void * b,const Activation * a,int n){return packed_impl<false>(b,a,n);}
struct Matrix { int cols, rows;const void * raw=nullptr; std::vector<Block> wide;std::vector<Expanded> expanded;std::vector<Planes> planes;std::vector<Binary> binary; };
static std::unordered_map<const ggml_tensor*,Matrix> weights;
static std::string mode;
static std::atomic<uint64_t> call_count{0};
uint64_t calls() { return call_count.load(); }
void reset_calls() { call_count=0; }
uint64_t storage_bytes() { uint64_t n=0;for(auto & v:weights){const auto & m=v.second;n+=m.wide.size()*sizeof(Block)+m.expanded.size()*sizeof(Expanded)+m.planes.size()*sizeof(Planes)+m.binary.size()*sizeof(Binary);}return n; }
static int hook(const ggml_compute_params * p,ggml_tensor * dst) {
    if(dst->op!=GGML_OP_MUL_MAT) return 0;
    const auto & frozen=weights;
    auto it=frozen.find(dst->src[0]); if(it==frozen.end())return 0;
    auto * input=dst->src[1];
    if(input->type!=GGML_TYPE_F32 || !ggml_is_contiguous(input) || input->ne[2]!=1 || input->ne[3]!=1 || !ggml_is_contiguous(dst))return 0;
    if((mode=="prepack" || mode=="prepack-wide" || mode=="packed" || mode=="packed-call") && input->ne[1]!=1)return 0; // retain upstream prompt GEMM
    const Matrix & m=it->second; int nb=m.cols/32;
    thread_local std::vector<block_q8_0> q8;
    thread_local std::vector<Activation> a;
    q8.resize(nb);a.resize(nb); // capacity reused across tokens, no steady-state allocations
    if(p->ith==0)++call_count;
    for(int col=0;col<input->ne[1];++col) {
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float((const float*)input->data+col*m.cols,q8.data(),m.cols);
        for(int i=0;i<nb;++i)a[i]=pack_activation(q8[i].qs,ggml_fp16_to_fp32(q8[i].d));
        const int begin=m.rows*p->ith/p->nth,end=m.rows*(p->ith+1)/p->nth;
        for(int row=begin;row<end;++row) {
            const size_t offset=size_t(row)*nb; float sum=0;
            if(mode=="prepack")sum=simd_dot(m.expanded.data()+offset,a.data(),nb);
            else if(mode=="packed")sum=packed_dot((const block_q4_0*)m.raw+offset,a.data(),nb);
            else if(mode=="packed-call")sum=packed_dot_call((const block_q4_0*)m.raw+offset,a.data(),nb);
            else if(mode=="prepack-wide")sum=simd_dot(m.wide.data()+offset,a.data(),nb);
            else for(int i=0;i<nb;++i) {
                if(mode=="bitplane") {
                    const auto & b=m.planes[offset+i];int dot=0;
                    for(int k=0;k<4;++k)for(int j=0;j<8;++j)dot+=(k==3?-8:1<<k)*(j==7?-128:1<<j)*__builtin_popcount(b.p[k]&a[i].planes[j]);
                    sum+=b.scale*a[i].scale*dot;
                } else {
                    const auto & b=m.binary[offset+i];int dot=0;
                    for(int j=0;j<8;++j){auto mask=a[i].planes[j]&b.nonzero;dot+=(j==7?-128:1<<j)*(2*__builtin_popcount(b.positive&mask)-__builtin_popcount(mask));}
                    sum+=b.scale*a[i].scale*dot;
                }
            }
            ((float*)dst->data)[col*m.rows+row]=sum;
        }
    }
    return 1;
}
void prepare(llama_model * model,const std::string & requested) {
    weights.clear();mode=requested;integer_llm_hook=nullptr;
    if(mode=="stock" || mode=="control")return;
    if(mode!="prepack" && mode!="prepack-wide" && mode!="packed" && mode!="packed-call" && mode!="bitplane" && mode!="binary" && mode!="ternary")throw std::runtime_error("unknown mode");
    for(auto & entry:llama_internal_get_tensor_map(model)) {
        auto * t=entry.second;
        if(t->type!=GGML_TYPE_Q4_0 || entry.first=="token_embd.weight" || (mode!="prepack" && mode!="prepack-wide" && mode!="packed" && mode!="packed-call" && entry.first!="blk.0.ffn_up.weight"))continue;
        if(t->ne[2]!=1 || t->ne[3]!=1 || !ggml_is_contiguous(t))continue;
        Matrix m{};m.cols=int(t->ne[0]);m.rows=int(t->ne[1]);
        m.raw=t->data;
        if(mode=="packed" || mode=="packed-call"){weights.emplace(t,std::move(m));continue;}
        int nb=m.cols/32;size_t count=size_t(nb)*m.rows;
        if(mode=="prepack")m.expanded.reserve(count);
        else if(mode=="prepack-wide")m.wide.reserve(count);
        else if(mode=="bitplane")m.planes.reserve(count);
        else m.binary.reserve(count);
        std::vector<block_q4_0> raw(size_t(nb)*m.rows);
        ggml_backend_tensor_get(t,raw.data(),0,raw.size()*sizeof(block_q4_0));
        for(auto & b:raw) { int8_t q[32];for(int j=0;j<16;++j){q[j]=(b.qs[j]&15)-8;q[j+16]=(b.qs[j]>>4)-8;}
            auto w=pack(q,ggml_fp16_to_fp32(b.d));
            if(mode=="prepack"){Expanded e{};e.scale=w.scale;std::memcpy(e.u,w.u,32);m.expanded.push_back(e);}
            else if(mode=="prepack-wide")m.wide.push_back(w);
            else if(mode=="bitplane"){Planes e{};e.scale=w.scale;std::memcpy(e.p,w.planes,16);m.planes.push_back(e);}
            else m.binary.push_back({mode=="binary"?w.binary_scale:w.ternary_scale,w.positive,mode=="binary"?UINT32_MAX:w.negative});
        }
        weights.emplace(t,std::move(m));
    }
    if(weights.empty())throw std::runtime_error("no eligible Q4_0 tensors; verify model and buffer overrides");
    integer_llm_hook=hook;
}
}
