// SPDX-License-Identifier: MIT
#include "kernels.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-backend.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <atomic>
#include <cstring>
#include <vector>
#include <stdexcept>
extern "C" { extern int (*integer_llm_hook)(const ggml_compute_params *,ggml_tensor *); }
namespace experiment {
struct Prepared {block_q8_0 q;int32_t correction[8];};
static std::vector<Prepared> scratch;
static std::atomic<uint64_t> count{0};
static bool decode_phase=false;
static bool four_rows=true;
static const ggml_tensor * target=nullptr;
bool supported(){
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}
#define ISA __attribute__((target("avx2,avx512vnni,avx512vl,f16c,fma")))
ISA static void prepare_block(Prepared & p){
    auto q=_mm256_loadu_si256((const __m256i*)p.q.qs);
    auto c=_mm256_dpbusd_epi32(_mm256_setzero_si256(),_mm256_set1_epi8(char(128)),q);
    _mm256_storeu_si256((__m256i*)p.correction,c);
}
ISA static float horizontal(__m256 a){
    __m128 s=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    s=_mm_add_ps(s,_mm_movehl_ps(s,s));s=_mm_add_ss(s,_mm_movehdup_ps(s));return _mm_cvtss_f32(s);
}
// XOR turns signed Q8 weights into unsigned (w+128). Subtract the shared
// 128*sum(activation) correction. No saturation or assumption excluding -128.
template<int ROWS> ISA static void tiled(const block_q8_0 * w,const Prepared * a,int nb,float * out){
    __m256 acc[ROWS];for(auto & v:acc)v=_mm256_setzero_ps();
    const auto sign=_mm256_set1_epi8(char(128));
    for(int b=0;b<nb;++b){
        auto y=_mm256_loadu_si256((const __m256i*)a[b].q.qs);
        auto correction=_mm256_loadu_si256((const __m256i*)a[b].correction);
        float dy=_cvtsh_ss(a[b].q.d);
        for(int row=0;row<ROWS;++row){
            const auto & x=w[row*nb+b];
            auto u=_mm256_xor_si256(_mm256_loadu_si256((const __m256i*)x.qs),sign);
            auto dot=_mm256_dpbusd_epi32(_mm256_setzero_si256(),u,y);
            dot=_mm256_sub_epi32(dot,correction);
            acc[row]=_mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(x.d)*dy),_mm256_cvtepi32_ps(dot),acc[row]);
        }
    }
    for(int row=0;row<ROWS;++row)out[row]=horizontal(acc[row]);
}
void dot4(const void * w,const void * y,int nb,float * out){
    if(!supported())throw std::runtime_error("VNNI unavailable");
    std::vector<Prepared> a(nb);for(int i=0;i<nb;++i){a[i].q=((const block_q8_0*)y)[i];prepare_block(a[i]);}
    tiled<4>((const block_q8_0*)w,a.data(),nb,out);
}
static int hook(const ggml_compute_params * p,ggml_tensor * dst){
    if(!decode_phase || dst->op!=GGML_OP_MUL_MAT || dst->src[0]!=target || std::strcmp(dst->name,"result_output"))return 0;
    const auto * x=dst->src[1];const auto * w=dst->src[0];
    if(x->type!=GGML_TYPE_F32 || x->ne[1]!=1 || x->ne[2]!=1 || x->ne[3]!=1 || !ggml_is_contiguous(x) || !ggml_is_contiguous(dst) || dst->type!=GGML_TYPE_F32)return 0;
    const int nb=int(w->ne[0]/32),rows=int(w->ne[1]);
    // Partition activation blocks once across the existing worker pool.
    auto from=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    for(int b=p->ith;b<nb;b+=p->nth){from((const float*)x->data+b*32,&scratch[b].q,32);prepare_block(scratch[b]);}
    ggml_barrier(p->threadpool);
    const auto * raw=(const block_q8_0*)w->data;
    int groups=rows/4,begin=groups*p->ith/p->nth,end=groups*(p->ith+1)/p->nth;
    for(int g=begin;g<end;++g){
        if(four_rows)tiled<4>(raw+size_t(g)*4*nb,scratch.data(),nb,(float*)dst->data+g*4);
        else for(int r=0;r<4;++r)tiled<1>(raw+(size_t(g)*4+r)*nb,scratch.data(),nb,(float*)dst->data+g*4+r);
    }
    if(p->ith==0)++count;
    return 1;
}
void set_decode_phase(bool enabled){decode_phase=enabled;}
uint64_t calls(){return count.load();}
uint64_t storage_bytes(){return 0;} // No copied/expanded weights.
void reset_calls(){count=0;}
void prepare(llama_model * model,const std::string & mode){
    integer_llm_hook=nullptr;target=nullptr;scratch.clear();count=0;decode_phase=false;
    if(mode=="stock")return;
    if(mode!="stock-plus" && mode!="stock-plus-single")throw std::runtime_error("unknown mode");
    four_rows=mode=="stock-plus";
    if(!supported())return; // Unsupported CPUs remain completely stock.
    target=model->get_tensor("output.weight");
    if(!target)target=model->get_tensor("token_embd.weight");
    if(!target || target->type!=GGML_TYPE_Q8_0 || target->ne[0]%32 || target->ne[1]%4 || target->ne[2]!=1 || target->ne[3]!=1 || !ggml_is_contiguous(target) || !target->buffer){target=nullptr;return;}
    // A host buffer alone does not guarantee ordinary layout. Explicitly reject
    // repacked/custom buffers; never reinterpret CPU_REPACK storage as GGUF.
    const char * name=ggml_backend_buft_name(ggml_backend_buffer_get_type(target->buffer));
    if(std::strcmp(name,"CPU") && std::strcmp(name,"CPU_Mapped")){target=nullptr;return;}
    scratch.resize(target->ne[0]/32);integer_llm_hook=hook;
}
}
