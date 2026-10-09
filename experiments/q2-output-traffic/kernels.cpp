// SPDX-License-Identifier: MIT
// Decode-only output compression. All quantizers and dot products are stock GGML.
#include "../stock-llama-integration/kernels.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-backend.h"
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <algorithm>
extern "C" { extern int (*integer_llm_hook)(const ggml_compute_params *,ggml_tensor *); }
namespace experiment {
static const ggml_tensor * target=nullptr;
static std::vector<uint8_t> weights,activation;
static ggml_type type=GGML_TYPE_Q8_0;
static std::atomic<uint64_t> count{0};
static bool decode_phase=false;
static int hook(const ggml_compute_params * p,ggml_tensor * dst){
    if(!decode_phase || dst->op!=GGML_OP_MUL_MAT || dst->src[0]!=target || std::strcmp(dst->name,"result_output"))return 0;
    const auto * x=dst->src[1];
    if(x->type!=GGML_TYPE_F32 || x->ne[1]!=1 || x->ne[2]!=1 || x->ne[3]!=1 || !ggml_is_contiguous(x) || !ggml_is_contiguous(dst) || dst->type!=GGML_TYPE_F32)return 0;
    const int original_width=int(target->ne[0]),rows=int(target->ne[1]);
    const int block=int(ggml_blck_size(type));
    const int width=(original_width+block-1)/block*block;
    const auto * traits=ggml_get_type_traits_cpu(type);
    auto from=ggml_get_type_traits_cpu(traits->vec_dot_type)->from_float;
    const size_t ab=ggml_type_size(traits->vec_dot_type);
    const int activation_block=int(ggml_blck_size(traits->vec_dot_type));
    for(int b=p->ith;b<width/activation_block;b+=p->nth){
        const int offset=b*activation_block;
        if(offset+activation_block<=original_width)from((const float*)x->data+offset,activation.data()+b*ab,activation_block);
        else {
            float padded[256]={};
            std::copy_n((const float*)x->data+offset,original_width-offset,padded);
            from(padded,activation.data()+b*ab,activation_block);
        }
    }
    ggml_barrier(p->threadpool);
    const size_t stride=ggml_row_size(type,width);
    for(int r=rows*p->ith/p->nth;r<rows*(p->ith+1)/p->nth;++r)
        traits->vec_dot(width,(float*)dst->data+r,0,weights.data()+size_t(r)*stride,0,activation.data(),0,1);
    if(p->ith==0)++count;
    return 1;
}
bool supported(){return true;} // Uses this build's stock GGML CPU dispatch.
void set_decode_phase(bool enabled){decode_phase=enabled;}
uint64_t calls(){return count.load();}
uint64_t storage_bytes(){return weights.size();}
void reset_calls(){count=0;}
void prepare(llama_model * model,const std::string & mode){
    integer_llm_hook=nullptr;target=nullptr;weights.clear();activation.clear();count=0;decode_phase=false;
    if(mode=="stock")return;
    if(mode=="output-q8")type=GGML_TYPE_Q8_0;
    else if(mode=="output-q5")type=GGML_TYPE_Q5_0;
    else if(mode=="output-q4")type=GGML_TYPE_Q4_0;
    else if(mode=="output-q2")type=GGML_TYPE_Q2_K;
    else throw std::runtime_error("unknown traffic mode");
    target=model->get_tensor("output.weight");
    if(!target)target=model->get_tensor("token_embd.weight");
    if(!target || target->type!=GGML_TYPE_Q8_0 || target->ne[0]%32 || target->ne[2]!=1 || target->ne[3]!=1 || !ggml_is_contiguous(target) || !target->buffer)throw std::runtime_error("expected contiguous Q8 output");
    const char * name=ggml_backend_buft_name(ggml_backend_buffer_get_type(target->buffer));
    if(std::strcmp(name,"CPU") && std::strcmp(name,"CPU_Mapped"))throw std::runtime_error("unsupported output buffer");
    const auto * traits=ggml_get_type_traits_cpu(type);
    if(traits->vec_dot_type!=GGML_TYPE_Q8_0 && traits->vec_dot_type!=GGML_TYPE_Q8_K)throw std::runtime_error("expected Q8 activation");
    const int block=int(ggml_blck_size(type));
    if(ggml_blck_size(traits->vec_dot_type)>256)throw std::runtime_error("activation block too large");
    if(type==GGML_TYPE_Q2_K && (block!=256 || ggml_type_size(type)!=84))throw std::runtime_error("unexpected Q2_K layout");
    const int original_width=int(target->ne[0]);
    const int width=(original_width+block-1)/block*block;
    const size_t stride=ggml_row_size(type,width);
    weights.resize(size_t(target->ne[1])*stride);
    activation.resize(ggml_row_size(traits->vec_dot_type,width));
    if(type==GGML_TYPE_Q8_0)std::memcpy(weights.data(),target->data,weights.size());
    else {
        std::vector<float> row(width);
        for(int r=0;r<target->ne[1];++r){
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float((const uint8_t*)target->data+size_t(r)*target->nb[1],row.data(),original_width);
            ggml_get_type_traits(type)->from_float_ref(row.data(),weights.data()+size_t(r)*stride,width);
        }
    }
    integer_llm_hook=hook;
}
}
