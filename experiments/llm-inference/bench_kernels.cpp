// SPDX-License-Identifier: MIT
#include "kernels.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cmath>
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
int main(int argc,char ** argv)try{
    if(argc!=4)throw std::runtime_error("model.gguf real-activation.f32 output.json");
    llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=false;
    auto * model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("model");
    const auto * t=model->get_tensor("blk.0.ffn_up.weight");if(!t || t->type!=GGML_TYPE_Q4_0)throw std::runtime_error("tensor");
    int n=t->ne[0],rows=t->ne[1],nb=n/32;
    std::vector<float> input(n);std::ifstream file(argv[2],std::ios::binary);file.read((char*)input.data(),n*4);if(file.gcount()!=n*4)throw std::runtime_error("input");
    std::vector<block_q4_0> weights(size_t(rows)*nb);ggml_backend_tensor_get(t,weights.data(),0,weights.size()*sizeof(block_q4_0));
    std::vector<block_q8_0> q8(nb);auto activation_start=Clock::now();
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(input.data(),q8.data(),n);
    std::vector<experiment::Activation> act(nb);for(int i=0;i<nb;++i)act[i]=experiment::pack_activation(q8[i].qs,ggml_fp16_to_fp32(q8[i].d));
    double activation_ms=elapsed(activation_start);auto packing=Clock::now();
    std::vector<experiment::Block> packed(weights.size());std::vector<experiment::Expanded> expanded(weights.size());
    for(size_t i=0;i<weights.size();++i){int8_t w[32];for(int j=0;j<16;++j){w[j]=(weights[i].qs[j]&15)-8;w[j+16]=(weights[i].qs[j]>>4)-8;}packed[i]=experiment::pack(w,ggml_fp16_to_fp32(weights[i].d));expanded[i].scale=packed[i].scale;std::memcpy(expanded[i].u,packed[i].u,32);}
    double packing_ms=elapsed(packing);
    std::vector<float> baseline(rows),result(rows);volatile float checksum=0;
    nlohmann::ordered_json out={{"tensor","blk.0.ffn_up.weight"},{"rows",rows},{"cols",n},{"activation_source",argv[2]},{"activation_pack_ms",activation_ms},{"combined_weight_pack_ms",packing_ms},{"runs",nlohmann::ordered_json::array()}};
    const std::vector<std::string> modes={"upstream","packed","expanded","bitplane","binary","ternary","packed-call"};
    for(int rep=0;rep<7;++rep)for(int k=0;k<int(modes.size());++k){
        int mode=(k+rep)%modes.size();auto start=Clock::now();
        for(int batch=0;batch<5;++batch)for(int row=0;row<rows;++row){
            float sum=0;size_t offset=size_t(row)*nb;
            if(mode==0)ggml_vec_dot_q4_0_q8_0(n,&sum,0,weights.data()+offset,0,q8.data(),0,1);
            else if(mode==1)sum=experiment::packed_dot(weights.data()+offset,act.data(),nb);
            else if(mode==2)sum=experiment::simd_dot(expanded.data()+offset,act.data(),nb);
            else if(mode==6)sum=experiment::packed_dot_call(weights.data()+offset,act.data(),nb);
            else for(int i=0;i<nb;++i){auto & w=packed[offset+i];auto & a=act[i];
                if(mode==3)sum+=w.scale*a.scale*experiment::bit_dot(w,a);
                else if(mode==4)sum+=w.binary_scale*a.scale*experiment::binary_dot(w,a);
                else sum+=w.ternary_scale*a.scale*experiment::ternary_dot(w,a);}
            result[row]=sum;
        }
        double duration=elapsed(start)/5;checksum=result[0];
        if(rep==0 && mode==0)baseline=result;
        double maxerr=0,sq=0,dot=0,aa=0,bb=0;
        for(int row=0;row<rows;++row){double a=baseline[row],b=result[row],e=a-b;maxerr=std::max(maxerr,std::abs(e));sq+=e*e;dot+=a*b;aa+=a*a;bb+=b*b;}
        out["runs"].push_back({{"mode",modes[mode]},{"repeat",rep},{"batch_ms",duration},{"max_abs_error",maxerr},{"rmse",std::sqrt(sq/rows)},{"cosine",dot/std::sqrt(aa*bb)}});
    }
    out["checksum"]=float(checksum);std::ofstream f(argv[3]);f<<out.dump(2)<<'\n';llama_model_free(model);llama_backend_free();
}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
