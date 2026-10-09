// SPDX-License-Identifier: MIT
#include "kernels.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-cpu/quants.h"
#include <random>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
int main(){
    llama_backend_init(); // Initialize upstream FP16 lookup tables before oracle calls.
    if(!experiment::supported()){std::cout<<"Unsupported ISA: stock fallback selected\n";return 0;}
    std::mt19937 rng(20261010);int cases=0;
    for(int nb:{1,3,28,64})for(int trial=0;trial<1000;++trial){
        std::vector<block_q8_0> w(nb*4),a(nb);
        for(auto & b:w){b.d=ggml_fp32_to_fp16(1);for(auto & q:b.qs)q=int(rng()%256)-128;}
        for(auto & b:a){b.d=ggml_fp32_to_fp16(1);for(auto & q:b.qs)q=int(rng()%256)-128;}
        float out[4];experiment::dot4(w.data(),a.data(),nb,out);
        for(int r=0;r<4;++r){int ref=0;for(int b=0;b<nb;++b)for(int j=0;j<32;++j)ref+=int(w[r*nb+b].qs[j])*a[b].qs[j];if(out[r]!=ref)throw std::runtime_error("integer oracle mismatch");++cases;}
        // Normal quantizer domain and nontrivial block scales: exact upstream
        // accumulation order, including positive and negative weight scales.
        for(auto & b:w){b.d=ggml_fp32_to_fp16((int(rng()%200)-100)/128.f);for(auto & q:b.qs)q=int(rng()%255)-127;}
        for(auto & b:a){b.d=ggml_fp32_to_fp16((rng()%100+1)/256.f);for(auto & q:b.qs)q=int(rng()%255)-127;}
        experiment::dot4(w.data(),a.data(),nb,out);
        for(int r=0;r<4;++r){float ref;ggml_vec_dot_q8_0_q8_0(nb*32,&ref,0,w.data()+r*nb,0,a.data(),0,1);if(std::memcmp(&ref,&out[r],4)){std::cerr<<"nb "<<nb<<" trial "<<trial<<" row "<<r<<" ref "<<std::hexfloat<<ref<<" got "<<out[r]<<"\n";throw std::runtime_error("upstream float mismatch");}++cases;}
    }
    std::cout<<cases<<" Q8 oracle/upstream cases passed\n";
}
