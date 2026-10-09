// SPDX-License-Identifier: MIT
// Check stock Q2_K/Q8_K dot against independently accumulated decoded values,
// including the real 896 -> 1024 zero-padded shape and all-zero inputs.
#include "llama.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
int main(){
    llama_backend_init();
    std::mt19937 rng(731);std::uniform_real_distribution<float> random(-2,2);
    double max_error=0;int cases=0;
    for(int original:{768,896,1024})for(int trial=0;trial<1000;++trial){
        int width=(original+255)/256*256;
        std::vector<float> w(width),a(width),decoded(width);
        for(int i=0;i<original;++i){w[i]=trial==0?0:random(rng);a[i]=trial==1?0:random(rng);}
        std::vector<block_q2_K> qw(width/256);std::vector<block_q8_K> qa(width/256);
        ggml_get_type_traits(GGML_TYPE_Q2_K)->from_float_ref(w.data(),qw.data(),width);
        auto at=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        for(int b=0;b<width/256;++b)at->from_float(a.data()+b*256,&qa[b],256);
        ggml_get_type_traits(GGML_TYPE_Q2_K)->to_float(qw.data(),decoded.data(),width);
        double reference=0,magnitude=0;
        for(int i=0;i<width;++i){
            double x=double(decoded[i])*double(qa[i/256].d)*qa[i/256].qs[i%256];
            reference+=x;magnitude+=std::abs(x);
            if(i>=original && (decoded[i]!=0 || qa[i/256].qs[i%256]!=0))throw std::runtime_error("nonzero padding");
        }
        float result=0;ggml_get_type_traits_cpu(GGML_TYPE_Q2_K)->vec_dot(width,&result,0,qw.data(),0,qa.data(),0,1);
        double error=std::abs(double(result)-reference);max_error=std::max(max_error,error);
        if(!std::isfinite(result) || error>1e-5*std::max(1.,magnitude))throw std::runtime_error("Q2 dot mismatch");
        ++cases;
    }
    std::cout<<"{\"cases\":"<<cases<<",\"padding_zero\":true,\"max_abs_dot_error\":"<<max_error<<"}\n";
    llama_backend_free();
}
