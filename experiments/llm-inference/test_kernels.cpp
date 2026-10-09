// SPDX-License-Identifier: MIT
#include "kernels.h"
#include <iostream>
#include <random>
#include <stdexcept>
#include <cmath>
int main(){
    std::mt19937 rng(20261009);uint64_t tests=0;
    for(int trial=0;trial<100000;++trial){
        int8_t w[32],x[32];int dot=0,bdot=0,tdot=0;
        for(int i=0;i<32;++i){w[i]=int(rng()%16)-8;x[i]=int(rng()%256)-128;dot+=w[i]*x[i];bdot+=(w[i]>=0?1:-1)*x[i];}
        auto b=experiment::pack(w,1);auto a=experiment::pack_activation(x,1);
        for(int i=0;i<32;++i)if((b.negative>>i)&1)tdot+=(w[i]>=0?1:-1)*x[i];
        if(experiment::bit_dot(b,a)!=dot || experiment::binary_dot(b,a)!=bdot || experiment::ternary_dot(b,a)!=tdot || experiment::simd_dot(&b,&a,1)!=dot)throw std::runtime_error("integer oracle mismatch");
        ++tests;
    }
    // Every 4-bit by signed 8-bit pair, including -128, repeated in every lane.
    for(int w=-8;w<=7;++w)for(int x=-128;x<=127;++x){
        int8_t ws[32],xs[32];for(int i=0;i<32;++i){ws[i]=w;xs[i]=x;}
        auto b=experiment::pack(ws,1);auto a=experiment::pack_activation(xs,1);
        if(experiment::bit_dot(b,a)!=32*w*x || experiment::simd_dot(&b,&a,1)!=32*w*x)throw std::runtime_error("edge mismatch");++tests;
    }
    std::cout<<tests<<" packed integer oracle cases passed\n";
}
