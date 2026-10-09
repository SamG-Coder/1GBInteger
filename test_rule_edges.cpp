#include "rule_kernels.h"
#include <iostream>
#include <vector>

int main(){
    // Emit edge-case transitions for an independent Python reference oracle.
    for(auto name:{"xor","add","local"})
    for(uint64_t a:{uint64_t(0),uint64_t(1),uint64_t(1)<<63,UINT64_MAX})
    for(uint64_t b:{uint64_t(0),uint64_t(1),uint64_t(1)<<63,UINT64_MAX}){
        std::vector<uint64_t> s{a,b,a},m{b,a,b},d(3);
        rules::step(rules::parse(name),s,d,m);
        std::cout<<name<<','<<a<<','<<b;
        for(auto x:d)std::cout<<','<<x;
        std::cout<<'\n';
    }
}
