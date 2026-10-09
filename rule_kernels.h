#pragma once
#include "temporal_kernel.h"
#include <stdexcept>
#include <string>
#include <vector>

namespace rules {
enum class Rule { Xor, Add, Local };
inline Rule parse(const std::string& name){
    if(name=="xor")return Rule::Xor;if(name=="add")return Rule::Add;
    if(name=="local")return Rule::Local;throw std::runtime_error("Unknown rule");
}
inline uint64_t rot(uint64_t x,unsigned n){return (x<<n)|(x>>(64-n));}
inline void init(uint64_t seed,std::vector<uint64_t>& s,std::vector<uint64_t>& m){
    for(size_t i=0;i<s.size();++i){s[i]=temporal::mix(seed+(i+1)*temporal::C);m[i]=temporal::mix(seed^(i*temporal::C));}
}
// Source and destination must be distinct and nonempty, with matching sizes.
inline void step(Rule rule,const std::vector<uint64_t>& s,std::vector<uint64_t>& d,const std::vector<uint64_t>& m){
    const size_t n=s.size();
    if(rule==Rule::Local){
        for(size_t i=0;i<n;++i)d[i]=rot(s[i]+rot(s[i?i-1:n-1],17)+m[i],23)^s[i+1<n?i+1:0];
    }else if(rule==Rule::Xor){
        uint64_t carry=s.back()>>63;
        for(size_t i=0;i<n;++i){const uint64_t x=s[i];d[i]=((x<<1)|carry)^m[i];carry=x>>63;}
    }else{
        uint64_t rotation_carry=s.back()>>63,addition_carry=0;
        for(size_t i=0;i<n;++i){
            const uint64_t x=s[i],r=(x<<1)|rotation_carry;rotation_carry=x>>63;
            const uint64_t sum=r+m[i],next=sum+addition_carry;
            addition_carry=uint64_t(sum<r || next<sum);d[i]=next;
        }
    }
}
// Exhaustive small-state analogues. Total bit width is at most 16.
inline uint32_t reduced(Rule rule,uint32_t s,uint32_t mask,unsigned w,unsigned limbs){
    const unsigned bits=w*limbs;const uint32_t whole=(1u<<bits)-1,word=(1u<<w)-1;
    const uint32_t r=((s<<1)|(s>>(bits-1)))&whole;
    if(rule==Rule::Xor)return r^mask;
    if(rule==Rule::Add)return (r+mask)&whole;
    auto rw=[&](uint32_t x,unsigned k){k%=w;return ((x<<k)|(x>>((w-k)%w)))&word;};
    auto at=[&](unsigned i){return (s>>(i*w))&word;};
    uint32_t out=0;
    for(unsigned i=0;i<limbs;++i){
        const uint32_t value=rw((at(i)+rw(at(i?i-1:limbs-1),17)+((mask>>(i*w))&word))&word,23)^at(i+1<limbs?i+1:0);
        out|=value<<(i*w);
    }
    return out;
}
} // namespace rules
