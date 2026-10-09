#pragma once
#include "temporal_kernel.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace inference {
using Bits=std::vector<uint64_t>;
inline Bits zeros(size_t n){return Bits((n+63)/64);}
inline bool bit(const Bits& a,size_t i){return (a[i/64]>>(i%64))&1;}
inline void flip(Bits& a,size_t i){a[i/64]^=uint64_t(1)<<(i%64);}
inline unsigned parity(const Bits& a,const Bits& b){unsigned p=0;for(size_t i=0;i<a.size();++i)p^=unsigned(__builtin_parityll(a[i]&b[i]));return p;}
inline int highest(const Bits& a){for(size_t i=a.size();i-->0;)if(a[i])return int(i*64+63-unsigned(__builtin_clzll(a[i])));return -1;}
struct Query {bool unique=false,value=false;};
class Solver {
    size_t n_;std::vector<Bits> basis_;std::vector<uint8_t> rhs_;
public:
    size_t rank=0,conflicts=0;
    explicit Solver(size_t n):n_(n),basis_(n),rhs_(n){}
    void add(Bits row,bool rhs){
        for(int p=highest(row);p>=0;p=highest(row)){
            if(basis_[size_t(p)].empty()){basis_[size_t(p)]=std::move(row);rhs_[size_t(p)]=rhs;++rank;return;}
            for(size_t k=0;k<row.size();++k)row[k]^=basis_[size_t(p)][k];
            rhs=rhs!=bool(rhs_[size_t(p)]);
        }
        conflicts+=rhs;
    }
    Query query(Bits row)const{
        if(conflicts)return {};
        bool rhs=false;
        for(int p=highest(row);p>=0;p=highest(row)){
            if(basis_[size_t(p)].empty())return {};
            for(size_t k=0;k<row.size();++k)row[k]^=basis_[size_t(p)][k];
            rhs=rhs!=bool(rhs_[size_t(p)]);
        }
        return {true,rhs};
    }
    Bits particular()const{
        if(conflicts)throw std::runtime_error("Inconsistent observations have no exact model");
        auto x=zeros(n_);
        for(size_t p=0;p<n_;++p)if(!basis_[p].empty()&&(parity(basis_[p],x)!=rhs_[p]))flip(x,p);
        return x;
    }
};
inline Bits observation(size_t n,size_t site,uint64_t time,bool unknown_mask){
    Bits row=zeros(unknown_mask?2*n:n);
    flip(row,(site+n-size_t(time%n))%n);
    if(unknown_mask){
        if((time/n)&1)for(size_t i=0;i<n;++i)flip(row,n+i);
        for(size_t k=0;k<time%n;++k)flip(row,n+(site+n-k)%n);
    }
    return row;
}
inline bool mask_offset(const Bits& mask,size_t n,size_t site,uint64_t time){
    bool value=false;
    if((time/n)&1)for(auto w:mask)value=value!=bool(__builtin_parityll(w));
    for(size_t k=0;k<time%n;++k)value=value!=bit(mask,(site+n-k)%n);
    return value;
}
inline void step(Bits& state,const Bits& mask){
    uint64_t carry=state.back()>>63;
    for(size_t i=0;i<state.size();++i){uint64_t old=state[i];state[i]=((old<<1)|carry)^mask[i];carry=old>>63;}
}
inline unsigned task(uint16_t x,unsigned id){
    auto b=[&](unsigned i){return unsigned((x>>i)&1);};
    switch(id){
        case 0:return unsigned(__builtin_parity(unsigned(x)&0x88a5u))^1u;
        case 1:return b(0)&b(1);
        case 2:return b(0)+b(1)+b(2)>=2;
        case 3:return b(0)?b(2):b(1);
        case 4:return (((x&7)+((x>>3)&7))>>2)&1;
        case 5:return (((x&7)+((x>>3)&7))>>3)&1;
        default:throw std::runtime_error("Unknown task");
    }
}
struct Features {
    std::string name;std::vector<uint16_t> masks;bool reservoir=false;
    explicit Features(std::string kind):name(std::move(kind)){
        reservoir=name=="reservoir";
        if(reservoir)return;
        unsigned degree=name=="linear"?1:name=="quadratic"?2:name=="cubic"?3:0;
        if(!degree)throw std::runtime_error("Unknown feature bank");
        masks.push_back(0);
        for(unsigned i=0;i<16;++i)masks.push_back(uint16_t(1u<<i));
        if(degree>=2)for(unsigned i=0;i<16;++i)for(unsigned j=i+1;j<16;++j)masks.push_back(uint16_t((1u<<i)|(1u<<j)));
        if(degree>=3)for(unsigned i=0;i<16;++i)for(unsigned j=i+1;j<16;++j)for(unsigned k=j+1;k<16;++k)masks.push_back(uint16_t((1u<<i)|(1u<<j)|(1u<<k)));
    }
    size_t size()const{return reservoir?65:masks.size();}
    Bits encode(uint16_t x)const{
        auto row=zeros(size());
        if(reservoir){
            uint64_t state=0;
            for(unsigned i=0;i<64;++i){
                const uint16_t projection=i<16?uint16_t(1u<<i):uint16_t(temporal::mix((i+1)*temporal::C));
                state|=uint64_t(__builtin_parity(unsigned(x&projection)))<<i;
            }
            for(unsigned t=0;t<17;++t)state=((state<<1)|(state>>63))^0xd1b54a32d192ed03ULL;
            row[0]=1|(state<<1);row[1]=state>>63;
        }else for(size_t i=0;i<masks.size();++i)if((x&masks[i])==masks[i])flip(row,i);
        return row;
    }
};
} // namespace inference
