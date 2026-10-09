#include "xor_inference.h"
#include <iostream>
#include <random>

using namespace inference;
void require(bool b){if(!b)throw std::runtime_error("Inference regression failed");}
int main(){
    std::mt19937_64 rng(918273);
    // Brute-force all possible assignments: independent of elimination.
    for(unsigned n=1;n<=8;++n)for(unsigned example=0;example<80;++example){
        Solver solver(n);std::vector<std::pair<uint64_t,bool>> equations;
        for(unsigned i=0;i<example%12;++i){const uint64_t a=rng()% (1u<<n);const bool b=rng()&1;equations.push_back({a,b});solver.add(Bits{a},b);}
        std::vector<uint64_t> solutions;
        for(uint64_t x=0;x<(1u<<n);++x){bool valid=true;for(auto e:equations)valid=valid&&(bool(__builtin_parityll(x&e.first))==e.second);if(valid)solutions.push_back(x);}
        require(bool(solver.conflicts)==solutions.empty());
        if(!solutions.empty()){
            auto x=solver.particular();require(std::find(solutions.begin(),solutions.end(),x[0])!=solutions.end());
            require(solutions.size()==(size_t(1)<<(n-solver.rank)));
        }
        for(uint64_t q=0;q<(1u<<n);++q){
            bool unique=!solutions.empty(),value=unique?bool(__builtin_parityll(solutions.front()&q)):false;
            for(auto x:solutions)unique=unique&&(bool(__builtin_parityll(x&q))==value);
            const auto got=solver.query(Bits{q});require(got.unique==unique);if(unique)require(got.value==value);
        }
    }
    // Cross-word elimination, known unique dense system from a triangular basis.
    for(size_t n:{65,137,697,2048}){
        Bits target=zeros(n);for(size_t i=0;i<n;++i)if(rng()&1)flip(target,i);
        Solver s(n);
        for(size_t i=0;i<n;++i){Bits row=zeros(n);flip(row,i);if(i)flip(row,i-1);s.add(row,parity(row,target));}
        require(s.rank==n&&s.particular()==target);
    }
    // Symbolic observations versus independently updated materialized rings.
    for(size_t n:{64,128,256}){
        Bits initial=zeros(n),mask=zeros(n);for(auto& x:initial)x=rng();for(auto& x:mask)x=rng();
        Bits state=initial,both=zeros(2*n);for(size_t i=0;i<n;++i){if(bit(initial,i))flip(both,i);if(bit(mask,i))flip(both,n+i);}
        for(size_t t=0;t<2*n+9;++t){
            for(size_t site=0;site<n;site+=7){
                require(parity(observation(n,site,t,true),both)==bit(state,site));
                require((parity(observation(n,site,t,false),initial)!=mask_offset(mask,n,site,t))==bit(state,site));
            }
            // Out-of-place reference, not inference::step.
            auto next=state;for(size_t i=0;i<state.size();++i)next[i]=((state[i]<<1)|(state[i?i-1:state.size()-1]>>63))^mask[i];state=next;
        }
    }
    // Constant/projection/monomial feature definitions and task labels.
    for(auto name:{"linear","quadratic","cubic"}){
        Features f(name);require(f.size()==(f.name=="linear"?17:f.name=="quadratic"?137:697));
        for(unsigned x=0;x<65536;x+=31){auto row=f.encode(uint16_t(x));for(size_t i=0;i<f.size();++i)require(bit(row,i)==((x&f.masks[i])==f.masks[i]));}
    }
    Features reservoir("reservoir");Solver rank(65);
    for(unsigned x=0;x<1024;++x)rank.add(reservoir.encode(uint16_t(rng())),false);
    require(rank.rank==17);
    Features q("quadratic");Solver learned(q.size());
    for(unsigned x=0;x<65536;x+=13)learned.add(q.encode(uint16_t(x)),task(uint16_t(x),2));
    require(!learned.conflicts);auto weights=learned.particular();
    for(unsigned x=0;x<65536;++x)require(parity(q.encode(uint16_t(x)),weights)==task(uint16_t(x),2));
    for(unsigned x=0;x<65536;++x){
        unsigned sum=(x%8)+((x/8)%8);
        require(task(uint16_t(x),4)==((sum/4)%2));require(task(uint16_t(x),5)==((sum/8)%2));
        unsigned p=1;for(unsigned i:{0,2,5,7,11,15})p^=(x>>i)&1;require(task(uint16_t(x),0)==p);
    }
    std::cout<<"PASS: exhaustive solver/query oracles, cross-word systems, exact observation equations, features and held-out majority truth table\n";
}
