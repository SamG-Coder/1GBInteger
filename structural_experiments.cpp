#include "temporal_kernel.h"
#include <iostream>
#include <vector>

static unsigned popcount(uint64_t x){unsigned n=0;while(x){x&=x-1;++n;}return n;}
int main(){
    std::cout<<"seed,limbs,bits,mask_parity,first_return,period_bound,midpoint_correct,final_return,hamming_min,hamming_max\n";
    for(uint64_t seed:{uint64_t(0),uint64_t(1),uint64_t(42),uint64_t(12345),uint64_t(20261009),
                       uint64_t(314159265),uint64_t(2718281828),UINT64_MAX})
    for(size_t limbs:{1,2,3,7,17}){
        std::vector<uint64_t> initial(limbs),mask(limbs);
        unsigned parity=0;
        for(size_t i=0;i<limbs;++i){
            initial[i]=temporal::mix(seed+(i+1)*temporal::C);
            mask[i]=temporal::mix(seed^(i*temporal::C));parity^=popcount(mask[i])&1;
        }
        auto a=initial,b=initial;b[0]^=1;
        auto step=[&](std::vector<uint64_t>& s){uint64_t carry=s.back()>>63;
            for(size_t i=0;i<s.size();++i){const auto old=s[i];s[i]=((old<<1)|carry)^mask[i];carry=old>>63;}};
        size_t first_return=0;unsigned low=1,high=1;bool midpoint=true;
        for(size_t t=1;t<=128*limbs;++t){
            step(a);step(b);unsigned distance=0;
            for(size_t i=0;i<limbs;++i)distance+=popcount(a[i]^b[i]);
            low=std::min(low,distance);high=std::max(high,distance);
            if(!first_return&&a==initial)first_return=t;
            if(t==64*limbs)for(size_t i=0;i<limbs;++i)
                midpoint=midpoint&&(a[i]==(initial[i]^(parity?UINT64_MAX:0)));
        }
        const bool returned=a==initial;
        std::cout<<seed<<','<<limbs<<','<<64*limbs<<','<<parity<<','<<first_return<<','<<128*limbs
                 <<','<<midpoint<<','<<returned<<','<<low<<','<<high<<'\n';
        if(!midpoint||!returned||low!=1||high!=1)return 1;
    }
}
