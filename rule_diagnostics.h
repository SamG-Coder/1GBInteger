#pragma once
// Frozen diagnostic definitions from the first case studies; baseline source is unchanged.
#include "temporal_kernel.h"
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>


namespace rule_diagnostics {
constexpr std::array<size_t,11> lags{1,2,4,8,16,32,63,64,65,128,256};
struct Counts {
    uint64_t ones=0, cross_equal=0, cross_n=0, predicted=0, predictions=0;
    uint64_t run_left_censored=0, run_right_censored=0, tail_censored=0;
    std::array<uint64_t,11> lag_equal{}, lag_n{};
    std::array<uint64_t,4> transitions{};
    std::array<uint64_t,256> blocks{};
    std::array<uint64_t,129> runs{}; // index 128 means >=128
    std::vector<uint64_t> first, gaps;
    explicit Counts(size_t trials):first(trials+1),gaps(trials+1){}
    void add(const Counts& b) {
        ones+=b.ones; cross_equal+=b.cross_equal; cross_n+=b.cross_n;
        predicted+=b.predicted; predictions+=b.predictions;
        run_left_censored+=b.run_left_censored; run_right_censored+=b.run_right_censored;
        tail_censored+=b.tail_censored;
        auto sum=[](auto& a,const auto& other){for(size_t i=0;i<a.size();++i)a[i]+=other[i];};
        sum(lag_equal,b.lag_equal);sum(lag_n,b.lag_n);sum(transitions,b.transitions);
        sum(blocks,b.blocks);sum(runs,b.runs);sum(first,b.first);sum(gaps,b.gaps);
    }
};
uint64_t analyze(const std::vector<uint8_t>& bits,Counts& c) {
    uint64_t ones=0; size_t length=1; bool first_run=true;
    for(size_t t=0;t<bits.size();++t){
        ones+=bits[t];
        if(t){
            ++c.transitions[bits[t-1]*2+bits[t]];
            if(bits[t]==bits[t-1])++length;
            else{
                if(first_run){++c.run_left_censored;first_run=false;}
                else ++c.runs[std::min<size_t>(128,length)];
                length=1;
            }
        }
        for(size_t k=0;k<lags.size();++k)if(t>=lags[k]){
            ++c.lag_n[k];c.lag_equal[k]+=bits[t]==bits[t-lags[k]];
        }
    }
    ++c.run_right_censored;
    if(first_run)++c.run_left_censored; // same run can touch both boundaries
    size_t first=0,last=0;
    const size_t trials=bits.size()/8;
    for(size_t trial=1;trial<=trials;++trial){
        unsigned block=0;
        for(unsigned k=0;k<8;++k)block|=unsigned(bits[(trial-1)*8+k])<<k;
        ++c.blocks[block];
        if(block==255){
            if(!first)first=trial;
            if(last)++c.gaps[trial-last];
            last=trial;
        }
    }
    ++c.first[first]; // zero means no event before the observation horizon
    if(last<trials)++c.tail_censored;
    c.ones+=ones;
    return ones;
}
template<class Range> void array(std::ostream& out,const Range& r){
    out<<'[';bool comma=false;for(auto v:r){if(comma)out<<',';out<<v;comma=true;}out<<']';
}
}
