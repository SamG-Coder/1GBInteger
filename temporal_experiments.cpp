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
int main(int argc,char** argv){try{
    uint64_t seed=12345,limbs=125000000,atoms=1024,observations=8192,threads=8;
    std::string source="native",layout="spaced",path;
    for(int i=1;i<argc;i+=2){
        if(i+1>=argc)throw std::runtime_error("Missing value");
        std::string k=argv[i],v=argv[i+1];
        if(k=="--source")source=v;else if(k=="--layout")layout=v;else if(k=="--out")path=v;
        else{if(v.empty()||v[0]=='-')throw std::runtime_error("Unsigned number required");
            size_t used=0;uint64_t n=std::stoull(v,&used);if(used!=v.size())throw std::runtime_error("Invalid number");
            if(k=="--seed")seed=n;else if(k=="--atoms")atoms=n;
            else if(k=="--observations")observations=n;else if(k=="--threads")threads=n;
            else if(k=="--limbs")limbs=n;else throw std::runtime_error("Unknown argument");}
    }
    if((source!="native"&&source!="control")||(layout!="spaced"&&layout!="adjacent")||
       atoms<2||atoms>1000000||atoms%2||atoms>limbs||limbs>12500000000ULL||
       observations<264||observations>1000000||observations%8||threads<1||threads>128||path.empty())
        throw std::runtime_error("Invalid parameters");
    threads=std::min(threads,atoms/2);
    const auto start=std::chrono::steady_clock::now();
    std::vector<Counts> counts(size_t(threads),Counts(size_t(observations/8)));
    std::vector<uint64_t> atom_ones(static_cast<size_t>(atoms));
    auto worker=[&](size_t id){
        auto& c=counts[id];
        std::vector<uint8_t> a(static_cast<size_t>(observations)),b(a.size());
        auto fill=[&](uint64_t j,std::vector<uint8_t>& bits){
            if(source=="native"){
                const uint64_t index=layout=="spaced"?(j*limbs)/atoms:j;
                temporal::sample(seed,index,limbs,observations-1,1,[&](uint64_t t,uint64_t value){bits[size_t(t)]=value&1;});
            }else{
                std::mt19937_64 rng(temporal::mix(seed^(j*temporal::C)));
                uint64_t word=0;
                for(size_t t=0;t<bits.size();++t){if(t%64==0)word=rng();bits[t]=(word>>(t%64))&1;}
            }
            atom_ones[size_t(j)]=analyze(bits,c);
        };
        const uint64_t pairs=atoms/2;
        for(uint64_t pair=id*pairs/threads;pair<(id+1)*pairs/threads;++pair){
            fill(pair*2,a);fill(pair*2+1,b);
            for(size_t t=0;t<a.size();++t){++c.cross_n;c.cross_equal+=a[t]==b[t];}
            for(size_t t=64;t+1<a.size();++t){
                ++c.predictions;c.predicted+=b[t+1]==(b[t]^a[t-63]^a[t-64]);
            }
        }
    };
    std::vector<std::thread> workers;
    for(size_t id=1;id<threads;++id)workers.emplace_back(worker,id);
    worker(0);for(auto& w:workers)w.join();
    Counts total(size_t(observations/8));for(const auto& c:counts)total.add(c);
    const double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream out(path);if(!out)throw std::runtime_error("Cannot open output");
    out<<std::setprecision(17)<<"{\"schema\":1,\"source\":\""<<source<<"\",\"layout\":\""<<layout
       <<"\",\"seed\":\""<<seed<<"\",\"limbs\":"<<limbs<<",\"atoms\":"<<atoms
       <<",\"observations_per_atom\":"<<observations<<",\"threads\":"<<threads
       <<",\"seconds\":"<<sec<<",\"ones\":"<<total.ones;
    auto scalar=[&](const char* name,uint64_t n){out<<",\""<<name<<"\":"<<n;};
    scalar("cross_equal",total.cross_equal);scalar("cross_n",total.cross_n);
    scalar("predicted",total.predicted);scalar("predictions",total.predictions);
    scalar("run_left_censored",total.run_left_censored);scalar("run_right_censored",total.run_right_censored);
    scalar("tail_censored",total.tail_censored);
    auto emit=[&](const char* name,const auto& r){out<<",\""<<name<<"\":";array(out,r);};
    emit("lags",lags);emit("lag_equal",total.lag_equal);emit("lag_n",total.lag_n);
    emit("transitions",total.transitions);emit("blocks",total.blocks);emit("runs",total.runs);
    emit("first",total.first);emit("gaps",total.gaps);emit("atom_ones",atom_ones);
    out<<"}\n";out.flush();if(!out)throw std::runtime_error("Write failed");
    std::cout<<"seconds="<<sec<<" observations="<<atoms*observations<<" out="<<path<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
