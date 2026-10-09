// SPDX-License-Identifier: MIT
// Independent packed-limb verifier for the published pair-assembly circuit data.
#include "../../xor_inference.h"
#include "nlohmann/json.hpp"
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <limits>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
using Bits=inference::Bits;
static void require(bool b,const char * message){if(!b)throw std::runtime_error(message);}
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
struct Frame{uint64_t core,cover;int rank;};
struct Op{int a,b;};
static int index(const json & j,int size){int64_t x=j.get<int64_t>();require(x>=0&&x<size,"index out of bounds");return int(x);}
static void xor_bits(Bits & a,const Bits & b){for(size_t k=0;k<a.size();++k)a[k]^=b[k];}
struct Matrix{
    size_t n,stride;std::vector<uint64_t> data;std::vector<size_t> low,high;bool active;
    Matrix(size_t count,bool bounded):n(count),stride((count+63)/64),data(n*stride),low(n),high(n),active(bounded){
        for(size_t i=0;i<n;++i){data[i*stride+i/64]=uint64_t(1)<<(i%64);low[i]=i/64;high[i]=i/64+1;}
    }
    void apply(int a,int b){
        auto * dst=data.data()+size_t(a)*stride;const auto * src=data.data()+size_t(b)*stride;
        if(!active){for(size_t k=0;k<stride;++k)dst[k]^=src[k];return;}
        for(size_t k=low[b];k<high[b];++k)dst[k]^=src[k];
        low[a]=std::min(low[a],low[b]);high[a]=std::max(high[a],high[b]);
        while(low[a]<high[a]&&dst[low[a]]==0)++low[a];
        while(high[a]>low[a]&&dst[high[a]-1]==0)--high[a];
    }
    void verify(size_t v,bool dual){
        // Check every coefficient, including all unused/padding bits.
        for(size_t r=0;r<n;++r)for(size_t k=0;k<stride;++k){
            uint64_t expected=k==r/64?uint64_t(1)<<(r%64):0;
            if((dual&&r<v)||(!dual&&r>=v&&r<2*v)){
                size_t bit=dual?r+v:r-v;
                if(k==bit/64)expected^=uint64_t(1)<<(bit%64);
            }
            require(data[r*stride+k]==expected,"full basis map mismatch");
        }
    }
};
int main(int argc,char ** argv){try{
    require(argc==3,"usage: replay input.json active|dense");
    std::string mode=argv[2];require(mode=="active"||mode=="dense","unknown mode");
    auto total=Clock::now();std::ifstream f(argv[1]);require(bool(f),"cannot open input");json d;f>>d;
    int h=d.at("h"),v=d.at("v"),R=d.at("R");
    require(h>=3&&h<=32&&v==h*(h-1)*(h-2)/6&&R>0&&int64_t(2)*v+R<=100000,"invalid dimensions");
    uint64_t all=(uint64_t(1)<<h)-1;
    std::vector<uint64_t> triples;std::map<std::array<int,3>,int> triple_index;
    for(int a=0;a<h;++a)for(int b=a+1;b<h;++b)for(int c=b+1;c<h;++c){triple_index[{a,b,c}]=int(triples.size());triples.push_back((uint64_t(1)<<a)|(uint64_t(1)<<b)|(uint64_t(1)<<c));}
    std::vector<Frame> frames;std::unordered_map<uint64_t,int> singleton;
    for(auto & x:d.at("frames")){
        require(x.size()==2,"bad frame");uint64_t c=x[0],u=x[1];int bc=__builtin_popcountll(c);
        require(c&&!(c&~u)&&!(u&~all)&&(bc==1||bc==2||(c==u&&bc==3)),"invalid frame");
        if(c==u&& !singleton.count(c))singleton[c]=int(frames.size());
        frames.push_back({c,u,c==u?1:int(__builtin_popcountll(u))-bc});
    }
    std::vector<int> physical(R,-1);std::map<int,uint64_t> hist;std::vector<Bits> symbols(R,inference::zeros(v));
    auto incidence=[&](int s,int g){
        int old=physical[s];const auto & b=frames[g];
        if(old<0)++hist[b.rank];
        else {const auto & a=frames[old];require(!(b.core&~a.core)&&!(a.cover&~b.cover),"frame inclusion failure");++hist[b.rank-a.rank];}
        physical[s]=g;
    };
    std::vector<Op> V,M,J;std::set<int> source_roles,source_ids;
    for(auto it=d.at("sources").begin();it!=d.at("sources").end();++it){
        size_t used=0;int i=std::stoi(it.key(),&used);require(used==it.key().size()&&i>=0&&i<v&&source_ids.insert(i).second,"invalid source");
        int s=index(it.value(),R);require(source_roles.insert(s).second,"duplicate source role");
        auto found=singleton.find(triples[i]);require(found!=singleton.end(),"missing source frame");
        incidence(s,found->second);inference::flip(symbols[s],i);V.push_back({2*v+s,i});
    }
    require(int(V.size())==v,"source count");
    for(auto & x:d.at("ops")){
        require(x.size()==3,"bad op");int a=index(x[0],R),b=index(x[1],R),g=index(x[2],int(frames.size()));require(a!=b,"self XOR");
        incidence(a,g);incidence(b,g);xor_bits(symbols[a],symbols[b]);M.push_back({2*v+a,2*v+b});
    }
    std::vector<Bits> scatter(v,inference::zeros(v));std::set<int> outputs;
    for(auto & x:d.at("outputs")){
        require(x.size()==4,"bad output");int s=index(x[0],R),g=index(x[1],int(frames.size())),common=index(x[2],h);
        require(outputs.insert(s).second,"duplicate output role");incidence(s,g);
        const auto & t=x[3];require(t.size()==1||t.size()==3,"bad triple size");
        uint64_t mask=0;std::array<int,3> tuple{};
        for(size_t k=0;k<t.size();++k){int j=index(t[k],h);require(!(mask&(uint64_t(1)<<j)),"duplicate triple index");mask|=uint64_t(1)<<j;tuple[k]=j;}
        uint64_t center=uint64_t(1)<<common,exclude=mask&~center;
        Bits expected=inference::zeros(v);
        for(int i=0;i<v;++i)if((triples[i]&center)&&!(triples[i]&exclude))inference::flip(expected,i);
        require(symbols[s]==expected,"symbolic output mismatch");const auto & frame=frames[g];
        if(t.size()==1){
            require(frame.core==center&&frame.cover==all,"total output frame");++hist[frame.rank];++hist[h-frame.rank];
            for(int i=0;i<v;++i)if(triples[i]&center)xor_bits(scatter[i],symbols[s]);
        }else{
            require(frame.core==center&&frame.cover==(all^exclude),"triple output frame");++hist[h-1-frame.rank];++hist[1];
            auto it=triple_index.find(tuple);require(it!=triple_index.end(),"unknown triple");xor_bits(scatter[it->second],symbols[s]);
        }
    }
    for(int s=0;s<R;++s){require(physical[s]>=0,"unused physical role");if(!outputs.count(s))++hist[h-frames[physical[s]].rank];}
    for(int i=0;i<v;++i){auto expected=inference::zeros(v);inference::flip(expected,i);require(scatter[i]==expected,"symbolic scatter mismatch");}
    uint64_t mass=0;json histogram=json::object();for(auto [rank,count]:hist){require(rank>=0,"negative rank");mass+=rank*count;histogram[std::to_string(rank)]=count;}
    require(mass==uint64_t(h)*R+uint64_t(h)*(h-1),"rank mass mismatch");
    int n=2*v+R;
    for(auto & x:d.at("scatter")){require(x.size()==2,"bad scatter");int a=index(x[0],n),b=index(x[1],n);require(a!=b,"self scatter");J.push_back({a,b});}
    std::vector<Op> word;word.reserve(4*M.size()+2*J.size()+2*V.size());
    for(int repeat=0;repeat<2;++repeat){word.insert(word.end(),M.begin(),M.end());word.insert(word.end(),J.begin(),J.end());word.insert(word.end(),M.rbegin(),M.rend());word.insert(word.end(),V.begin(),V.end());}
    double prepare_seconds=elapsed(total);auto dirty=Clock::now();
    // Deliberately replay every literal operation, including both reverse copies.
    for(bool dual:{false,true}){
        Matrix state(n,mode=="active");
        if(dual){for(auto it=word.rbegin();it!=word.rend();++it)state.apply(it->b,it->a);}
        else for(auto op:word)state.apply(op.a,op.b);
        state.verify(v,dual);
    }
    double dirty_seconds=elapsed(dirty);
    size_t peak=0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pm{};pm.cb=sizeof(pm);if(GetProcessMemoryInfo(GetCurrentProcess(),&pm,sizeof(pm)))peak=pm.PeakWorkingSetSize;
#endif
    std::cout<<json({{"status","PASS"},{"mode",mode},{"h",h},{"roles",R},{"output_roles",outputs.size()},{"elementary_xors",M.size()},{"word_length",word.size()},{"full_basis_vectors",n},{"rank_mass",mass},{"histogram",histogram},{"prepare_seconds",prepare_seconds},{"dirty_seconds",dirty_seconds},{"total_seconds",elapsed(total)},{"matrix_bytes",uint64_t(n)*((n+63)/64)*8},{"peak_working_set_bytes",peak}}).dump()<<'\n';
}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}}
