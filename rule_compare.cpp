#include "rule_kernels.h"
#include "rule_diagnostics.h"
#include <algorithm>
#include <chrono>
#include <numeric>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
double cpu_seconds(){
#ifdef _WIN32
    FILETIME create,exit,kernel,user;
    if(!GetProcessTimes(GetCurrentProcess(),&create,&exit,&kernel,&user))throw std::runtime_error("GetProcessTimes failed");
    auto val=[](FILETIME v){return (uint64_t(v.dwHighDateTime)<<32)|v.dwLowDateTime;};
    return double(val(kernel)+val(user))/1e7;
#else
    rusage r{};if(getrusage(RUSAGE_SELF,&r))throw std::runtime_error("getrusage failed");
    return r.ru_utime.tv_sec+r.ru_utime.tv_usec/1e6+r.ru_stime.tv_sec+r.ru_stime.tv_usec/1e6;
#endif
}
uint64_t peak_bytes(){
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS p{};p.cb=sizeof(p);
    if(!GetProcessMemoryInfo(GetCurrentProcess(),&p,sizeof(p)))throw std::runtime_error("GetProcessMemoryInfo failed");
    return p.PeakWorkingSetSize;
#else
    rusage r{};if(getrusage(RUSAGE_SELF,&r))throw std::runtime_error("getrusage failed");
    return uint64_t(r.ru_maxrss)*1024; // Linux runner
#endif
}
unsigned pop(uint64_t x){unsigned c=0;while(x){x&=x-1;++c;}return c;}
uint64_t digest(const std::vector<uint64_t>& s){uint64_t h=0;for(size_t i=0;i<s.size();++i)h^=temporal::mix(s[i]^i);return h;}
void diagnostics(std::ostream& out,const std::vector<uint8_t>& history,size_t atoms,size_t observations){
    using namespace rule_diagnostics;
    Counts c(observations/8);std::vector<uint64_t> ones(atoms);
    std::vector<uint8_t> a(observations),b(observations);
    for(size_t j=0;j<atoms;j+=2){
        std::copy_n(history.begin()+j*observations,observations,a.begin());
        std::copy_n(history.begin()+(j+1)*observations,observations,b.begin());
        ones[j]=analyze(a,c);ones[j+1]=analyze(b,c);
        for(size_t t=0;t<observations;++t){++c.cross_n;c.cross_equal+=a[t]==b[t];}
        for(size_t t=64;t+1<observations;++t){++c.predictions;c.predicted+=b[t+1]==(b[t]^a[t-63]^a[t-64]);}
    }
    out<<"\"ones\":"<<c.ones;
    auto scalar=[&](const char* name,uint64_t n){out<<",\""<<name<<"\":"<<n;};
    auto emit=[&](const char* name,const auto& r){out<<",\""<<name<<"\":";array(out,r);};
    scalar("cross_equal",c.cross_equal);scalar("cross_n",c.cross_n);
    scalar("predicted",c.predicted);scalar("predictions",c.predictions);
    scalar("run_left_censored",c.run_left_censored);scalar("run_right_censored",c.run_right_censored);
    scalar("tail_censored",c.tail_censored);
    emit("lags",lags);emit("lag_equal",c.lag_equal);emit("lag_n",c.lag_n);
    emit("transitions",c.transitions);emit("blocks",c.blocks);emit("runs",c.runs);
    emit("first",c.first);emit("gaps",c.gaps);emit("atom_ones",ones);
}
void graph(std::ostream& out,rules::Rule rule,uint64_t seed,unsigned w,unsigned limbs){
    const size_t n=size_t(1)<<(w*limbs);uint32_t mask=0;
    for(unsigned i=0;i<limbs;++i)mask|=uint32_t(temporal::mix(seed^(i*temporal::C))&((1u<<w)-1))<<(i*w);
    std::vector<uint32_t> next(n),indegree(n),queue;queue.reserve(n);
    for(size_t i=0;i<n;++i){next[i]=rules::reduced(rule,uint32_t(i),mask,w,limbs);++indegree[next[i]];}
    const auto distinct=std::count_if(indegree.begin(),indegree.end(),[](auto x){return x!=0;});
    const uint32_t maxpre=*std::max_element(indegree.begin(),indegree.end());
    for(size_t i=0;i<n;++i)if(!indegree[i])queue.push_back(uint32_t(i));
    for(size_t k=0;k<queue.size();++k)if(--indegree[next[queue[k]]]==0)queue.push_back(next[queue[k]]);
    uint64_t cycles=0,cyclic=0,mincycle=n,maxcycle=0;std::vector<uint8_t> seen(n);
    for(size_t i=0;i<n;++i)if(indegree[i]&&!seen[i]){
        uint64_t length=0;size_t v=i;
        do{seen[v]=1;v=next[v];++length;}while(v!=i);
        ++cycles;cyclic+=length;mincycle=std::min(mincycle,length);maxcycle=std::max(maxcycle,length);
    }
    std::vector<uint32_t> depth(n);
    for(auto it=queue.rbegin();it!=queue.rend();++it)depth[*it]=depth[next[*it]]+1;
    out<<"{\"word_bits\":"<<w<<",\"limbs\":"<<limbs<<",\"states\":"<<n
       <<",\"distinct_images\":"<<distinct<<",\"collision_excess\":"<<n-size_t(distinct)
       <<",\"maximum_preimages\":"<<maxpre<<",\"cycles\":"<<cycles
       <<",\"cyclic_states\":"<<cyclic<<",\"min_cycle\":"<<mincycle<<",\"max_cycle\":"<<maxcycle
       <<",\"max_transient\":"<<*std::max_element(depth.begin(),depth.end());
    if(n<=256){out<<",\"edges\":";rule_diagnostics::array(out,next);}
    out<<'}';
}
int main(int argc,char** argv){try{
    std::string mode="stats",name="xor",path;uint64_t seed=12345,bytes=1000000,steps=8191,atoms=1024;
    for(int i=1;i<argc;i+=2){
        if(i+1>=argc)throw std::runtime_error("Missing argument");
        std::string k=argv[i],v=argv[i+1];
        if(k=="--mode")mode=v;else if(k=="--rule")name=v;else if(k=="--out")path=v;
        else{if(v.empty()||v[0]=='-')throw std::runtime_error("Unsigned value required");
            size_t used=0;auto n=std::stoull(v,&used);if(used!=v.size())throw std::runtime_error("Invalid number");
            if(k=="--seed")seed=n;else if(k=="--bytes")bytes=n;else if(k=="--steps")steps=n;
            else if(k=="--atoms")atoms=n;else throw std::runtime_error("Unknown option");}
    }
    const auto rule=rules::parse(name);
    if(path.empty()||bytes<8||bytes%8||bytes>1000000000||steps>100000||atoms<2||atoms%2||atoms>1000000)
        throw std::runtime_error("Invalid parameters");
    std::ofstream out(path);if(!out)throw std::runtime_error("Cannot open output");out<<std::setprecision(17);
    out<<"{\"rule\":\""<<name<<"\",\"mode\":\""<<mode<<"\",\"seed\":\""<<seed<<"\"";
    if(mode=="graph"){
        out<<",\"graphs\":[";bool comma=false;
        for(auto pair:{std::pair<unsigned,unsigned>{4,1},{4,2},{4,3},{8,1},{8,2}}){
            if(comma)out<<',';comma=true;graph(out,rule,seed,pair.first,pair.second);
        }out<<']';
    }else if(mode=="damage"){
        if(steps>4096)throw std::runtime_error("Damage horizon limited to 4096");
        out<<",\"damage\":[";bool comma=false;
        for(size_t limbs:{1,3,17,256}){
            std::vector<uint64_t> a(limbs),m(limbs),ad(limbs),bd(limbs);rules::init(seed,a,m);
            const auto initial=a;auto b=a;b[0]^=1;
            std::vector<uint64_t> distance,changed;uint64_t first=0;
            for(uint64_t t=0;t<=steps;++t){
                uint64_t h=0,c=0;for(size_t i=0;i<limbs;++i){h+=pop(a[i]^b[i]);c+=a[i]!=b[i];}
                distance.push_back(h);changed.push_back(c);
                if(t&&!first&&a==initial)first=t;
                if(t<steps){rules::step(rule,a,ad,m);a.swap(ad);rules::step(rule,b,bd,m);b.swap(bd);}
            }
            if(comma)out<<',';comma=true;
            out<<"{\"limbs\":"<<limbs<<",\"steps\":"<<steps<<",\"first_return\":"<<first<<",\"hamming\":";
            rule_diagnostics::array(out,distance);out<<",\"changed_limbs\":";rule_diagnostics::array(out,changed);out<<'}';
        }out<<']';
    }else if(mode=="stats"||mode=="bench"||mode=="trace"){
        if(mode=="stats"&&(atoms>bytes/8||steps<263||(steps+1)%8||atoms*(steps+1)>64000000))
            throw std::runtime_error("Invalid statistics size");
        if(mode=="trace"&&(bytes>256||steps>128))throw std::runtime_error("Trace too large");
        const auto init_start=Clock::now();
        std::vector<uint64_t> s(size_t(bytes/8)),d(s.size()),m(s.size());rules::init(seed,s,m);
        const double init_sec=elapsed(init_start);const size_t observations=size_t(steps+1);
        std::array<std::vector<uint8_t>,2> history;
        if(mode=="stats")for(auto& h:history)h.resize(size_t(atoms)*observations);
        std::vector<double> step_seconds;
        if(mode=="trace"){out<<",\"states\":[";rule_diagnostics::array(out,s);}
        const double cpu_start=cpu_seconds();const auto start=Clock::now();
        for(uint64_t t=0;t<=steps;++t){
            if(mode=="stats")for(uint64_t j=0;j<atoms;++j){
                history[0][size_t(j)*observations+size_t(t)]=s[size_t(j*s.size()/atoms)]&1;
                history[1][size_t(j)*observations+size_t(t)]=s[size_t(j)]&1;
            }
            if(t<steps){
                const auto tick=Clock::now();rules::step(rule,s,d,m);s.swap(d);
                if(mode=="bench")step_seconds.push_back(elapsed(tick));
                if(mode=="trace"){out<<',';rule_diagnostics::array(out,s);}
            }
        }
        const double seconds=elapsed(start),cpu=cpu_seconds()-cpu_start;
        if(mode=="trace")out<<']';
        out<<",\"bytes\":"<<bytes<<",\"steps\":"<<steps<<",\"init_seconds\":"<<init_sec
           <<",\"evolution_seconds\":"<<seconds<<",\"evolution_cpu_seconds\":"<<cpu
           <<",\"checksum\":\""<<std::hex<<digest(s)<<std::dec<<"\",\"step_seconds\":";
        rule_diagnostics::array(out,step_seconds);
        if(mode=="stats"){
            out<<",\"layouts\":[";
            for(size_t layout=0;layout<2;++layout){
                if(layout)out<<',';
                out<<"{\"layout\":\""<<(layout?"adjacent":"spaced")<<"\",\"atoms\":"<<atoms
                   <<",\"observations_per_atom\":"<<observations<<',';
                diagnostics(out,history[layout],size_t(atoms),observations);out<<'}';
            }out<<']';
        }
        out<<",\"peak_resident_bytes\":"<<peak_bytes();
    }else throw std::runtime_error("Unknown mode");
    out<<"}\n";out.flush();if(!out)throw std::runtime_error("Write failed");
    std::cout<<mode<<" rule="<<name<<" seed="<<seed<<" bytes="<<bytes<<" out="<<path<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
