#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <iomanip>
static constexpr uint64_t C=0x9e3779b97f4a7c15ULL;
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
static uint64_t at(uint64_t seed,uint64_t idx,uint64_t shift,uint64_t limbs,bool init){
 uint64_t w=(shift/64)%limbs;unsigned bits=unsigned(shift%64);
 uint64_t a=(idx+limbs-w)%limbs,prev=(a+limbs-1)%limbs;
 auto value=[&](uint64_t i){return init?mix(seed+(i+1)*C):mix(seed^(i*C));};
 uint64_t x=value(a);return bits?(x<<bits)|(value(prev)>>(64-bits)):x;
}
struct Atom {uint64_t index,accum=0;unsigned bit;};
struct Stats {uint64_t observations=0,ones=0;uint64_t checksum=0;};
int main(int argc,char**argv){try{
 uint64_t gb=1,mb=0,seed=12345,steps=10000,atoms=1000000,interval=10,threads=std::max(1u,std::thread::hardware_concurrency());
 std::string output="";bool csv=false;
 for(int i=1;i<argc;i++){std::string k=argv[i];if(i+1>=argc)throw std::runtime_error("Missing argument "+k);std::string v=argv[++i];uint64_t n=k=="--out"?0:std::stoull(v);
 if(k=="--gb")gb=n;else if(k=="--mb")mb=n;else if(k=="--seed")seed=n;else if(k=="--steps")steps=n;else if(k=="--atoms")atoms=n;else if(k=="--interval")interval=n;else if(k=="--threads")threads=n;else if(k=="--out"){output=v;csv=true;}else throw std::runtime_error("Unknown argument "+k);}
 uint64_t bytes=mb?mb*1000000ULL:gb*1000000000ULL;
 if(!bytes||bytes%8||bytes>100000000000ULL||!atoms||atoms>10000000||!threads||threads>128||!interval||steps>10000000)throw std::runtime_error("Invalid parameters");
 uint64_t limbs=bytes/8;if(atoms>limbs)throw std::runtime_error("Atom count exceeds integer limbs");
 // CSV is intentionally limited: 1 million atoms x 10k steps can generate terabytes.
 uint64_t observationTimes=steps/interval+1;
 if(csv && atoms>1000000ULL/observationTimes)throw std::runtime_error("CSV output exceeds 1 million rows; omit --out for aggregate-only benchmark");
 std::vector<Atom> state(size_t(atoms));
 for(uint64_t j=0;j<atoms;j++)state[size_t(j)]={(j*limbs)/atoms,0,unsigned(j%64)};
 size_t nthreads=size_t(std::min(threads,atoms));std::vector<Stats> stats(nthreads);
 std::vector<std::thread> workers;workers.reserve(nthreads);
 auto start=std::chrono::steady_clock::now();
 // Each atom has its own independent accumulator; parallelize by atom range.
 // No cross-thread state access or per-step synchronization.
 for(size_t tid=0;tid<nthreads;tid++)workers.emplace_back([&,tid]{
  size_t begin=size_t((uint64_t(tid)*atoms)/nthreads),end=size_t((uint64_t(tid+1)*atoms)/nthreads);
  Stats local;
  for(size_t j=begin;j<end;j++){
   auto &a=state[j];uint64_t acc=0;
   for(uint64_t t=0;t<=steps;t++){
    if(t%interval==0){uint64_t value=at(seed,a.index,t,limbs,true)^acc;uint64_t bit=(value>>a.bit)&1ULL;
      local.observations++;local.ones+=bit;local.checksum^=mix(value^((uint64_t(j)<<32)^t));
    }
    if(t<steps)acc^=at(seed,a.index,t,limbs,false);
   }
   a.accum=acc;
  }
  stats[tid]=local;
 });
 for(auto &worker:workers)worker.join();
 Stats total;for(auto s:stats){total.observations+=s.observations;total.ones+=s.ones;total.checksum^=s.checksum;}
 if(csv){
  // Serial deterministic CSV emission, bounded by one million rows.
  std::ofstream out(output);if(!out)throw std::runtime_error("Cannot open output");
  out<<"step,atom,limb_index,bit_index,measurement,limb_hex\n";
  for(uint64_t t=0;t<=steps;t++)if(t%interval==0)for(uint64_t j=0;j<atoms;j++){
   uint64_t idx=(j*limbs)/atoms,acc=0;
   for(uint64_t k=0;k<t;k++)acc^=at(seed,idx,k,limbs,false);
   uint64_t value=at(seed,idx,t,limbs,true)^acc;
   out<<t<<','<<j<<','<<idx<<','<<(j%64)<<','<<((value>>(j%64))&1ULL)<<','<<std::hex<<value<<std::dec<<'\n';
  }
 }
 double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::cout<<"logical_bytes="<<bytes<<" atoms="<<atoms<<" steps="<<steps<<" interval="<<interval<<" threads="<<nthreads
 <<" observations="<<total.observations<<" ones="<<total.ones<<" checksum="<<std::hex<<total.checksum<<std::dec
 <<" seconds="<<std::setprecision(9)<<sec<<" observations_per_second="<<double(total.observations)/sec<<"\n";
 }catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 1;}
}
