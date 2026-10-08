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
#include "temporal_kernel.h"
using temporal::mix;
struct Stats {uint64_t observations=0,ones=0;uint64_t checksum=0;};
int main(int argc,char**argv){try{
 uint64_t gb=1,mb=0,seed=12345,steps=10000,atoms=1000000,interval=10,threads=std::max(1u,std::thread::hardware_concurrency());
 std::string output="";bool csv=false;
 for(int i=1;i<argc;i++){std::string k=argv[i];if(i+1>=argc)throw std::runtime_error("Missing argument "+k);std::string v=argv[++i];uint64_t n=k=="--out"?0:std::stoull(v);
 if(k=="--gb")gb=n;else if(k=="--mb")mb=n;else if(k=="--seed")seed=n;else if(k=="--steps")steps=n;else if(k=="--atoms")atoms=n;else if(k=="--interval")interval=n;else if(k=="--threads")threads=n;else if(k=="--out"){output=v;csv=true;}else throw std::runtime_error("Unknown argument "+k);}
 if(mb>100000 || (!mb && gb>100))throw std::runtime_error("Size out of range");
 uint64_t bytes=mb?mb*1000000ULL:gb*1000000000ULL;
 if(!bytes||bytes%8||bytes>100000000000ULL||!atoms||atoms>10000000||!threads||threads>128||!interval||steps>10000000)throw std::runtime_error("Invalid parameters");
 uint64_t limbs=bytes/8;if(atoms>limbs)throw std::runtime_error("Atom count exceeds integer limbs");
 // CSV is intentionally limited: 1 million atoms x 10k steps can generate terabytes.
 uint64_t observationTimes=steps/interval+1;
 if(csv && atoms>1000000ULL/observationTimes)throw std::runtime_error("CSV output exceeds 1 million rows; omit --out for aggregate-only benchmark");
 std::vector<uint64_t> samples(csv ? size_t(atoms*observationTimes) : 0);
 size_t nthreads=size_t(std::min(threads,atoms));std::vector<Stats> stats(nthreads);
 std::vector<std::thread> workers;workers.reserve(nthreads);
 auto start=std::chrono::steady_clock::now();
 // Each atom has its own independent accumulator; parallelize by atom range.
 // No cross-thread state access or per-step synchronization.
 auto work=[&](size_t tid){
  size_t begin=size_t((uint64_t(tid)*atoms)/nthreads),end=size_t((uint64_t(tid+1)*atoms)/nthreads);
  Stats local;
  for(size_t j=begin;j<end;j++){
   uint64_t observation=0;
   temporal::sample(seed,(uint64_t(j)*limbs)/atoms,limbs,steps,interval,[&](uint64_t t,uint64_t value){
    local.observations++;local.ones+=(value>>(j%64))&1ULL;
    local.checksum^=mix(value^((uint64_t(j)<<32)^t));
    if(csv)samples[j*size_t(observationTimes)+size_t(observation)]=value;
    ++observation;
   });
  }
  stats[tid]=local;
 };
 // The calling thread participates; workers live for their entire atom range.
 for(size_t tid=1;tid<nthreads;tid++)workers.emplace_back(work,tid);
 work(0);
 for(auto &worker:workers)worker.join();
 Stats total;for(auto s:stats){total.observations+=s.observations;total.ones+=s.ones;total.checksum^=s.checksum;}
 if(csv){
  // Serial deterministic CSV emission, bounded by one million rows.
  std::ofstream out(output);if(!out)throw std::runtime_error("Cannot open output");
  out<<"step,atom,limb_index,bit_index,measurement,limb_hex\n";
  for(uint64_t t=0;t<=steps;t++)if(t%interval==0)for(uint64_t j=0;j<atoms;j++){
   uint64_t idx=(j*limbs)/atoms;
   uint64_t value=samples[size_t(j*observationTimes+t/interval)];
   out<<t<<','<<j<<','<<idx<<','<<(j%64)<<','<<((value>>(j%64))&1ULL)<<','<<std::hex<<value<<std::dec<<'\n';
  }
  out.flush();if(!out)throw std::runtime_error("Write failed");
 }
 double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::cout<<"logical_bytes="<<bytes<<" atoms="<<atoms<<" steps="<<steps<<" interval="<<interval<<" threads="<<nthreads
 <<" observations="<<total.observations<<" ones="<<total.ones<<" checksum="<<std::hex<<total.checksum<<std::dec
 <<" seconds="<<std::setprecision(9)<<sec<<" observations_per_second="<<double(total.observations)/sec<<"\n";
 }catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 1;}
}
