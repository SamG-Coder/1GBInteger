#include "fast_kernel.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
static uint64_t splitmix(uint64_t &s){s+=0x9e3779b97f4a7c15ULL;uint64_t z=s;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
class WorkerPool {
 std::mutex mu; std::condition_variable startCv,doneCv;
 std::vector<std::thread> workers;
 std::function<void(size_t)> job;
 size_t pending=0; uint64_t epoch=0; bool stop=false;
 void worker(size_t id){uint64_t seen=0;for(;;){
   std::unique_lock<std::mutex> lk(mu);
   startCv.wait(lk,[&]{return stop||epoch!=seen;});
   if(stop)return;
   seen=epoch;auto fn=job;lk.unlock();fn(id);lk.lock();
   if(--pending==0)doneCv.notify_one();
 }}
public:
 explicit WorkerPool(size_t count){for(size_t i=1;i<count;i++)workers.emplace_back([this,i]{worker(i);});}
 ~WorkerPool(){{std::lock_guard<std::mutex> lk(mu);stop=true;}startCv.notify_all();for(auto &t:workers)t.join();}
 void run(const std::function<void(size_t)>& fn){
   if(workers.empty()){fn(0);return;}
   {std::lock_guard<std::mutex> lk(mu);job=fn;pending=workers.size();++epoch;}
   startCv.notify_all();fn(0);
   std::unique_lock<std::mutex> lk(mu);doneCv.wait(lk,[&]{return pending==0;});
 }
};
int main(int argc,char**argv){try{
 uint64_t gb=1,seed=12345,steps=1,mb=0,sampleBytes=1048576,threads=1;bool fast=false,avx2=false,cacheMask=false,avx2Cache=false,persistent=true;std::string samplePath;
 for(int i=1;i<argc;i++){std::string a=argv[i];if(a=="--legacy-threads"){persistent=false;continue;}if(a=="--fast"){fast=true;continue;}if(a=="--avx2"){fast=true;avx2=true;continue;}if(a=="--cache-mask"){cacheMask=true;fast=true;continue;}if(a=="--avx2-cache"){cacheMask=true;fast=true;avx2Cache=true;continue;}if(i+1==argc)throw std::runtime_error("Missing argument for "+a);std::string v=argv[++i];if(a=="--out")samplePath=v;else{uint64_t n=std::stoull(v,nullptr,0);if(a=="--gb")gb=n;else if(a=="--mb")mb=n;else if(a=="--seed")seed=n;else if(a=="--steps")steps=n;else if(a=="--sample-bytes")sampleBytes=n;else if(a=="--threads")threads=n;else throw std::runtime_error("Unknown option "+a);}}
 if(steps>1000||gb>100||mb>100000||sampleBytes>16000000||threads<1||threads>128)throw std::runtime_error("Parameter out of range");
 uint64_t bytes=mb?mb*1000000ULL:gb*1000000000ULL;if(bytes==0||bytes%8)throw std::runtime_error("Size must be nonzero and multiple of 8");size_t limbs=bytes/8;
 std::unique_ptr<uint64_t[]> state(new uint64_t[limbs]);uint64_t stream=seed;
 auto start=Clock::now();for(size_t i=0;i<limbs;i++)state[i]=splitmix(stream);
 std::cout<<"bytes="<<bytes<<" init_seconds="<<seconds(start)<<"\n";
 std::unique_ptr<uint64_t[]> mask; if(cacheMask){auto t=Clock::now();mask.reset(new uint64_t[limbs]);for(size_t i=0;i<limbs;i++)mask[i]=mix(seed^(uint64_t(i)*0x9e3779b97f4a7c15ULL));std::cout<<"mask_init_seconds="<<seconds(t)<<" extra_bytes="<<bytes<<"\n";}
 size_t workers=std::min<size_t>(threads,limbs);
 std::vector<size_t> cuts(workers+1);
 for(size_t w=0;w<=workers;w++)cuts[w]=size_t((uint64_t(w)*limbs)/workers);
 std::vector<uint64_t> boundary(workers);
 std::unique_ptr<WorkerPool> pool;
 if(fast&&persistent)pool.reset(new WorkerPool(workers));
 for(uint64_t step=0;step<steps;step++){
  auto t=Clock::now();uint64_t ones=0,digest=0;
  if(!fast){uint64_t carry=state[limbs-1]>>63;for(size_t i=0;i<limbs;i++){uint64_t old=state[i];uint64_t next=((old<<1)|carry)^mix(seed^(uint64_t(i)*0x9e3779b97f4a7c15ULL));carry=old>>63;state[i]=next;ones+=__builtin_popcountll(next);digest^=mix(next^i);}}
  else {
   for(size_t w=0;w<workers;w++)boundary[w]=state[cuts[w]?cuts[w]-1:limbs-1]>>63;
   auto work=[&](size_t w){size_t begin=cuts[w],end=cuts[w+1];if(avx2Cache){evolve_avx2_cached(state.get(),mask.get(),begin,end,boundary[w]);return;}if(cacheMask){uint64_t carry=boundary[w];for(size_t i=begin;i<end;i++){uint64_t old=state[i];state[i]=((old<<1)|carry)^mask[i];carry=old>>63;}return;}if(avx2){evolve_avx2(state.get(),begin,end,seed,boundary[w]);return;}uint64_t carry=boundary[w];for(size_t i=begin;i<end;i++){uint64_t old=state[i];uint64_t next=((old<<1)|carry)^mix(seed^(uint64_t(i)*0x9e3779b97f4a7c15ULL));carry=old>>63;state[i]=next;}};
   if(pool)pool->run(work);
   else {std::vector<std::thread> temporary;temporary.reserve(workers-1);
     for(size_t w=1;w<workers;w++)temporary.emplace_back(work,w);
     work(0);for(auto &worker:temporary)worker.join();}
  }
  double elapsed=seconds(t);std::cout<<"step="<<step+1<<" seconds="<<elapsed<<" gb_per_sec_rw="<<(2.0*bytes/1e9)/elapsed;
  if(!fast)std::cout<<" ones_fraction="<<std::setprecision(10)<<double(ones)/double(bytes*8)<<" digest="<<std::hex<<digest<<std::dec;
  std::cout<<"\n";
 }
 if(!samplePath.empty()){std::ofstream out(samplePath,std::ios::binary);if(!out)throw std::runtime_error("Cannot open sample output");size_t n=std::min<uint64_t>(sampleBytes,bytes);size_t count=n/8;std::vector<char> sample(count*8);for(size_t j=0;j<count;j++){uint64_t value=state[(uint64_t(j)*limbs)/count];for(int k=0;k<8;k++)sample[j*8+k]=char(value>>(8*k));}out.write(sample.data(),sample.size());std::cout<<"sample_bytes="<<sample.size()<<" sample_path="<<samplePath<<"\n";}
 }catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 1;}}
