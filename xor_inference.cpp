#include "xor_inference.h"
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

using namespace inference;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
constexpr std::array<uint64_t,8> seeds{0,1,42,12345,20261009,314159265,2718281828,UINT64_MAX};
std::string group(size_t i){return i<4?"development":"validation";}
struct Score {uint64_t answered=0,correct=0,total=0;void add(Query q,bool truth){++total;if(q.unique){++answered;correct+=q.value==truth;}};};
void state_case(std::ostream& out,uint64_t seed,const std::string& stage,size_t n,size_t sensors,size_t horizon,
                bool unknown,unsigned noise,unsigned repeats){
    Bits initial=zeros(n),mask=zeros(n);
    for(size_t i=0;i<initial.size();++i){
        initial[i]=temporal::mix(seed^0x243f6a8885a308d3ULL^(i*temporal::C));
        mask[i]=temporal::mix(seed^0x13198a2e03707344ULL^(i*temporal::C));
    }
    std::vector<Bits> truth(horizon+32,initial);
    for(size_t t=1;t<truth.size();++t){truth[t]=truth[t-1];step(truth[t],mask);}
    Solver solver(unknown?2*n:n);std::mt19937_64 rng(seed^0xa4093822299f31d0ULL);
    uint64_t raw_errors=0,voted_errors=0;const auto start=Clock::now();
    for(size_t t=0;t<horizon;++t)for(size_t j=0;j<sensors;++j){
        const size_t site=j*n/sensors;const bool clean=bit(truth[t],site);unsigned votes=0;
        for(unsigned r=0;r<repeats;++r){const bool error=rng()%10000<noise;raw_errors+=error;votes+=clean!=error;}
        const bool observed=votes>repeats/2;voted_errors+=observed!=clean;
        const bool rhs=unknown?observed:observed!=mask_offset(mask,n,site,t);
        solver.add(observation(n,site,t,unknown),rhs);
    }
    const double solve_seconds=seconds(start);Score recovered,recovered_mask,full_forecast,sensor_forecast;
    for(size_t i=0;i<n;++i){
        auto row=zeros(unknown?2*n:n);flip(row,i);recovered.add(solver.query(row),bit(initial,i));
        if(unknown){row=zeros(2*n);flip(row,n+i);recovered_mask.add(solver.query(row),bit(mask,i));}
        auto q=solver.query(observation(n,i,horizon,unknown));
        if(q.unique&&!unknown)q.value=q.value!=mask_offset(mask,n,i,horizon);
        full_forecast.add(q,bit(truth[horizon],i));
    }
    for(size_t t=horizon;t<horizon+32;++t)for(size_t j=0;j<sensors;++j){
        size_t i=j*n/sensors;auto q=solver.query(observation(n,i,t,unknown));
        if(q.unique&&!unknown)q.value=q.value!=mask_offset(mask,n,i,t);
        sensor_forecast.add(q,bit(truth[t],i));
    }
    out<<stage<<','<<seed<<','<<n<<','<<sensors<<','<<horizon<<','<<unknown<<','<<noise<<','<<repeats
       <<','<<sensors*horizon<<','<<solver.rank<<','<<solver.conflicts<<','<<raw_errors<<','<<voted_errors;
    for(const auto& s:{recovered,recovered_mask,full_forecast,sensor_forecast})out<<','<<s.answered<<','<<s.correct<<','<<s.total;
    out<<','<<solve_seconds<<'\n';
}
std::string weights(const Bits& x){std::ostringstream out;out<<std::hex;for(size_t i=0;i<x.size();++i){if(i)out<<';';out<<x[i];}return out.str();}
void learn_case(std::ostream& out,const std::string& stage,uint64_t seed,const Features& features,
                const std::vector<uint16_t>& inputs,const std::vector<Bits>& training,const std::vector<Bits>& test,
                unsigned task_id,size_t count,unsigned noise,unsigned repeats){
    Solver solver(features.size());std::mt19937_64 rng(seed^0x082efa98ec4e6c89ULL);
    std::vector<uint8_t> labels(count);uint64_t ones=0,raw_errors=0,voted_errors=0;
    const auto start=Clock::now();
    for(size_t i=0;i<count;++i){
        const bool clean=task(inputs[i],task_id);unsigned votes=0;
        for(unsigned r=0;r<repeats;++r){const bool error=rng()%10000<noise;raw_errors+=error;votes+=clean!=error;}
        labels[i]=votes>repeats/2;ones+=labels[i];voted_errors+=labels[i]!=clean;
        solver.add(training[i],labels[i]);
    }
    const double train_seconds=seconds(start);const bool majority=ones*2>=count;
    uint64_t baseline=0;for(size_t i=0;i<test.size();++i)baseline+=majority==bool(task(inputs[1024+i],task_id));
    out<<stage<<','<<seed<<','<<features.name<<','<<task_id<<','<<count<<','<<features.size()<<','<<noise<<','<<repeats
       <<','<<solver.rank<<','<<solver.conflicts<<','<<raw_errors<<','<<voted_errors<<','<<baseline<<','<<test.size()<<','<<train_seconds;
    if(solver.conflicts){out<<",,,,,,,\n";return;}
    const auto model=solver.particular();uint64_t train_correct=0,test_correct=0;Score certified;
    for(size_t i=0;i<count;++i)train_correct+=parity(training[i],model)==labels[i];
    for(size_t i=0;i<test.size();++i){
        const bool clean=task(inputs[1024+i],task_id);
        test_correct+=parity(test[i],model)==clean;certified.add(solver.query(test[i]),clean);
    }
    double median=0;uint64_t check=0;
    if(count==1024&&!noise){
        std::array<double,3> timings{};
        for(size_t r=0;r<3;++r){const auto tick=Clock::now();uint64_t h=0;
            for(size_t i=0;i<test.size();++i)h+=uint64_t(parity(features.encode(inputs[1024+i]),model))*(i+1);
            timings[r]=seconds(tick);check^=h;
        }
        std::sort(timings.begin(),timings.end());median=timings[1];
    }
    out<<','<<train_correct<<','<<test_correct<<','<<certified.answered<<','<<certified.correct<<','<<median<<','<<check<<','<<weights(model)<<'\n';
}
void binary_dot(std::ostream& out,uint64_t seed){
    constexpr size_t pairs=4096,words=16,bits=1024;
    std::mt19937_64 rng(seed);std::vector<uint64_t> a(pairs*words),b(a.size());
    std::vector<int8_t> sa(pairs*bits),sb(sa.size());
    for(size_t i=0;i<sa.size();++i){sa[i]=(rng()&1)?1:-1;sb[i]=(rng()&1)?1:-1;}
    const auto packing=Clock::now();
    for(size_t i=0;i<sa.size();++i){if(sa[i]>0)a[i/64]|=uint64_t(1)<<(i%64);if(sb[i]>0)b[i/64]|=uint64_t(1)<<(i%64);}
    const double pack_seconds=seconds(packing);
    std::vector<int> reference(pairs),packed(pairs);std::array<double,3> scalar_times{},packed_times{};
    for(size_t r=0;r<3;++r){
        auto scalar=[&]{const auto t=Clock::now();for(size_t j=0;j<pairs;++j){int dot=0;
            for(size_t k=0;k<bits;++k)dot+=int(sa[j*bits+k])*int(sb[j*bits+k]);reference[j]=dot;}scalar_times[r]=seconds(t);};
        auto binary=[&]{const auto t=Clock::now();for(size_t j=0;j<pairs;++j){unsigned differing=0;
            for(size_t k=0;k<words;++k)differing+=unsigned(__builtin_popcountll(a[j*words+k]^b[j*words+k]));packed[j]=int(bits)-2*int(differing);}packed_times[r]=seconds(t);};
        if(r%2){binary();scalar();}else{scalar();binary();}
        if(reference!=packed)throw std::runtime_error("Packed dot mismatch");
    }
    const int64_t checksum=std::accumulate(reference.begin(),reference.end(),int64_t(0));
    for(size_t r=0;r<3;++r)out<<seed<<','<<r<<','<<pairs<<','<<bits<<','<<pack_seconds<<','<<scalar_times[r]<<','<<packed_times[r]<<','<<checksum<<'\n';
}
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("Usage: xor_inference OUTPUT_DIRECTORY");
    const std::string folder=argv[1];std::ofstream state(folder+"/state.csv"),learn(folder+"/learning.csv"),dot(folder+"/binary-dot.csv");
    if(!state||!learn||!dot)throw std::runtime_error("Cannot open outputs");
    state<<std::setprecision(17);learn<<std::setprecision(17);dot<<std::setprecision(17);
    state<<"group,seed,bits,sensors,horizon,unknown_mask,noise_bp,repeats,equations,rank,conflicts,raw_errors,voted_errors,initial_answered,initial_correct,initial_total,mask_answered,mask_correct,mask_total,full_answered,full_correct,full_total,sensor_answered,sensor_correct,sensor_total,solve_seconds\n";
    learn<<"group,seed,model,task,train_count,features,noise_bp,repeats,rank,conflicts,raw_errors,voted_errors,baseline_correct,test_count,train_seconds,train_correct,test_correct,certified_answered,certified_correct,inference_seconds,checksum,weights_hex\n";
    dot<<"seed,repeat,pairs,bits,packing_seconds,scalar_seconds,packed_seconds,checksum\n";
    for(size_t si=0;si<seeds.size();++si){
        uint64_t seed=seeds[si];const auto stage=group(si);
        for(size_t n:{64,256,1024}){
            for(auto pair:{std::pair<size_t,size_t>{1,n/2},{1,n},{4,n/4},{16,n/16},{n,1}})
                state_case(state,seed,stage,n,pair.first,pair.second,false,0,1);
            for(auto pair:{std::pair<size_t,size_t>{1,n},{1,2*n},{4,n},{n,2}})
                state_case(state,seed,stage,n,pair.first,pair.second,true,0,1);
        }
        for(size_t h:{256,1024})for(unsigned p:{100,500})for(unsigned r:{1,5})state_case(state,seed,stage,256,1,h,false,p,r);
        std::vector<uint16_t> inputs(65536);std::iota(inputs.begin(),inputs.end(),uint16_t(0));std::mt19937_64 shuffle(seed^0x452821e638d01377ULL);
        for(size_t i=inputs.size();i>1;--i)std::swap(inputs[i-1],inputs[size_t(shuffle()%i)]);
        for(const auto& name:{"linear","reservoir","quadratic","cubic"}){
            Features features(name);std::vector<Bits> training(1024),test(8192);
            for(size_t i=0;i<training.size();++i)training[i]=features.encode(inputs[i]);
            for(size_t i=0;i<test.size();++i)test[i]=features.encode(inputs[1024+i]);
            for(unsigned task_id=0;task_id<6;++task_id)for(size_t n:{64,256,1024})learn_case(learn,stage,seed,features,inputs,training,test,task_id,n,0,1);
            if(features.name=="linear")for(unsigned p:{100,500})for(unsigned r:{1,5})learn_case(learn,stage,seed,features,inputs,training,test,0,1024,p,r);
        }
        binary_dot(dot,seed);std::cout<<"Completed inference seed "<<seed<<'\n'<<std::flush;
    }
    state.flush();learn.flush();dot.flush();if(!state||!learn||!dot)throw std::runtime_error("Write failed");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
