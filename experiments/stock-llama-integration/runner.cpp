// SPDX-License-Identifier: MIT
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include "kernels.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static double cpu_ms(){FILETIME c,e,k,u;GetProcessTimes(GetCurrentProcess(),&c,&e,&k,&u);ULARGE_INTEGER x{},y{};x.LowPart=k.dwLowDateTime;x.HighPart=k.dwHighDateTime;y.LowPart=u.dwLowDateTime;y.HighPart=u.dwHighDateTime;return (x.QuadPart+y.QuadPart)/10000.;}
struct Trace {std::string prefix;json nodes=json::array();int step=-1;Clock::time_point start;bool active=false;};
static bool trace_cb(ggml_tensor * t,bool ask,void * data){
    auto & d=*(Trace*)data;
    if(!d.active)return ask?false:true;
    if(ask){d.start=Clock::now();return true;}
    double elapsed=ms(d.start,Clock::now());
    json node={{"step",d.step},{"name",t->name},{"op",ggml_op_name(t->op)},{"ms",elapsed},{"ne",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}}};
    std::string name=t->name;
    if(d.step==0 && t->op==GGML_OP_MUL_MAT && std::string(t->src[0]->name)=="blk.0.ffn_up.weight" && t->src[1]->type==GGML_TYPE_F32 && ggml_is_contiguous(t->src[1])){
        auto * input=t->src[1];std::vector<float> v(input->ne[0]);
        ggml_backend_tensor_get(input,v.data(),ggml_nbytes(input)-v.size()*4,v.size()*4);
        std::string path=d.prefix+"-activation.f32";std::ofstream f(path,std::ios::binary);f.write((char*)v.data(),v.size()*4);node["input_file"]=path;
    }
    if(d.step<2 && t->type==GGML_TYPE_F32 && ggml_is_contiguous(t) && (name.rfind("l_out-",0)==0 || name.rfind("ffn_out-",0)==0)){
        std::vector<float> v(t->ne[0]);
        ggml_backend_tensor_get(t,v.data(),ggml_nbytes(t)-v.size()*4,v.size()*4);
        std::string path=d.prefix+"-"+std::to_string(d.step)+"-"+name+".f32";
        std::ofstream f(path,std::ios::binary);f.write((char*)v.data(),v.size()*4);node["file"]=path;
    }
    d.nodes.push_back(node);return true;
}
static void decode(llama_context * ctx,llama_batch_ext * b,const std::vector<llama_token>& v,int pos){
    for(size_t offset=0;offset<v.size();){
        llama_batch_ext_clear(b);int count=std::min(size_t(llama_n_batch(ctx)),v.size()-offset);
        for(int k=0;k<count;++k){int i=llama_batch_ext_add_token(b,0,v[offset+k]);if(i<0)throw std::runtime_error("batch overflow");llama_pos p=pos++;llama_batch_ext_set_pos(b,i,&p);}
        llama_batch_ext_set_output_logits(b,count-1,true);
        if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,b))throw std::runtime_error("decode failed");offset+=count;
    }
}
int main(int argc,char ** argv)try{
    std::map<std::string,std::string> opt;
    for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("key value arguments required");opt[argv[i]]=argv[i+1];}
    auto get=[&](std::string k,std::string fallback){return opt.count(k)?opt[k]:fallback;};
    std::string mode=get("--mode","stock"),model_path=opt.at("--model");
    int threads=std::stoi(get("--threads","8")),context=std::stoi(get("--context","2048")),generate=std::stoi(get("--generate","64")),repeat=std::stoi(get("--repeat","3"));
    if(threads<1 || threads>64 || context<128 || generate<2 || repeat<1)throw std::runtime_error("invalid numeric option");
    std::ifstream pf(opt.at("--prompt"),std::ios::binary);if(!pf)throw std::runtime_error("prompt file missing");
    std::string prompt((std::istreambuf_iterator<char>(pf)),{});
    llama_log_set([](ggml_log_level,const char * text,void *){std::cerr<<text;},nullptr);
    llama_backend_init();auto loading=Clock::now();auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=true;
    llama_model * model=llama_model_load_from_file(model_path.c_str(),mp);if(!model)throw std::runtime_error("model load failed");
    double load=ms(loading,Clock::now());auto packing=Clock::now();experiment::prepare(model,mode);double pack=ms(packing,Clock::now());
    const llama_vocab * vocab=llama_model_get_vocab(model);int nv=llama_vocab_n_tokens(vocab);
    int nt=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);if(nt<=0)throw std::runtime_error("empty prompt");
    std::vector<llama_token> tokens(nt);if(llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),nt,true,true)!=nt)throw std::runtime_error("tokenize failed");
    if(nt+generate>context)throw std::runtime_error("context too small");
    std::vector<llama_token> teacher;
    if(opt.count("--teacher")){std::ifstream f(opt.at("--teacher"));json j;f>>j;teacher=j.get<std::vector<llama_token>>();if(int(teacher.size())<generate)throw std::runtime_error("teacher too short");}
    Trace trace;trace.prefix=get("--trace","");
    auto cp=llama_context_default_params();cp.n_ctx=context;cp.n_batch=2048;cp.n_ubatch=512;cp.n_threads=threads;cp.n_threads_batch=threads;cp.no_perf=false;
    if(!trace.prefix.empty()){cp.cb_eval=trace_cb;cp.cb_eval_user_data=&trace;}
    llama_context * ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("context failed");
    auto * batch=llama_batch_ext_init(ctx);
    // Full prompt and 2 decode steps warm caches, thread-local scratch and graph.
    experiment::set_decode_phase(false);decode(ctx,batch,tokens,0);experiment::set_decode_phase(true);decode(ctx,batch,{tokens.back()},nt);decode(ctx,batch,{tokens.back()},nt+1);
    json output={{"mode",mode},{"model",model_path},{"system",llama_print_system_info()},{"threads",threads},{"context",context},{"prompt_tokens",tokens},{"load_ms",load},{"prepack_ms",pack},{"extra_packed_bytes",experiment::storage_bytes()},{"samples",json::array()}};
    std::ofstream logits;
    if(opt.count("--logits")){logits.open(opt.at("--logits"),std::ios::binary);if(!logits)throw std::runtime_error("cannot open logits");}
    for(int r=0;r<repeat;++r){
        llama_memory_clear(llama_get_memory(ctx),true);experiment::reset_calls();
        trace.active=!trace.prefix.empty() && r==0;trace.step=0;
        experiment::set_decode_phase(false);auto start=Clock::now();double cpu0=cpu_ms();decode(ctx,batch,tokens,0);auto pp_end=Clock::now();auto prefill_calls=experiment::calls();experiment::set_decode_phase(true);
        std::vector<llama_token> out;std::string text;double first=0;auto gen_start=pp_end;
        for(int step=0;step<generate;++step){
            float * l=llama_get_logits_ith(ctx,-1);int token=int(std::max_element(l,l+nv)-l);
            for(int k=0;k<nv;++k)if(!std::isfinite(l[k]))throw std::runtime_error("nonfinite logits");
            if(step==0){first=ms(start,Clock::now());gen_start=Clock::now();}
            // Correctness dumps are deliberately excluded from published performance runs.
            if(logits && r==0)logits.write((char*)l,nv*sizeof(float));
            out.push_back(token);char piece[256];int n=llama_token_to_piece(vocab,token,piece,sizeof(piece),0,true);if(n>=0)text.append(piece,n);
            if(step+1<generate){trace.step=step+1;decode(ctx,batch,{teacher.empty()?token:teacher[step]},nt+step);}
        }
        auto end=Clock::now();double elapsed=ms(start,end),cpu=cpu_ms()-cpu0;PROCESS_MEMORY_COUNTERS pm{};pm.cb=sizeof(pm);GetProcessMemoryInfo(GetCurrentProcess(),&pm,sizeof(pm));
        output["samples"].push_back({{"repeat",r},{"pp_ms",ms(start,pp_end)},{"pp_tps",nt*1000/ms(start,pp_end)},{"generation_ms",ms(gen_start,end)},{"generation_tps",(generate-1)*1000/ms(gen_start,end)},{"first_token_ms",first},{"elapsed_ms",elapsed},{"cpu_ms",cpu},{"cpu_machine_percent",100*cpu/(elapsed*GetActiveProcessorCount(ALL_PROCESSOR_GROUPS))},{"peak_working_set_bytes",pm.PeakWorkingSetSize},{"hook_calls",experiment::calls()},{"prefill_hook_calls",prefill_calls},{"tokens",out},{"response",text}});
    }
    output["vocab_size"]=nv;output["instrumented"]=(!trace.prefix.empty()||opt.count("--logits"));output["forced_length"]=generate;output["teacher_forced"]=!teacher.empty();
    std::ofstream f(opt.at("--output"));f<<output.dump(2)<<'\n';if(!f)throw std::runtime_error("write failed");
    if(!trace.prefix.empty()){std::ofstream tf(trace.prefix+"-nodes.json");tf<<trace.nodes.dump(2);}
    llama_batch_ext_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}catch(const std::exception & e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
