//Kokkos profiling hook that counts kernel launches per label.
//
//   cmake --build build --target kp_count
//   KOKKOS_TOOLS_LIBS=build/libkp_count.dylib ./build/spd_K <input>
//
//Prints a launch-count table at finalize, which is how the per-block launch
//overhead is measured (launches scale with block count, work does not).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {
std::map<std::string, uint64_t>& counts(){
    static std::map<std::string, uint64_t> c;
    return c;
}
std::vector<std::string>& regions(){
    static std::vector<std::string> r;
    return r;
}
uint64_t total = 0;

//Attribute each launch to the innermost open region, so the phase-level
//instrumentation in Mesh shows which phase owns the launches.
void bump(const char* name){
    std::string key = regions().empty() ? std::string("(no region)") : regions().back();
    key += " | ";
    key += name ? name : "(anonymous)";
    counts()[key]++;
    total++;
}
}  // namespace

extern "C" {

void kokkosp_init_library(int, uint64_t, uint32_t, void*){
    printf("[kp_count] counting kernel launches\n");
}

void kokkosp_begin_parallel_for(const char* name, uint32_t, uint64_t* id){
    *id = 0; bump(name);
}
void kokkosp_begin_parallel_reduce(const char* name, uint32_t, uint64_t* id){
    *id = 0; bump(name);
}
void kokkosp_begin_parallel_scan(const char* name, uint32_t, uint64_t* id){
    *id = 0; bump(name);
}
void kokkosp_end_parallel_for(uint64_t){}
void kokkosp_end_parallel_reduce(uint64_t){}
void kokkosp_end_parallel_scan(uint64_t){}

void kokkosp_push_profile_region(const char* name){
    regions().push_back(name ? name : "(unnamed)");
}
void kokkosp_pop_profile_region(){
    if(!regions().empty()) regions().pop_back();
}

void kokkosp_finalize_library(){
    std::vector<std::pair<std::string, uint64_t>> v(counts().begin(), counts().end());
    std::sort(v.begin(), v.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });
    printf("\n[kp_count] %llu launches across %zu labels\n",
           (unsigned long long)total, v.size());
    for(const auto& e : v)
        printf("[kp_count] %10llu  %s\n", (unsigned long long)e.second, e.first.c_str());
    if(const char* steps = getenv("KP_STEPS")){
        double n = atof(steps);
        if(n > 0) printf("[kp_count] %.1f launches/step over %g steps\n", total/n, n);
    }
}

}  // extern "C"
