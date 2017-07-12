
#ifndef BL_DEVICE_H
#define BL_DEVICE_H

#include <cstdlib>
#include "AMReX_REAL.H"
#ifdef CUDA
#include <cuda_runtime.h>
#endif

#ifdef CUDA
extern "C" {
    void initialize_cuda();
    void finalize_cuda();
    void gpu_malloc(void** p, const std::size_t* sz);
    void gpu_malloc_managed(void** p, const std::size_t* sz);
    void gpu_free(void* p);
    void cpu_malloc_pinned(void** p, const std::size_t* sz);
    void cpu_free_pinned(void* p);
    void gpu_synchronize();
    void gpu_synchronize_stream(const int* idx);
    void gpu_htod_memcpy_async(void* p_d, void* p_h, const std::size_t* sz, const int* idx);
    void gpu_dtoh_memcpy_async(void* p_h, void* p_d, const std::size_t* sz, const int* idx);
    void gpu_htod_memprefetch_async(void* p, const std::size_t* sz, const int* idx);
    void gpu_dtoh_memprefetch_async(void* p, const std::size_t* sz, const int* idx);
    void mem_advise_set_preferred(void* p, const std::size_t* sz, const int* device);
    void get_cuda_time(int* id, amrex::Real* time);
    void get_cuda_num_calls(int* id, int* ncalls);
    void get_stream(const int* id, cudaStream_t* pStream); 

#ifdef CUDA_ARRAY
    void gpu_malloc_2d(void** p, std::size_t* _pitch, const std::size_t* _isize, const std::size_t* _jsize);
    void gpu_htod_memcpy_2d_async(void* p_d, const std::size_t* pitch_d, void* p_h, const std::size_t* pitch_h, const std::size_t* isize, const std::size_t* jsize, int* idx);
    void gpu_dtoh_memcpy_2d_async(void* p_h, const std::size_t* pitch_h, void* p_d, const std::size_t* pitch_d, const std::size_t* isize, const std::size_t* jsize, int* idx);
#endif // CUDA_ARRAY
}
#endif

namespace amrex {

class Device
{

public:

    static void beginDeviceLaunchRegion() { in_device_launch_region = true; }

    static void endDeviceLaunchRegion() { in_device_launch_region = false; }

    static void setDeviceLaunchRegion(bool r) { in_device_launch_region = r; }

    static bool inDeviceLaunchRegion() {
#ifdef CUDA
      gpu_synchronize();
#endif
      return in_device_launch_region; 
    }

#ifdef CUDA
    static int cudaDeviceId() { return cuda_device_id; }
#endif

private:

    static bool in_device_launch_region;
    static int cuda_device_id;

};

}

#endif
