
#include <AMReX_BArena.H>
#include <AMReX_Device.H>

void*
amrex::BArena::alloc (std::size_t _sz)
{
    void* pt;

#ifdef CUDA
#ifdef CUDA_UM
    gpu_malloc_managed(&pt, &_sz);
    const int device = Device::cudaDeviceId();
    mem_advise_set_preferred(&pt, &_sz, &device);
#else
    // Unpageable (pinned) memory on host is necessary for asynchrounous
    // data transfer between host and device
    cpu_malloc_pinned(&pt, &_sz);
#endif // CUDA_UM
#else
    pt = operator new(_sz);
#endif

    return pt;
}

void
amrex::BArena::free (void* pt)
{
#ifdef CUDA
#ifdef CUDA_UM
    gpu_free(pt);
#else
    cpu_free_pinned(pt);
#endif // CUDA_UM
#else
    operator delete(pt);
#endif // CUDA
}

void*
amrex::BArena::alloc_device (std::size_t _sz)
{
    void* pt = 0;

#ifdef CUDA
    gpu_malloc(&pt, &_sz);
#endif

    return pt;
}

void
amrex::BArena::free_device (void* pt)
{
#ifdef CUDA
    gpu_free(pt);
#endif
}
