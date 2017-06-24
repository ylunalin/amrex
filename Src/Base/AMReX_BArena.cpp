
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
    // add HEADER_SIZE*sizeof(int) byte to the front as header
    // will store loVect, hiVect and ncomp of this fab here
    std::size_t sz = _sz + HEADER_SIZE;

    // Unpageable (pinned) memory on host is necessary for asynchrounous
    // data transfer between host and device
    cpu_malloc_pinned(&pt, &sz);

    // pt returned points to where the dataPtr of this fab is
    pt = static_cast<void*>(
            static_cast<char*>(pt) + HEADER_SIZE
         );

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
    // cpu_free_pinned(pt);

    // header is at pt - HEADER_SIZE
    cpu_free_pinned(
        static_cast<void*>(
            static_cast<char*>(pt) - HEADER_SIZE
        )
    );
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
#ifdef CUDA_UM
    gpu_malloc(&pt, &_sz);
#else
    std::size_t sz = _sz +  HEADER_SIZE;
    gpu_malloc(&pt, &sz);
    pt = static_cast<void*>(
            static_cast<char*>(pt) + HEADER_SIZE
         );
#endif
#endif

    return pt;
}

void
amrex::BArena::free_device (void* pt)
{
#ifdef CUDA
    gpu_free(
        static_cast<void*>(
            static_cast<char*>(pt) - HEADER_SIZE
        )
    );
#endif
}
