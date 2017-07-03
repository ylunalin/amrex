#include "advance_kernel.H"
#include <cstring>
#include <AMReX_BLFort.H>
#include <AMReX_BArena.H> 
#include <AMReX_Box.H>

#ifdef CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <AMReX_Device.H>
#define BLOCKSIZE_2D 16
#endif

#define ARRAY_2D(PHI, LO_X, LO_Y, HI_X, HI_Y, I, J) PHI[(J-LO_Y)*(HI_X-LO_X+1)+I-LO_X]
#define CUDA_CHECK(x) std::cerr << (x) << std::endl

#ifdef CUDA
__device__
void get_fab_dimension(int& lox, int& loy,
#if (BL_SPACEDIM == 3)
                  int& loz,
#endif
                  int& hix, int& hiy,
#if (BL_SPACEDIM == 3)
                  int& hiz,
#endif
                  void* data_)
{
    char* data = reinterpret_cast<char*>(data_) - HEADER_SIZE;
    std::memcpy(&lox, data + 0 * sizeof(int), sizeof(int));
    std::memcpy(&loy, data + 1 * sizeof(int), sizeof(int));
    std::memcpy(&hix, data + 3 * sizeof(int), sizeof(int));
    std::memcpy(&hiy, data + 4 * sizeof(int), sizeof(int));
#if (BL_SPACEDIM == 3)
    std::memcpy(&loz, data + 2 * sizeof(int), sizeof(int));
    std::memcpy(&hiz, data + 5 * sizeof(int), sizeof(int));
#endif
}
#endif

#ifdef CUDA
__global__
void compute_flux_doit_gpu(int id, void* buffer)
{
    // unpack data
    amrex::Real dt, dx, dy;
    int lox, loy, hix, hiy;
    int phi_old_lox, phi_old_loy, phi_old_hix, phi_old_hiy;
    int phi_new_lox, phi_new_loy, phi_new_hix, phi_new_hiy;
    int fluxx_lox, fluxx_loy, fluxx_hix, fluxx_hiy;
    int fluxy_lox, fluxy_loy, fluxy_hix, fluxy_hiy;
    amrex::Real* phi_old = 0; 
    amrex::Real* phi_new = 0;
    amrex::Real* fluxx   = 0; 
    amrex::Real* fluxy   = 0;
    unpack(id, buffer,
           dt, dx, dy,
           lox, loy, hix, hiy,
           &phi_old, phi_old_lox, phi_old_loy, phi_old_hix, phi_old_hiy,
           &phi_new, phi_new_lox, phi_new_loy, phi_new_hix, phi_new_hiy,
           &fluxx, fluxx_lox, fluxx_loy, fluxx_hix, fluxx_hiy,
           &fluxy, fluxy_lox, fluxy_loy, fluxy_hix, fluxy_hiy);

    // map cuda thread (cudai, cudaj) to cell edge (i,j) it works on 
    int cudai = threadIdx.x + blockDim.x * blockIdx.x;
    int cudaj = threadIdx.y + blockDim.y * blockIdx.y;
    int i = cudai + lox;
    int j = cudaj + loy;

    // TODO
    //debug
    // if (cudai == 0 && cudaj == 0) {
    //     printf("phi_old at: %x \n", phi_old);
    //     printf("phi_new at: %x \n", phi_new);
    //     printf("fluxx at: %x \n", fluxx);
    //     printf("fluxy at: %x \n", fluxy);
    // }

    // compute flux
    // flux in x direction
    if ( i <= (hix+1) && j <= hiy ) {
        ARRAY_2D(fluxx,fluxx_lox,fluxx_loy,fluxx_hix,fluxx_hiy,i,j) = 
            ( ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i,j) - ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i-1,j) ) / dx;
    }
    // flux in y direction
    if ( i <= hix && j <= (hiy+1) ) {
        ARRAY_2D(fluxy,fluxy_lox,fluxy_loy,fluxy_hix,fluxy_hiy,i,j) = 
            ( ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i,j) - ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i,j-1) ) / dy;
    }
}
#endif

void compute_flux_doit_cpu(
            const int& lox, const int& loy, const int& hix, const int& hiy,
            const amrex::Real* __restrict__ phi, const int& phi_lox, const int& phi_loy, const int& phi_hix, const int& phi_hiy,
            amrex::Real* __restrict__ flux, const int& flux_lox, const int& flux_loy, const int& flux_hix, const int& flux_hiy,
            const amrex::Real& dx, const amrex::Real& dy, const int& idir)
{
    if (idir == 1) {// flux in x direction
        // double dxflip = 1 / dx;
        for (int j = loy; j <= hiy; ++j ) {
            for (int i = lox; i <= hix+1; ++i ) {
                ARRAY_2D(flux,flux_lox,flux_loy,flux_hix,flux_hiy,i,j) = 
                    ( ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i,j) - ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i-1,j) ) / dx;
            }
        }
    }
    else if (idir == 2) {// flux in y direction
        // double dyflip = 1 / dy;
        for (int j = loy; j <= hiy+1; ++j ) {
            for (int i = lox; i <= hix; ++i ) {
                ARRAY_2D(flux,flux_lox,flux_loy,flux_hix,flux_hiy,i,j) = 
                    ( ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i,j) - ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i,j-1) ) / dy ;
            }
        }
    }
    else {// error
        exit(0);
    }
}

#ifdef CUDA
__global__
void update_phi_doit_gpu(int id, void* buffer)
{
    // unpack data
    amrex::Real dt, dx, dy;
    int lox, loy, hix, hiy;
    int phi_old_lox, phi_old_loy, phi_old_hix, phi_old_hiy;
    int phi_new_lox, phi_new_loy, phi_new_hix, phi_new_hiy;
    int fx_lox, fx_loy, fx_hix, fx_hiy;
    int fy_lox, fy_loy, fy_hix, fy_hiy;
    amrex::Real* phi_old = 0; 
    amrex::Real* phi_new = 0;
    amrex::Real* fx   = 0; 
    amrex::Real* fy   = 0;
    unpack(id, buffer,
           dt, dx, dy,
           lox, loy, hix, hiy,
           &phi_old, phi_old_lox, phi_old_loy, phi_old_hix, phi_old_hiy,
           &phi_new, phi_new_lox, phi_new_loy, phi_new_hix, phi_new_hiy,
           &fx, fx_lox, fx_loy, fx_hix, fx_hiy,
           &fy, fy_lox, fy_loy, fy_hix, fy_hiy);
    // map cuda thread (cudai, cudaj) to cell edge (i,j) it works on 
    int cudai = threadIdx.x + blockDim.x * blockIdx.x;
    int cudaj = threadIdx.y + blockDim.y * blockIdx.y;
    int i = cudai + lox;
    int j = cudaj + loy;
    if ( i > hix || j > hiy ) return;
    ARRAY_2D(phi_new,phi_new_lox,phi_new_loy,phi_new_hix,phi_new_hiy,i,j) =
        ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i,j) +
        dt/dx * ( ARRAY_2D(fx,fx_lox,fx_loy,fx_hix,fx_hiy,i+1,j) - ARRAY_2D(fx,fx_lox,fx_loy,fx_hix,fx_hiy,i,j) ) +
        dt/dy * ( ARRAY_2D(fy,fy_lox,fy_loy,fy_hix,fy_hiy,i,j+1) - ARRAY_2D(fy,fy_lox,fy_loy,fy_hix,fy_hiy,i,j) ); 
}
#endif

void update_phi_doit_cpu(
            const int& lox, const int& loy, const int& hix, const int& hiy,
            const amrex::Real* __restrict__ phi_old, const int& phi_old_lox, const int& phi_old_loy, const int& phi_old_hix, const int& phi_old_hiy,
            amrex::Real* __restrict__ phi_new, const int& phi_new_lox, const int& phi_new_loy, const int& phi_new_hix, const int& phi_new_hiy,
            const amrex::Real* __restrict__ fx, const int& fx_lox, const int& fx_loy, const int& fx_hix, const int& fx_hiy,
            const amrex::Real* __restrict__ fy, const int& fy_lox, const int& fy_loy, const int& fy_hix, const int& fy_hiy,
            const amrex::Real& dx, const amrex::Real& dy, const amrex::Real& dt)
{
    for (int j = loy; j <= hiy; ++j ) {
        for (int i = lox; i <= hix; ++i ) {
            ARRAY_2D(phi_new,phi_new_lox,phi_new_loy,phi_new_hix,phi_new_hiy,i,j) =
                ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i,j) +
                dt/dx * ( ARRAY_2D(fx,fx_lox,fx_loy,fx_hix,fx_hiy,i+1,j) - ARRAY_2D(fx,fx_lox,fx_loy,fx_hix,fx_hiy,i,j) ) +
                dt/dy * ( ARRAY_2D(fy,fy_lox,fy_loy,fy_hix,fy_hiy,i,j+1) - ARRAY_2D(fy,fy_lox,fy_loy,fy_hix,fy_hiy,i,j) ); 
        }
    }
}


#ifdef CUDA
void compute_flux_on_box(const amrex::Box& bx, int idx, void* buffer){
#if (BL_SPACEDIM == 2)
    dim3 blockSize(BLOCKSIZE_2D,BLOCKSIZE_2D,1);
    dim3 gridSize( (bx.size()[0] + blockSize.x) / blockSize.x, 
                   (bx.size()[1] + blockSize.y) / blockSize.y, 
                    1 
                 );
    cudaStream_t pStream;
    get_stream(&idx, &pStream);
    compute_flux_doit_gpu<<<gridSize, blockSize, 0, pStream>>>(idx, buffer);
#elif (BL_SPACEDIM == 3)
    // TODO
#endif

}
#endif

#ifdef CUDA
void update_phi_on_box(const amrex::Box& bx, int idx, void* buffer){
#if (BL_SPACEDIM == 2)
    dim3 blockSize(BLOCKSIZE_2D,BLOCKSIZE_2D,1);
    dim3 gridSize( (bx.size()[0] + blockSize.x - 1) / blockSize.x, 
                   (bx.size()[1] + blockSize.y - 1) / blockSize.y, 
                    1 
                 );
    cudaStream_t pStream;
    get_stream(&idx, &pStream);
    update_phi_doit_gpu<<<gridSize, blockSize, 0, pStream>>>(idx, buffer);
#elif (BL_SPACEDIM == 3)
    // TODO
#endif
}
#endif

#ifdef CUDA
__device__
void unpack(const int& id, void* buffer,
        amrex::Real& dt, amrex::Real& dx, amrex::Real& dy,
        int& lox, int& loy,int& hix,int& hiy,
        amrex::Real** phi_old, int& phi_old_lox, int& phi_old_loy, int& phi_old_hix, int& phi_old_hiy,
        amrex::Real** phi_new, int& phi_new_lox, int& phi_new_loy, int& phi_new_hix, int& phi_new_hiy,
        amrex::Real** fluxx, int& fluxx_lox, int& fluxx_loy, int& fluxx_hix, int& fluxx_hiy,
        amrex::Real** fluxy, int& fluxy_lox, int& fluxy_loy, int& fluxy_hix, int& fluxy_hiy) 
{
    char* data_real = static_cast<char*>(buffer);
    std::memcpy(&dt, data_real                        , sizeof(amrex::Real));
    std::memcpy(&dx, data_real +   sizeof(amrex::Real), sizeof(amrex::Real));
    std::memcpy(&dy, data_real + 2*sizeof(amrex::Real), sizeof(amrex::Real));
    char* data_int = data_real + 4*sizeof(amrex::Real);
    int nb, nmfab;
    std::memcpy(&nb,    data_int              , sizeof(int));
    std::memcpy(&nmfab, data_int + sizeof(int), sizeof(int));

    data_int = data_int + 4 * sizeof(int);
    int pos = id * 6;
    std::memcpy(&lox, data_int + (pos + 0) * sizeof(int), sizeof(int));
    std::memcpy(&loy, data_int + (pos + 1) * sizeof(int), sizeof(int));
    std::memcpy(&hix, data_int + (pos + 3) * sizeof(int), sizeof(int));
    std::memcpy(&hiy, data_int + (pos + 4) * sizeof(int), sizeof(int));

    char* data_pointer = data_real + 4 * sizeof(amrex::Real) + (4 + 6 * nb) * sizeof(int);
    char* phi_old_data;
    char* phi_new_data;
    char* fluxx_data;
    char* fluxy_data;
    // since char is only one byte and we assume we know data alignment issue here
    std::memcpy(&phi_old_data, data_pointer + (id*nmfab+0)*sizeof(char*), sizeof(char*));
    std::memcpy(&phi_new_data, data_pointer + (id*nmfab+1)*sizeof(char*), sizeof(char*));
    std::memcpy(&fluxx_data, data_pointer + (id*nmfab+2)*sizeof(char*), sizeof(char*));
    std::memcpy(&fluxy_data, data_pointer + (id*nmfab+3)*sizeof(char*), sizeof(char*));

    // get fab sizes
    get_fab_dimension(phi_old_lox, phi_old_loy, phi_old_hix, phi_old_hiy, phi_old_data);
    get_fab_dimension(phi_new_lox, phi_new_loy, phi_new_hix, phi_new_hiy, phi_new_data);

    get_fab_dimension(fluxx_lox, fluxx_loy, fluxx_hix, fluxx_hiy, fluxx_data); 
    get_fab_dimension(fluxy_lox, fluxy_loy, fluxy_hix, fluxy_hiy, fluxy_data);


    // assume it's aligned so we can cast
    *phi_old = reinterpret_cast<amrex::Real*>(phi_old_data);
    *phi_new = reinterpret_cast<amrex::Real*>(phi_new_data);
    *fluxx = reinterpret_cast<amrex::Real*>(fluxx_data);
    *fluxy = reinterpret_cast<amrex::Real*>(fluxy_data);
}
#endif

