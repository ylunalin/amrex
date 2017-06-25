#include "advance_kernel.H"
#include <cstring>
#include <AMReX_BLFort.H>
#include <AMReX_BArena.H> // for def of HEADER_SIZE

// #define ARRAY_2D(PHI, XSIZE, YSIZE, I, J) PHI[I*YSIZE+J]
#define ARRAY_2D(PHI, LO_X, LO_Y, HI_X, HI_Y, I, J) PHI[(I-LO_X)*(HI_Y-LO_Y+1)+J-LO_Y]
// #define ARRAY_2D(PHI, XSIZE, YSIZE, I, J) (PHI[I+YSIZE*J])
//

// cudaStream_t stream_from_id(int id) 
// {
//     return stream[id % max_cuda_stream];
// }
//
//
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

void compute_flux_doit_gpu(
            int& lox, int& loy, int& hix, int& hiy,
            amrex::Real* phi, int& phi_lox, int& phi_loy, int& phi_hix, int& phi_hiy,
            amrex::Real* flux, int& flux_lox, int& flux_loy, int& flux_hix, int& flux_hiy,
            amrex::Real dx, amrex::Real dy, const int& idir)
{
    /*
    // map cuda thread (cudai, cudaj) to cell edge (i,j) it works on 
    int cudai = threadIdx.x + blockDim.x * blockIdx.x;
    int cudaj = threadIdx.y + blockDim.y * blockIdx.y;
    int i = cudai + lox;
    int j = cudaj + loy;
    if (idir == 1) {// flux in x direction
        if ( i > (hix+1) || j > hiy ) return;
        ARRAY_2D(flux,flux_lox,flux_loy,flux_hix,flux_hiy,i,j) = 
            ( ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i,j) - ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i-1,j) ) / dx;
    }
    else if (idir == 2) {// flux in y direction
        if ( i > hix || j > (hiy+1) ) return;
        ARRAY_2D(flux,flux_lox,flux_loy,flux_hix,flux_hiy,i,j) = 
            ( ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i,j) - ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i,j-1) ) / dy;
    }
    else {// error
        exit(0);
    }
    */
}

void compute_flux_doit_cpu(
            int& lox, int& loy, int& hix, int& hiy,
            amrex::Real* phi, int& phi_lox, int& phi_loy, int& phi_hix, int& phi_hiy,
            amrex::Real* flux, int& flux_lox, int& flux_loy, int& flux_hix, int& flux_hiy,
            amrex::Real dx, amrex::Real dy, const int& idir)
{
    if (idir == 1) {// flux in x direction
        for (int i = lox; i <= hix+1; ++i ) {
            for (int j = loy; j <= hiy; ++j ) {
                ARRAY_2D(flux,flux_lox,flux_loy,flux_hix,flux_hiy,i,j) = 
                    ( ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i,j) - ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i-1,j) ) / dx;
            }
        }
    }
    else if (idir == 2) {// flux in y direction
        for (int i = lox; i <= hix; ++i ) {
            for (int j = loy; j <= hiy+1; ++j ) {
                ARRAY_2D(flux,flux_lox,flux_loy,flux_hix,flux_hiy,i,j) = 
                    ( ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i,j) - ARRAY_2D(phi,phi_lox,phi_loy,phi_hix,phi_hiy,i,j-1) ) / dy;
            }
        }
    }
    else {// error
        exit(0);
    }
}

void update_phi_doit_gpu(
            int& lox, int& loy, int& hix, int& hiy,
            amrex::Real* phi_old, int& phi_old_lox, int& phi_old_loy, int& phi_old_hix, int& phi_old_hiy,
            amrex::Real* phi_new, int& phi_new_lox, int& phi_new_loy, int& phi_new_hix, int& phi_new_hiy,
            amrex::Real* fx, int& fx_lox, int& fx_loy, int& fx_hix, int& fx_hiy,
            amrex::Real* fy, int& fy_lox, int& fy_loy, int& fy_hix, int& fy_hiy,
            amrex::Real dx, amrex::Real dy, amrex::Real dt)
{
    /*
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
    */
}

void update_phi_doit_cpu(
            int& lox, int& loy, int& hix, int& hiy,
            amrex::Real* phi_old, int& phi_old_lox, int& phi_old_loy, int& phi_old_hix, int& phi_old_hiy,
            amrex::Real* phi_new, int& phi_new_lox, int& phi_new_loy, int& phi_new_hix, int& phi_new_hiy,
            amrex::Real* fx, int& fx_lox, int& fx_loy, int& fx_hix, int& fx_hiy,
            amrex::Real* fy, int& fy_lox, int& fy_loy, int& fy_hix, int& fy_hiy,
            amrex::Real dx, amrex::Real dy, amrex::Real dt)
{
    for (int i = lox; i <= hix; ++i ) {
        for (int j = loy; j <= hiy; ++j ) {
            ARRAY_2D(phi_new,phi_new_lox,phi_new_loy,phi_new_hix,phi_new_hiy,i,j) =
                ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i,j) +
                dt/dx * ( ARRAY_2D(fx,fx_lox,fx_loy,fx_hix,fx_hiy,i+1,j) - ARRAY_2D(fx,fx_lox,fx_loy,fx_hix,fx_hiy,i,j) ) +
                dt/dy * ( ARRAY_2D(fy,fy_lox,fy_loy,fy_hix,fy_hiy,i,j+1) - ARRAY_2D(fy,fy_lox,fy_loy,fy_hix,fy_hiy,i,j) ); 
        }
    }
}

void compute_flux_on_box(const int& id, void* buffer){
    amrex::Real dt, dx, dy;
#if (BL_SPACEDIM == 3)
    amrex::Real dz;
#endif

    int lox, loy, hix, hiy;
#if (BL_SPACEDIM == 3)
    int loz, hiz;
#endif

    int phi_old_lox, phi_old_loy, phi_old_hix, phi_old_hiy;
#if (BL_SPACEDIM == 3)
    int phi_old_loz, phi_old_hiz;
#endif

    int phi_new_lox, phi_new_loy, phi_new_hix, phi_new_hiy;
#if (BL_SPACEDIM == 3)
    int phi_new_loz, phi_new_hiz;
#endif

    int fluxx_lox, fluxx_loy, fluxx_hix, fluxx_hiy;
#if (BL_SPACEDIM == 3)
    int fluxx_loz, fluxx_hiz;
#endif

    int fluxy_lox, fluxy_loy, fluxy_hix, fluxy_hiy;
#if (BL_SPACEDIM == 3)
    int fluxy_loz, fluxy_hiz;
#endif

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


#if (BL_SPACEDIM == 2)
    compute_flux_doit_cpu(
            lox, loy, hix, hiy,
            phi_old, phi_old_lox, phi_old_loy, phi_old_hix, phi_old_hiy,
            fluxx, fluxx_lox, fluxx_loy, fluxx_hix, fluxx_hiy,
            dx,dy,1);
    compute_flux_doit_cpu(
            lox, loy, hix, hiy,
            phi_old, phi_old_lox, phi_old_loy, phi_old_hix, phi_old_hiy,
            fluxy, fluxy_lox, fluxy_loy, fluxy_hix, fluxy_hiy,
            dx,dy,2);
#elif (BL_SPACEDIM == 3)
    // TODO
#endif

}

void update_phi_on_box(const int& id, void* buffer){
    amrex::Real dt, dx, dy;
#if (BL_SPACEDIM == 3)
    amrex::Real dz;
#endif

    int lox, loy, hix, hiy;
#if (BL_SPACEDIM == 3)
    int loz, hiz;
#endif

    int phi_old_lox, phi_old_loy, phi_old_hix, phi_old_hiy;
#if (BL_SPACEDIM == 3)
    int phi_old_loz, phi_old_hiz;
#endif

    int phi_new_lox, phi_new_loy, phi_new_hix, phi_new_hiy;
#if (BL_SPACEDIM == 3)
    int phi_new_loz, phi_new_hiz;
#endif

    int fluxx_lox, fluxx_loy, fluxx_hix, fluxx_hiy;
#if (BL_SPACEDIM == 3)
    int fluxx_loz, fluxx_hiz;
#endif

    int fluxy_lox, fluxy_loy, fluxy_hix, fluxy_hiy;
#if (BL_SPACEDIM == 3)
    int fluxy_loz, fluxy_hiz;
#endif


    amrex::Real* phi_old =   0; 
    amrex::Real* phi_new =   0;
    amrex::Real* fluxx   =   0; 
    amrex::Real* fluxy   =   0;

    unpack(id, buffer,
           dt, dx, dy,
           lox, loy, hix, hiy,
           &phi_old, phi_old_lox, phi_old_loy, phi_old_hix, phi_old_hiy,
           &phi_new, phi_new_lox, phi_new_loy, phi_new_hix, phi_new_hiy,
           &fluxx, fluxx_lox, fluxx_loy, fluxx_hix, fluxx_hiy,
           &fluxy, fluxy_lox, fluxy_loy, fluxy_hix, fluxy_hiy);

#if (BL_SPACEDIM == 2)
    update_phi_doit_cpu(
            lox, loy, hix, hiy,
            phi_old, phi_old_lox, phi_old_loy, phi_old_hix, phi_old_hiy,
            phi_new, phi_new_lox, phi_new_loy, phi_new_hix, phi_new_hiy,
            fluxx, fluxx_lox, fluxx_loy, fluxx_hix, fluxx_hiy,
            fluxy, fluxy_lox, fluxy_loy, fluxy_hix, fluxy_hiy,
            dx,dy,dt);
#elif (BL_SPACEDIM == 3)
    // TODO
#endif

}

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
    char* data_int = data_real + 4 * sizeof(amrex::Real);
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
    // TODO: can replace copy here with pointer movement 
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

