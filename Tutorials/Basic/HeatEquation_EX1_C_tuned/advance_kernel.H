#ifndef ADVANCE_KERNEL_H__ 
#define ADVANCE_KERNEL_H__

#include <AMReX_REAL.H>
#include <AMReX_Box.H>

#ifdef CUDA
__global__
void compute_flux_doit_gpu(
        const int lox, const int loy, const int hix, const int hiy,
        const __restrict__ amrex::Real* phi,
        const int phi_lox, const int phi_loy, const int phi_hix, const int phi_hiy,
        __restrict__ amrex::Real* fluxx,
        const int fluxx_lox, const int fluxx_loy, const int fluxx_hix, const int fluxx_hiy,
        __restrict__ amrex::Real* fluxy,
        const int fluxy_lox, const int fluxy_loy, const int fluxy_hix, const int fluxy_hiy,
        const amrex::Real dx, const amrex::Real dy);


__global__
void update_phi_doit_gpu(
        const int lox, const int loy, const int hix, const int hiy,
        const __restrict__ amrex::Real* phi_old,
        const int phi_old_lox, const int phi_old_loy, const int phi_old_hix, const int phi_old_hiy,
        __restrict__ amrex::Real* phi_new,
        const int phi_new_lox, const int phi_new_loy, const int phi_new_hix, const int phi_new_hiy,
        const __restrict__ amrex::Real* fx,
        const int fx_lox, const int fx_loy, const int fx_hix, const int fx_hiy,
        const __restrict__ amrex::Real* fy,
        const int fy_lox, const int fy_loy, const int fy_hix, const int fy_hiy,
        const amrex::Real dx, const amrex::Real dy, const amrex::Real dt); 

void compute_flux_c(const int& lox, const int& loy, const int& hix, const int& hiy,
                  const amrex::Real* phi,
                  const int& phi_lox, const int& phi_loy, const int& phi_hix, const int& phi_hiy,
                  amrex::Real* fluxx,
                  const int& fx_lox, const int& fx_loy, const int& fx_hix, const int& fx_hiy,
                  amrex::Real* fluxy,
                  const int& fy_lox, const int& fy_loy, const int& fy_hix, const int& fy_hiy,
                  const amrex::Real& dx, const amrex::Real& dy, const int& idx); 
void update_phi_c(const int& lox, const int& loy, const int& hix, const int& hiy,
                const amrex::Real* phi_old,
                const int& phi_old_lox, const int& phi_old_loy, const int& phi_old_hix, const int& phi_old_hiy,
                amrex::Real* phi_new,
                const int& phi_new_lox, const int& phi_new_loy, const int& phi_new_hix, const int& phi_new_hiy,
                const amrex::Real* fluxx,
                const int& fx_lox, const int& fx_loy, const int& fx_hix, const int& fx_hiy,
                const amrex::Real* fluxy,
                const int& fy_lox, const int& fy_loy, const int& fy_hix, const int& fy_hiy,
                const amrex::Real& dx, const amrex::Real& dy, const amrex::Real& dt, const int& idx);
#endif // CUDA


void compute_flux_doit_cpu(
            const int& lox, const int& loy, const int& hix, const int& hiy,
            const amrex::Real* __restrict__ phi, const int& phi_lox, const int& phi_loy, const int& phi_hix, const int& phi_hiy,
            amrex::Real* __restrict__ flux, const int& flux_lox, const int& flux_loy, const int& flux_hix, const int& flux_hiy,
            const amrex::Real& dx, const amrex::Real& dy, const int& idir);

void update_phi_doit_cpu(
            const int& lox, const int& loy, const int& hix, const int& hiy,
            const amrex::Real* __restrict__ phi_old, const int& phi_old_lox, const int& phi_old_loy, const int& phi_old_hix, const int& phi_old_hiy,
            amrex::Real* __restrict__       phi_new, const int& phi_new_lox, const int& phi_new_loy, const int& phi_new_hix, const int& phi_new_hiy,
            const amrex::Real* __restrict__ fx,      const int& fx_lox,      const int& fx_loy,      const int& fx_hix,      const int& fx_hiy,
            const amrex::Real* __restrict__ fy,      const int& fy_lox,      const int& fy_loy,      const int& fy_hix,      const int& fy_hiy,
            const amrex::Real& dx, const amrex::Real& dy, const amrex::Real& dt);

#endif // ADVANCE_KERNEL_H__
