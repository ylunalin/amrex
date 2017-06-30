#ifndef ADVANCE_KERNEL_H__ 
#define ADVANCE_KERNEL_H__

#include <AMReX_REAL.H>
#include <AMReX_Box.H>

#ifdef CUDA
__device__
void get_fab_dimension(int& lox, int& loy, int& hix, int& hiy, void* data_); 

__global__
void compute_flux_doit_gpu(int id, void* buffer);

__global__
void update_phi_doit_gpu(int id, void* buffer);

__device__
void unpack(const int& id, void* buffer,
        amrex::Real& dt, amrex::Real& dx, amrex::Real& dy,
        int& lox, int& loy,int& hix,int& hiy,
        amrex::Real** phi_old, int& phi_old_lox, int& phi_old_loy, int& phi_old_hix, int& phi_old_hiy,
        amrex::Real** phi_new, int& phi_new_lox, int& phi_new_loy, int& phi_new_hix, int& phi_new_hiy,
        amrex::Real** fx, int& fx_lox, int& fx_loy, int& fx_hix, int& fx_hiy,
        amrex::Real** fy, int& fy_lox, int& fy_loy, int& fy_hix, int& fy_hiy);

void compute_flux_on_box(const amrex::Box& bx, int idx, void* buffer);
void update_phi_on_box(const amrex::Box& bx, int idx, void* buffer);
#endif // CUDA


void compute_flux_doit_cpu(
            const int& lox, const int& loy, const int& hix, const int& hiy,
            amrex::Real* phi, const int& phi_lox, const int& phi_loy, const int& phi_hix, const int& phi_hiy,
            amrex::Real* flux, const int& flux_lox, const int& flux_loy, const int& flux_hix, const int& flux_hiy,
            amrex::Real dx, amrex::Real dy, const int& idir);

void update_phi_doit_cpu(
            const int& lox, const int& loy, const int& hix, const int& hiy,
            amrex::Real* phi_old, const int& phi_old_lox, const int& phi_old_loy, const int& phi_old_hix, const int& phi_old_hiy,
            amrex::Real* phi_new, const int& phi_new_lox, const int& phi_new_loy, const int& phi_new_hix, const int& phi_new_hiy,
            amrex::Real* fx, const int& fx_lox, const int& fx_loy, const int& fx_hix, const int& fx_hiy,
            amrex::Real* fy, const int& fy_lox, const int& fy_loy, const int& fy_hix, const int& fy_hiy,
            amrex::Real dx, amrex::Real dy, amrex::Real dt);

#endif // ADVANCE_KERNEL_H__
