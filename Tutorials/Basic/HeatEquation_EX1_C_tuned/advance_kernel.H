#ifndef ADVANCE_KERNEL_H__ 
#define ADVANCE_KERNEL_H__

#include <AMReX_REAL.H>

void get_fab_dimension(int& lox, int& loy, int& hix, int& hiy, void* data_); 

void compute_flux_doit_gpu(
            int& lox, int& loy, int& hix, int& hiy,
            amrex::Real* phi, int& phi_lox, int& phi_loy, int& phi_hix, int& phi_hiy,
            amrex::Real* flux, int& flux_lox, int& flux_loy, int& flux_hix, int& flux_hiy,
            amrex::Real dx, amrex::Real dy, const int& idir);

void compute_flux_doit_cpu(
            int& lox, int& loy, int& hix, int& hiy,
            amrex::Real* phi, int& phi_lox, int& phi_loy, int& phi_hix, int& phi_hiy,
            amrex::Real* flux, int& flux_lox, int& flux_loy, int& flux_hix, int& flux_hiy,
            amrex::Real dx, amrex::Real dy, const int& idir);

void update_phi_doit_gpu(
            int& lox, int& loy, int& hix, int& hiy,
            amrex::Real* phi_old, int& phi_old_lox, int& phi_old_loy, int& phi_old_hix, int& phi_old_hiy,
            amrex::Real* phi_new, int& phi_new_lox, int& phi_new_loy, int& phi_new_hix, int& phi_new_hiy,
            amrex::Real* fx, int& fx_lox, int& fx_loy, int& fx_hix, int& fx_hiy,
            amrex::Real* fy, int& fy_lox, int& fy_loy, int& fy_hix, int& fy_hiy,
            amrex::Real dx, amrex::Real dy, amrex::Real dt);

void update_phi_doit_cpu(
            int& lox, int& loy, int& hix, int& hiy,
            amrex::Real* phi_old, int& phi_old_lox, int& phi_old_loy, int& phi_old_hix, int& phi_old_hiy,
            amrex::Real* phi_new, int& phi_new_lox, int& phi_new_loy, int& phi_new_hix, int& phi_new_hiy,
            amrex::Real* fx, int& fx_lox, int& fx_loy, int& fx_hix, int& fx_hiy,
            amrex::Real* fy, int& fy_lox, int& fy_loy, int& fy_hix, int& fy_hiy,
            amrex::Real dx, amrex::Real dy, amrex::Real dt);

void compute_flux_on_box(const int& id, void* buffer);

void update_phi_on_box(const int& id, void* buffer);

void unpack(int& id, void* buffer,
        amrex::Real& dt, amrex::Real& dx, amrex::Real& dy,
        amrex::Real** phi_old, int& phi_old_lox, int& phi_old_loy, int& phi_old_hix, int& phi_old_hiy,
        amrex::Real** phi_new, int& phi_new_lox, int& phi_new_loy, int& phi_new_hix, int& phi_new_hiy,
        amrex::Real** fx, int& fx_lox, int& fx_loy, int& fx_hix, int& fx_hiy,
        amrex::Real** fy, int& fy_lox, int& fy_loy, int& fy_hix, int& fy_hiy);


#endif // ADVANCE_KERNEL_H__
