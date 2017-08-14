#ifndef MY_ABL_H_
#define MY_ABL_H_

#include <AMReX_MultiFab.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_BndryData.H>

// We solve (a alpha - b del dot beta grad) soln = rhs
// where a and b are scalars, alpha and beta are arrays

class ABL
{
public:
    ABL ();

    void solve ();
    void solve_with_Cpp ();
    void set_boundary(amrex::BndryData& bd);

private:

    // runtime parameters
    int n_cell = 128;
    int max_grid_size = 64;
    amrex::LinOpBCType bc_type = amrex::LinOpBCType::Dirichlet;
    amrex::Real bc_value = 0.0;
    amrex::Real tol_rel = 1.e-8;
    amrex::Real tol_abs = 0.0;
    int maxiter = 200;
    int verbose = 2;

    // parameters in setting up the problem
    const amrex::Real a = 1.e-3;
    const amrex::Real b = 1.0;
    const amrex::Real sigma = 10.0;
    const amrex::Real w = 0.05;

    // ba and dmap
    amrex::Array<amrex::BoxArray> ba;
    amrex::Array<amrex::DistributionMapping> dmap;

    // data
    amrex::Array<amrex::MultiFab> rhs;
    amrex::Array<amrex::MultiFab> alpha;
    amrex::Array<amrex::Array<std::unique_ptr<amrex::MultiFab>>> beta;
    amrex::Array<amrex::MultiFab> soln;
    amrex::Array<amrex::MultiFab> the_soln;
    amrex::Array<amrex::Geometry> geom;
    
    void init_rhs ();
    void init_coeffs ();
    void comp_the_solution ();
};

#endif
