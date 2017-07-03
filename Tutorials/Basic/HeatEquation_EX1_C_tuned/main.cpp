#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_MFIter.H>

#include <array>
#include <memory>

#include "myfunc_F.H"
#include "advance_kernel.H"
#include <AMReX_Device.H>



using namespace amrex;

void main_main ();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    
    main_main();
    
    amrex::Finalize();
    return 0;
}

void advance (MultiFab& old_phi, MultiFab& new_phi,
	      std::array<MultiFab, BL_SPACEDIM>& flux,
	      Real dt, const Geometry& geom)
{
    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries
    old_phi.FillBoundary(geom.periodicity());

    int Ncomp = old_phi.nComp();
    int ng_p = old_phi.nGrow();
    int ng_f = flux[0].nGrow();

    const Real* dx = geom.CellSize();

    //
    // Note that this simple example is not optimized.
    // The following two MFIter loops could be merged
    // and we do not have to use flux MultiFab.
    // 


#ifdef CUDA
    MFIterRegister mfir;
    mfir.registerMultiFab(old_phi);
    mfir.registerMultiFab(new_phi);
    mfir.registerMultiFab(flux[0]);
    mfir.registerMultiFab(flux[1]);
#if (BL_SPACEDIM == 3)   
    mfir.registerMultiFab(&(flux[2]));
#endif
    mfir.registerCellSize(dx[0], dx[1]
#if (BL_SPACEDIM == 3)   
            , dx[2]
#else
            , 0.0
#endif
    );
    mfir.registerTimeStep(dt);
    mfir.closeRegister();
#endif // CUDA
    
    // Compute fluxes one grid at a time
    // When construct a MFIter with MFIterRegister, kick off
    // transfer of arraydata registered in the MFIterRegister from htod
    for ( MFIter mfi(old_phi); mfi.isValid(); ++mfi )
    {
        int idx = mfi.LocalIndex();
        const Box& bx = mfi.validbox();
#if (BL_SPACEDIM == 2)
#ifdef CUDA
        // copy old solution from host to device
        old_phi[mfi].toDevice(idx);
        compute_flux_on_box(bx, idx, mfir.get_device_buffer());
        update_phi_on_box(bx, idx, mfir.get_device_buffer());
        // copy updated solution from device to host
        new_phi[mfi].toHost(idx);
#else
        const int* lo = bx.loVect();
        const int* hi = bx.hiVect();
        compute_flux_doit_cpu(lo[0],lo[1],hi[0],hi[1],
                old_phi[mfi].dataPtr(), 
                old_phi[mfi].loVect()[0], old_phi[mfi].loVect()[1],
                old_phi[mfi].hiVect()[0], old_phi[mfi].hiVect()[1],
                flux[0][mfi].dataPtr(), 
                flux[0][mfi].loVect()[0], flux[0][mfi].loVect()[1],
                flux[0][mfi].hiVect()[0], flux[0][mfi].hiVect()[1],
                dx[0], dx[1], 1);
        compute_flux_doit_cpu(lo[0],lo[1],hi[0],hi[1],
                old_phi[mfi].dataPtr(), 
                old_phi[mfi].loVect()[0], old_phi[mfi].loVect()[1],
                old_phi[mfi].hiVect()[0], old_phi[mfi].hiVect()[1],
                flux[1][mfi].dataPtr(), 
                flux[1][mfi].loVect()[0], flux[1][mfi].loVect()[1],
                flux[1][mfi].hiVect()[0], flux[1][mfi].hiVect()[1],
                dx[0], dx[1], 2);
#endif //CUDA
#elif (BL_SPACEDIM == 3)
        // TODO
#else 
        exit(0);
#endif

    }

//     // Advance the solution one grid at a time
//     for ( MFIter mfi(old_phi); mfi.isValid(); ++mfi )
//     {
//         const int idx = mfi.LocalIndex();
//         const Box& bx = mfi.validbox();
// #ifdef CUDA
//         update_phi_on_box(bx, idx, mfir.get_device_buffer());
//         new_phi[mfi].toHost(idx);
// #else
//         const int* lo = bx.loVect();
//         const int* hi = bx.hiVect();
//         update_phi_doit_cpu(lo[0],lo[1],hi[0],hi[1],
//                 old_phi[mfi].dataPtr(), 
//                 old_phi[mfi].loVect()[0], old_phi[mfi].loVect()[1],
//                 old_phi[mfi].hiVect()[0], old_phi[mfi].hiVect()[1],
//                 new_phi[mfi].dataPtr(), 
//                 new_phi[mfi].loVect()[0], new_phi[mfi].loVect()[1],
//                 new_phi[mfi].hiVect()[0], new_phi[mfi].hiVect()[1],
//                 flux[0][mfi].dataPtr(), 
//                 flux[0][mfi].loVect()[0], flux[0][mfi].loVect()[1],
//                 flux[0][mfi].hiVect()[0], flux[0][mfi].hiVect()[1],
//                 flux[1][mfi].dataPtr(), 
//                 flux[1][mfi].loVect()[0], flux[1][mfi].loVect()[1],
//                 flux[1][mfi].hiVect()[0], flux[1][mfi].hiVect()[1],
//                 dx[0], dx[1], dt);
// #endif //CUDA
//         
//     }

#ifdef CUDA
    // copy all data d2h
    // mfir.allFabToHost();
    gpu_synchronize();
#endif

}

void main_main ()
{
    // What time is it now?  We'll use this to compute total run time.
    Real strt_time = ParallelDescriptor::second();

    // BL_SPACEDIM: number of dimensions
    int n_cell, max_grid_size, nsteps, plot_int, is_periodic[BL_SPACEDIM];

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of 
        //   a square (or cubic) domain.
        pp.get("n_cell",n_cell);

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be writtenq
        plot_int = -1;
        pp.query("plot_int",plot_int);

        // Default nsteps to 0, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);
    }

    // make BoxArray and Geometry
    BoxArray ba;
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
        IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);
        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(max_grid_size);

       // This defines the physical box, [-1,1] in each direction.
        RealBox real_box({AMREX_D_DECL(-1.0,-1.0,-1.0)},
                         {AMREX_D_DECL( 1.0, 1.0, 1.0)});

        // This says we are using Cartesian coordinates
        int coord = 0;
	
        // This sets the boundary conditions to be doubly or triply periodic
        std::array<int,BL_SPACEDIM> is_periodic {AMREX_D_DECL(1,1,1)};
        
        // This defines a Geometry object
        geom.define(domain,&real_box,coord,is_periodic.data());
    }

    // Nghost = number of ghost cells for each array 
    int Nghost = 1;
    
    // Ncomp = number of components for each array
    int Ncomp  = 1;

    // time = starting time in the simulation
    Real time = 0.0;
  
    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // we allocate two phi multifabs; one will store the old state, the other the new.
    std::unique_ptr<MultiFab> phi_old(new MultiFab(ba, dm, Ncomp, Nghost));
    std::unique_ptr<MultiFab> phi_new(new MultiFab(ba, dm, Ncomp, Nghost));


    phi_old->setVal(0.0);
    phi_new->setVal(0.0);

    // Initialize phi_new by calling a Fortran routine.
    // MFIter = MultiFab Iterator
    // for ( MFIter mfi(*phi_new); mfi.isValid(); ++mfi )
    for ( MFIter mfi(*phi_old); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.validbox();

        // modify host data
        init_phi(bx.loVect(), bx.hiVect(),
                 BL_TO_FORTRAN_ANYD((*phi_new)[mfi]),
                 geom.CellSize(), geom.ProbLo(), geom.ProbHi());
    }

    // compute the time step
    const Real* dx = geom.CellSize();
    Real dt = 0.9*dx[0]*dx[0] / (2.0*BL_SPACEDIM);

    // Write a plotfile of the initial data if plot_int > 0 (plot_int was defined in the inputs file)
    if (plot_int > 0)
    {
        int n = 0;
        const std::string& pltfile = amrex::Concatenate("plt",n,5);
        WriteSingleLevelPlotfile(pltfile, *phi_new, {"phi"}, geom, time, 0);
    }

    std::array<MultiFab, BL_SPACEDIM> flux;
    for (int dir = 0; dir < BL_SPACEDIM; dir++)
    {
        // flux(dir) has one component, zero ghost cells, and is nodal in direction dir
        BoxArray edge_ba = ba;
        edge_ba.surroundingNodes(dir);
        flux[dir].define(edge_ba, dm, 1, 0);
    }


    MultiFab::Copy(*phi_old, *phi_new, 0, 0, 1, 0);

// This acts as initialization of device memory
// Otherwise memcheck will report accessing to uninitialized global memory
// TODO: move this to somewhere else or wrap it in a member function
// of Basefab
#ifdef CUDA
    for ( MFIter mfi(*phi_old); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.validbox();
        int idx = mfi.LocalIndex();
        (*phi_old)[mfi].toDevice(idx);
        (*phi_new)[mfi].toDevice(idx);
        flux[0][mfi].toDevice(idx);
        flux[1][mfi].toDevice(idx);
    }
#endif

    for (int n = 1; n <= nsteps; ++n)
    {

        // new_phi = old_phi + dt * (something)
        advance(*phi_old, *phi_new, flux, dt, geom); 
        time = time + dt;
        
        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << n << "\n";

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && n%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",n,5);
            WriteSingleLevelPlotfile(pltfile, *phi_new, {"phi"}, geom, time, n);
        }
        // switch new and old
        phi_new.swap(phi_old);
    }

    // Call the timer again and compute the maximum difference between the start time and stop time
    //   over all processors
    Real stop_time = ParallelDescriptor::second() - strt_time;
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

    // Tell the I/O Processor to write out the "run time"
    amrex::Print() << "Run time = " << stop_time << std::endl;


}






