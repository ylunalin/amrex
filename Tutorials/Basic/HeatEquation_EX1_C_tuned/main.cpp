
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
#include <AMReX_Device.H>

#ifdef CUDA
#include "cuda_profiler_api.h"
#endif

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
    BL_PROFILE("main::advance")
// #ifdef CUDA
//     cudaProfilerStart();
// #endif
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


    MFIterRegister mfir;
    mfir.registerMultiFab(&old_phi);
    mfir.registerMultiFab(&new_phi);
    mfir.registerMultiFab(&(flux[0]));
    mfir.registerMultiFab(&(flux[1]));
#if (BL_SPACEDIM == 3)   
    mfir.registerMultiFab(&(flux[2]));
#endif
    mfir.registerCellSize(dx[1], dx[1]
#if (BL_SPACEDIM == 3)   
            , dx[2]
#else
            , 0.0
#endif
    );
    mfir.registerTimeStep(dt);
    mfir.closeRegister();
    // mfir.printInfo();
    // exit(0);
    // Compute fluxes one grid at a time
    // When construct a MFIter with MFIterRegister, kick off
    // transfer of arraydata registered in the MFIterRegister
    // from htod
    for ( MFIter mfi(old_phi, mfir); mfi.isValid(); ++mfi )
    {
        // const Box& bx = mfi.validbox();
	// const int idx = mfi.tileIndex();

//         {
//         BL_PROFILE("compute_flux_cpu_side")
//         compute_flux(bx.loVect(), bx.hiVect(),
//                      BL_TO_FORTRAN_ANYD(old_phi[mfi]),
//                      BL_TO_FORTRAN_ANYD(flux[0][mfi]),
//                      BL_TO_FORTRAN_ANYD(flux[1][mfi]),
// #if (BL_SPACEDIM == 3)   
//                      BL_TO_FORTRAN_ANYD(flux[2][mfi]),
// #endif
//                      dx, &idx);
//         }
        const int idx = mfi.LocalIndex();
        work_on_box<<<>>>(idx, mfi.get_device_buffer());

// #ifdef CUDA
//     cudaProfilerStop();
// #endif
    }

#ifdef CUDA
    gpu_synchronize();
#endif

    // Advance the solution one grid at a time
    for ( MFIter mfi(old_phi); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.validbox();
	const int idx = mfi.tileIndex();
        
        {
        BL_PROFILE("update_phi_cpu_side")
        update_phi(bx.loVect(), bx.hiVect(),
                   BL_TO_FORTRAN_ANYD(old_phi[mfi]),
                   BL_TO_FORTRAN_ANYD(new_phi[mfi]),
                   BL_TO_FORTRAN_ANYD(flux[0][mfi]),
                   BL_TO_FORTRAN_ANYD(flux[1][mfi]),
#if (BL_SPACEDIM == 3)   
                   BL_TO_FORTRAN_ANYD(flux[2][mfi]),
#endif
                   dx, &dt, &idx);
        }
    }

#ifdef CUDA
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


    // // debug MFIterRegister
    // {
    // MFIterRegister mfir;
    // mfir.registerMultiFab(phi_old.get());
    // mfir.registerMultiFab(phi_new.get());
    // mfir.registerCellSize(0.1, 0.2, 0.3);
    // mfir.registerTimeStep(0.01);
    // mfir.closeRegister();
    // for ( MFIter mfi(*phi_old); mfi.isValid(); ++mfi ) {
    //     const Box& bx = mfi.validbox();

    //     amrex::Print() << "Box:" << mfi.index() << std::endl;
    //     // amrex::Print() << "local index: " << mfi.LocalIndex() << std::endl;
    //     // amrex::Print() << "index: " << mfi.index() << std::endl;
    //     // amrex::Print() << bx.loVect()[0] << std::endl;
    //     // amrex::Print() << bx.loVect()[1] << std::endl;
    //     // amrex::Print() << bx.hiVect()[0] << std::endl;
    //     // amrex::Print() << bx.hiVect()[1] << std::endl;
    //     amrex::Print() << "lo and hi:" << std::endl;
    //     amrex::Print() << bx.smallEnd() << std::endl;
    //     amrex::Print() << bx.bigEnd() << std::endl;
    //     amrex::Print() << "phi_old lo and hi:" << std::endl;
    //     amrex::Print() << (*phi_old)[mfi].smallEnd() << std::endl;
    //     amrex::Print() << (*phi_old)[mfi].bigEnd() << std::endl;
    //     amrex::Print() << "phi_old device pointer:" << (*phi_old)[mfi].devicePtr() << std::endl;
    //     amrex::Print() << "phi_new lo and hi:" << std::endl;
    //     amrex::Print() << (*phi_new)[mfi].smallEnd() << std::endl;
    //     amrex::Print() << (*phi_new)[mfi].bigEnd() << std::endl;
    //     amrex::Print() << "phi_new device pointer:" << (*phi_new)[mfi].devicePtr() << std::endl;

    //     amrex::Print() << std::endl;
    // }
    // mfir.printInfo();
    // return;
    // }

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


    // MultiFab::Copy(*phi_old, *phi_new, 0, 0, 1, 0);
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

#ifdef CUDA
#ifdef BL_TINY_PROFILING
    // Time device subroutines
    {
        Real cuda_time;
        int ncalls;
        int timer_id;

        timer_id = 1;
        get_cuda_time(&timer_id, &cuda_time);
        get_cuda_num_calls(&timer_id, &ncalls);
        amrex::Print() << "Time for memory copy from host to device in compute_flux(): " << std::endl;
        amrex::Print() << cuda_time << std::endl;
        amrex::Print() << "Number of calls of the timer: " << std::endl;
        amrex::Print() << ncalls << std::endl;

        timer_id = 2;
        get_cuda_time(&timer_id, &cuda_time);
        get_cuda_num_calls(&timer_id, &ncalls);
        amrex::Print() << "Time for kernel in compute_flux(): " << std::endl;
        amrex::Print() << cuda_time << std::endl;
        amrex::Print() << "Number of calls of the timer: " << std::endl;
        amrex::Print() << ncalls << std::endl;

        timer_id = 3;
        get_cuda_time(&timer_id, &cuda_time);
        get_cuda_num_calls(&timer_id, &ncalls);
        amrex::Print() << "Time for memory copy from host to device in update_phi(): " << std::endl;
        amrex::Print() << cuda_time << std::endl;
        amrex::Print() << "Number of calls of the timer: " << std::endl;
        amrex::Print() << ncalls << std::endl;

        timer_id = 4;
        get_cuda_time(&timer_id, &cuda_time);
        get_cuda_num_calls(&timer_id, &ncalls);
        amrex::Print() << "Time for kernel in update_phi(): " << std::endl;
        amrex::Print() << cuda_time << std::endl;
        amrex::Print() << "Number of calls of the timer: " << std::endl;
        amrex::Print() << ncalls << std::endl;
    }
#endif
#endif

}

__global__
void work_on_fab(const int& id, const void* buffer){
    // decode
    // amrex::Print() << "Print information in MFIter::buffer ..." << std::endl;
    amrex::Real* real_ptr = static_cast<amrex::Real*>(buffer);
    // amrex::Print() << "dt: " << real_ptr[0] << std::endl;
    // amrex::Print() << "dx: " << real_ptr[1] << std::endl;
    // amrex::Print() << "dy: " << real_ptr[2] << std::endl;
    amrex::Real dt = real_ptr[0];
    amrex::Real dx = real_ptr[1];
    amrex::Real dy = real_ptr[2];
#if (BL_SPACEDIM == 3)
    amrex::Real dz = real_ptr[3];
#endif
    void* pos_int = static_cast<char*>(buffer) + 4 * sizeof(amrex::Real);
    int* int_ptr = static_cast<int*>( pos_int );
    int nbox = int_ptr[0]; // number of boxes
    int nmfab = int_ptr[1]; // number of multifabs
    // amrex::Print() << "num of Boxes: " << nb  << std::endl;
    // amrex::Print() << "num of MultiFab: " << nmfab  << std::endl;
    // amrex::Print() << std::endl;

    int_ptr = int_ptr + 4;
    // get id_th box lo and hi
    int pos = 6 * id_th;
    lox = int_ptr[pos+0];
    loy = int_ptr[pos+1];
    hix = int_ptr[pos+3];
    hiy = int_ptr[pos+4];
#if (BL_SPACEDIM == 3)
    loz = int_ptr[pos+2];
    hiz = int_ptr[pos+5];
#endif
    // for (int i = 0; i < nb; ++i) {
    //     int pos = i * 6;
    //     amrex::Print() << "Box: " << i << std::endl;
    //     amrex::Print() << "lo: " << "(" << int_ptr[pos + 0] << "," << int_ptr[pos + 1] << "," << int_ptr[pos + 2] << ")" << std::endl;
    //     amrex::Print() << "hi: " << "(" << int_ptr[pos + 3] << "," << int_ptr[pos + 4] << "," << int_ptr[pos + 5] << ")" << std::endl;
    //     amrex::Print() << std::endl;
    // }

    void* pos_data = static_cast<char*>(buffer) + 4 * sizeof(amrex::Real) + (4 + 6 * 4) * sizeof(int);
    amrex::Real** device_data_ptrs = static_cast<amrex::Real**>(pos_data);
    void* phi_old_void = device_data_ptrs[id*nmfab+0];
    void* phi_new_void = device_data_ptrs[id*nmfab+1];
    void* fluxx_void = device_data_ptrs[id*nmfab+2];
    void* fluxy_void = device_data_ptrs[id*nmfab+3];
#if (BL_SPACEDIM == 3)
    void* fluxz_void = device_data_ptrs[id*nmfab+4];
#endif


    // for (int i = 0; i < nb; ++i) {
    //     amrex::Print() << "Box: " << i << std::endl;
    //     for (int j = 0; j < nmfab; ++j) {
    //         amrex::Print() << "GPU memory address of data array " << j << ":" << device_data_ptrs[i*nmfab+j] << std::endl;;
    //     }
    //     amrex::Print() << std::endl;
    // }

}


