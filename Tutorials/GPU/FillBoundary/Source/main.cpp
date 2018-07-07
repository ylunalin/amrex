#include <AMReX_Device.H>
#include <AMReX_Utility.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

using namespace amrex;

void main_main();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    
    main_main();
    
    amrex::Finalize();
    return 0;
}

void main_main ()
{
    // What time is it now?  We'll use this to compute total run time.
    Real strt_time = ParallelDescriptor::second();

    // AMREX_SPACEDIM: number of dimensions
    int n_cell, max_grid_size, nsteps, plot_int;
    Vector<int> is_periodic(AMREX_SPACEDIM,1);  // periodic in all direction by default

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
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

        // Default nsteps to 0, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);

        pp.queryarr("is_periodic", is_periodic);
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

        // This defines a Geometry object
        geom.define(domain,&real_box,CoordSys::cartesian,is_periodic.data());
    }
    DistributionMapping dm(ba);
    int Nghost = 1;
    int Ncomp  = 1;

    MultiFab phi_old(ba, dm, Ncomp, Nghost);

    // .....................
    amrex::Print() << "Before FillBoundary" << std::endl << std::endl;
    amrex::USleep(1);
    // .....................

    phi_old.FillBoundary(geom.periodicity());

    // .....................
    amrex::USleep(1);
    amrex::Print() << "After FillBoundary. Hooray!" << std::endl << std::endl;
    // .....................
}
