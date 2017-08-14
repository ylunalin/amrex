
#include <ABL.H>
#include <ABL_F.H>

#include <AMReX_MultiFabUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_BndryData.H>
#include <AMReX_MultiGrid.H>
#include <AMReX_CGSolver.H>
#include <AMReX_Laplacian.H>
#include <AMReX_ABecLaplacian.H>

#include <string>

using namespace amrex;

ABL::ABL ()
{
    // runtime parameters
    {
        ParmParse pp;

        pp.query("n_cell", n_cell);
        pp.query("max_grid_size", max_grid_size);

        std::string bc_type_s{"Dirichlet"};
        pp.query("bc_type", bc_type_s);
        std::transform(bc_type_s.begin(), bc_type_s.end(), bc_type_s.begin(), ::tolower);
        if (bc_type_s == "dirichlet") {
            bc_type = amrex::LinOpBCType::Dirichlet;
        } else if (bc_type_s == "neumann") {
            bc_type = amrex::LinOpBCType::Neumann;            
        } else if (bc_type_s == "periodic") {
            bc_type = amrex::LinOpBCType::interior;
        } else {
            amrex::Abort("Unknown bc_type: "+bc_type_s);
        }
        
        pp.query("bc_value", bc_value);

        pp.query("tol_rel", tol_rel);
        pp.query("tol_abs", tol_abs);
        pp.query("maxiter", maxiter);
        
        pp.query("verbose", verbose);
    }

    int nlev = 1;
    
    ba.resize(nlev);
    geom.resize(nlev);
    dmap.resize(nlev);
    rhs.resize(nlev);
    soln.resize(nlev);
    the_soln.resize(nlev);
    alpha.resize(nlev);
    beta.resize(nlev);

    {
        IntVect dom_lo(0,0,0);
        IntVect dom_hi(n_cell-1,n_cell-1,n_cell-1);
        Box domain(dom_lo,dom_hi);
        ba[0].define(domain);
        ba[0].maxSize(max_grid_size);

        RealBox real_box({0.0,0.0,0.0}, {1.0,1.0,1.0});
        std::array<int,3> is_periodic {0,0,0};
        if (bc_type == amrex::LinOpBCType::interior) {
            is_periodic = {1,1,1};
        }
        geom[0].define(domain, &real_box, CoordSys::cartesian, is_periodic.data());
    }

    dmap[0].define(ba[0]);

    rhs[0].define(ba[0], dmap[0], 1, 0);
    init_rhs();

    alpha[0].define(ba[0], dmap[0], 1, 0);

    beta[0].resize(BL_SPACEDIM);

    for (int idim = 0; idim < BL_SPACEDIM; ++idim) {
        BoxArray nba = ba[0];
        nba.surroundingNodes(idim);
        beta[0][idim].reset(new MultiFab(nba, dmap[0], 1, 0));
    }
    init_coeffs();

    soln[0].define(ba[0], dmap[0], 1, 0);
    the_soln[0].define(ba[0], dmap[0], 1, 0);

    comp_the_solution();
}

void
ABL::init_rhs ()
{
    const int ibnd = static_cast<int>(bc_type);

    int nlev = 1;
    for (int lev = 0; lev < nlev; lev++)
    {
       const Real* dx = geom[lev].CellSize();
       for (MFIter mfi(rhs[lev],true); mfi.isValid(); ++mfi)
       {
           const Box& tbx = mfi.tilebox();
           fort_init_rhs(BL_TO_FORTRAN_BOX(tbx),
                         BL_TO_FORTRAN_ANYD(rhs[lev][mfi]),
                         dx, &a, &b, &sigma, &w, &ibnd);
       }
   }
}

void
ABL::init_coeffs ()
{
    int nlev = 1;
    MultiFab betacc;
    for (int lev = 0; lev < nlev; lev++)
    {
       alpha[lev].setVal(1.0);
       const Real* dx = geom[lev].CellSize();

       betacc.define(alpha[lev].boxArray(), alpha[lev].DistributionMap(), 1, 1);

       for (MFIter mfi(betacc,true); mfi.isValid(); ++mfi)
       {
           const Box& tbx = mfi.growntilebox();
           fort_init_cc_coef(BL_TO_FORTRAN_BOX(tbx),
                             BL_TO_FORTRAN_ANYD(betacc[mfi]),
                             dx, &sigma, &w);
       }

       amrex::average_cellcenter_to_face(GetArrOfPtrs(beta[lev]),
// {beta[lev][0], beta[lev][1], beta[lev][2]},
                                         betacc, geom[0]);
    }
}

void
ABL::comp_the_solution ()
{
    const int ibnd = static_cast<int>(bc_type);

    MultiFab betacc;

    int nlev = 1;
    for (int lev = 0; lev < nlev; lev++)
    {
        const Real* dx = geom[lev].CellSize();
        for (MFIter mfi(the_soln[lev]); mfi.isValid(); ++mfi)
        {
            fort_comp_asol(BL_TO_FORTRAN_ANYD(the_soln[lev][mfi]),
                           dx, &ibnd);
        }
    }
}

void
ABL::solve ()
{
    solve_with_Cpp();

    int nlev = 1;
    MultiFab diff;
    for (int lev = 0; lev < nlev; lev++)
    {
        diff.define(ba[lev], dmap[lev], 1, 0);
        MultiFab::Copy(diff, soln[lev], 0, 0, 1, 0);
        MultiFab::Subtract(diff, the_soln[lev], 0, 0, 1, 0);
        amrex::Print() << "\nMax-norm of the error at level " << lev << " is " << diff.norm0()
                       << "\nMaximum absolute value of the solution is " << the_soln[lev].norm0()
                       << "\nMaximum absolute value of the rhs is " << rhs[lev].norm0()
                       << "\n";
    }
}

void
ABL::solve_with_Cpp ()
{
  BL_PROFILE("solve_with_Cpp()");

  int  lev = 0;
  int nlev = 1;

  const Real* dx = geom[0].CellSize();

  Array<BndryData> bd;
  bd.resize(nlev);
  bd[lev].define(ba[lev], dmap[lev], 1, geom[lev]);
  set_boundary(bd[lev]);

  ABecLaplacian abec_operator(bd[lev], dx);
  abec_operator.setScalars(a, b);
  abec_operator.setCoefficients(alpha[lev], GetArrOfPtrs(beta[lev]));

  MultiGrid mg(abec_operator);
  mg.setVerbose(verbose);
  
//if (fixediter) {
//    mg.setMaxIter(maxiter);
//    mg.setFixedIter(fixediter);
//}

  mg.solve(soln[lev], rhs[lev], tol_rel, tol_abs);
}

void 
ABL::set_boundary(BndryData& bd)
{
  BL_PROFILE("set_boundary()");
  Real bc_value = 0.0;

  int  lev = 0;
  int nlev = 1;
  int comp = 0;

  const Real* dx = geom[lev].CellSize();

  for (int n=0; n<BL_SPACEDIM; ++n) {
    for (MFIter mfi(rhs[lev]); mfi.isValid(); ++mfi ) {
      int i = mfi.index(); 
      
      const Box& bx = mfi.validbox();
      
      // Our default will be that the face of this grid is either touching another grid
      //  across an interior boundary or a periodic boundary.  We will test for the other
      //  cases below.
      {
	// Define the type of boundary conditions to be Dirichlet (even for periodic)
	bd.setBoundCond(Orientation(n, Orientation::low) ,i,comp,LO_DIRICHLET);
	bd.setBoundCond(Orientation(n, Orientation::high),i,comp,LO_DIRICHLET);
	
	// Set the boundary conditions to the cell centers outside the domain
	bd.setBoundLoc(Orientation(n, Orientation::low) ,i,0.5*dx[n]);
	bd.setBoundLoc(Orientation(n, Orientation::high),i,0.5*dx[n]);
      }

      // Now test to see if we should override the above with Dirichlet or Neumann physical bc's
      if (bc_type != amrex::LinOpBCType::interior)
      {

	int ibnd = static_cast<int>(bc_type); // either LO_DIRICHLET or LO_NEUMANN

	// We are on the low side of the domain in coordinate direction n
	if (bx.smallEnd(n) == geom[lev].Domain().smallEnd(n)) {
	  // Set the boundary conditions to live exactly on the faces of the domain
	  bd.setBoundLoc(Orientation(n, Orientation::low) ,i,0.0 );
	  
	  // Set the Dirichlet/Neumann boundary values 
	  bd.setValue(Orientation(n, Orientation::low) ,i, bc_value);
	  
	  // Define the type of boundary conditions 
	  bd.setBoundCond(Orientation(n, Orientation::low) ,i,comp,ibnd);
	}
	
	// We are on the high side of the domain in coordinate direction n
	if (bx.bigEnd(n) == geom[lev].Domain().bigEnd(n)) {
	  // Set the boundary conditions to live exactly on the faces of the domain
	  bd.setBoundLoc(Orientation(n, Orientation::high) ,i,0.0 );
	  
	  // Set the Dirichlet/Neumann boundary values
	  bd.setValue(Orientation(n, Orientation::high) ,i, bc_value);

	  // Define the type of boundary conditions 
	  bd.setBoundCond(Orientation(n, Orientation::high) ,i,comp,ibnd);
	}
      }
    }
  }
}
