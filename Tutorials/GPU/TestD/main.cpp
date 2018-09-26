#include <iostream>
#include <omp.h>

#include <AMReX.H>
#include <AMReX_Device.H>
#include <AMReX_Print.H>
#include <AMReX_Managed.H>

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

using namespace amrex;

int main (int argc, char* argv[])
{

    amrex::Initialize(argc, argv);

    std::cout << std::endl;

#pragma omp parallel
{
    cudaSetDevice(Device::deviceId());

    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    int max_threads = omp_get_max_threads();

    if (thread_id == 0)
    {
      std::cout << "OMP Max Num threads = " << max_threads << std::endl;
      std::cout << "Num OMP threads = " << num_threads << std::endl;
    }

#pragma omp barrier
    std::cout << "OMP Id = " << thread_id << std::endl;
#pragma omp barrier

    if (thread_id == 0)
    {
      std::cout << std::endl;
      IntVect* intvect;
      cudaMallocManaged(&intvect, sizeof(amrex::IntVect));
      *intvect = amrex::IntVect::TheUnitVector();
      IntVect ZeroVector = amrex::IntVect::TheZeroVector();

      amrex::Print() << "intvect before = " << *intvect << std::endl; 

      AMREX_SIMPLE_L_LAUNCH(1,1,
      [=] AMREX_CUDA_DEVICE ()
      {
          *intvect = ZeroVector; 
          printf("invect during = (%i,%i,%i)\n", (*intvect)[0], (*intvect)[1], (*intvect)[2]);
      });

      Device::synchronize();

      amrex::Print() << "intvect after = " << *intvect << std::endl << std::endl;
    }

    #pragma omp barrier

    if (thread_id != 0)
    {
       std::cout << thread_id << ": And I helped!!" << std::endl;
    }

}   

    amrex::Finalize(); 
}
