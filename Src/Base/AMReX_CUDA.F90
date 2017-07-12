module cuda_module

    use cudafor, only: cuda_stream_kind, cudaEvent
    use amrex_fort_module, only: amrex_real

    implicit none

    integer, parameter :: max_cuda_streams = 100
    integer(kind=cuda_stream_kind) :: cuda_streams(max_cuda_streams)

    integer, save :: cuda_device_id

    ! For timing
    ! use 0-index array
    integer, parameter :: max_cuda_timer =  100
    real(amrex_real) :: elapsed_time(max_cuda_timer)
    character(len=20) :: timer_name(max_cuda_timer)
    logical :: timer_initialized(max_cuda_timer)
    integer :: n_calls(max_cuda_timer)
    type(cudaEvent) :: event_start(max_cuda_timer)
    type(cudaEvent) :: event_stop(max_cuda_timer)
    integer :: n_timer

contains

  subroutine initialize_cuda() bind(c, name='initialize_cuda')

    use cudafor, only: cudaStreamCreate,cudaDeviceSetSharedMemConfig,cudaSharedMemBankSizeEightByte
    
    implicit none

    integer :: i, cudaResult
    ! TODO: for now always assume double-precision floats are used
    cudaResult = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)

    do i = 1, max_cuda_streams
       cudaResult = cudaStreamCreate(cuda_streams(i))
    enddo

    cuda_device_id = 0
    n_timer = 0

  end subroutine initialize_cuda

  ! put all finalization code for CUDA here
  subroutine finalize_cuda() bind(c, name='finalize_cuda')

    use cudafor, only: cudaDeviceReset
    implicit none
    integer :: cudaResult

    ! cudaDeviceReset causes the driver to clean up all state. While
    ! not mandatory in normal operation, it is good practice.  It is also
    ! needed to ensure correct operation when the application is being
    ! profiled. Calling cudaDeviceReset causes all profile data to be
    ! flushed before the application exits
    cudaResult = cudaDeviceReset()

  end subroutine finalize_cuda



  subroutine get_cuda_device_id(id) bind(c, name='get_cuda_device_id')

    implicit none

    integer :: id

    id = cuda_device_id

  end subroutine get_cuda_device_id



  integer function stream_from_index(idx)

    implicit none

    integer :: idx

    ! note that available streams are indexed from 1 to 100
    ! reserve the stream 1 to 10 for special purposes
    if (idx < 0 .and. idx >= -10) then
        stream_from_index = -idx
    else
        ! stream_from_index below ranges from 11 to 100
        stream_from_index = MOD(idx, max_cuda_streams-10) + 11
    endif

  end function stream_from_index



#ifdef BL_SPACEDIM
  subroutine threads_and_blocks(lo, hi, numBlocks, numThreads)

    use cudafor, only: dim3

    implicit none

    integer, intent(in)       :: lo(BL_SPACEDIM), hi(BL_SPACEDIM)
    type(dim3), intent(inout) :: numBlocks, numThreads

    integer :: tile_size(BL_SPACEDIM)

    tile_size = hi - lo + 1

    if (BL_SPACEDIM .eq. 1) then

       numThreads % x = 256
       numThreads % y = 1
       numThreads % z = 1

       numBlocks % x = (tile_size(1) + numThreads % x - 1) / numThreads % x
       numBlocks % y = 1
       numBlocks % z = 1

    else if (BL_SPACEDIM .eq. 2) then

       numThreads % x = 16
       numThreads % y = 16
       numThreads % z = 1

       numBlocks % x = (tile_size(1) + numThreads % x - 1) / numThreads % x
       numBlocks % y = (tile_size(2) + numThreads % y - 1) / numThreads % y
       numBlocks % z = 1

    else

       numThreads % x = 8
       numThreads % y = 8
       numThreads % z = 8

       numBlocks % x = (tile_size(1) + numThreads % x - 1) / numThreads % x
       numBlocks % y = (tile_size(2) + numThreads % y - 1) / numThreads % y
       numBlocks % z = (tile_size(3) + numThreads % z - 1) / numThreads % z

    endif

  end subroutine threads_and_blocks
#endif



  subroutine gpu_malloc(x, sz) bind(c, name='gpu_malloc')

    use cudafor, only: cudaMalloc, c_devptr
    use iso_c_binding, only: c_size_t

    implicit none

    type(c_devptr) :: x
    integer(c_size_t) :: sz

    integer :: cudaResult

    cudaResult = cudaMalloc(x, sz)

  end subroutine gpu_malloc


#ifdef CUDA_ARRAY
  subroutine gpu_malloc_2d(x, pitch, isize, jsize) bind(c, name='gpu_malloc_2d')

    use cudafor, only: cudaMallocPitch, c_devptr
    use iso_c_binding, only: c_size_t

    implicit none

    type(c_devptr) :: x
    integer(c_size_t) :: pitch, isize, jsize

    integer :: cudaResult

    cudaResult = cudaMallocPitch(x, pitch, isize, jsize)

  end subroutine gpu_malloc_2d
#endif

  subroutine gpu_malloc_managed(x, sz) bind(c, name='gpu_malloc_managed')

    use cudafor, only: cudaMallocManaged, cudaMemAttachGlobal, c_devptr
    use iso_c_binding, only: c_size_t

    implicit none

    type(c_devptr) :: x
    integer(c_size_t) :: sz

    integer :: cudaResult

    cudaResult = cudaMallocManaged(x, sz, cudaMemAttachGlobal)

  end subroutine gpu_malloc_managed



  subroutine gpu_free(x) bind(c, name='gpu_free')

    use cudafor, only: cudaFree, c_devptr

    implicit none

    type(c_devptr), value :: x

    integer :: cudaResult

    cudaResult = cudaFree(x)

  end subroutine gpu_free

  subroutine cpu_free_pinned(x) bind(c, name='cpu_free_pinned')

    use cudafor, only: cudaFreeHost, c_ptr

    implicit none

    type(c_ptr), value :: x

    integer :: cudaResult

    cudaResult = cudaFreeHost(x)

  end subroutine cpu_free_pinned



  subroutine gpu_htod_memcpy_async(p_d, p_h, sz, idx) bind(c, name='gpu_htod_memcpy_async')

    use cudafor, only: cudaMemcpyAsync, cudaMemcpyHostToDevice, c_devptr, cuda_stream_kind
    use iso_c_binding, only: c_ptr, c_size_t

    implicit none

    type(c_devptr), value :: p_d
    type(c_ptr), value :: p_h
    integer(c_size_t) :: sz
    integer :: idx

    integer :: s
    integer :: cudaResult

    s = stream_from_index(idx)

    cudaResult = cudaMemcpyAsync(p_d, p_h, sz, cudaMemcpyHostToDevice, cuda_streams(s))

  end subroutine gpu_htod_memcpy_async



  subroutine gpu_dtoh_memcpy_async(p_h, p_d, sz, idx) bind(c, name='gpu_dtoh_memcpy_async')

    use cudafor, only: cudaMemcpyAsync, cudaMemcpyDeviceToHost, c_devptr
    use iso_c_binding, only: c_ptr, c_size_t

    implicit none

    type(c_ptr), value :: p_h
    type(c_devptr), value :: p_d
    integer(c_size_t) :: sz
    integer :: idx

    integer :: s
    integer :: cudaResult

    s = stream_from_index(idx)

    cudaResult = cudaMemcpyAsync(p_h, p_d, sz, cudaMemcpyDeviceToHost, cuda_streams(s))

  end subroutine gpu_dtoh_memcpy_async
 
#ifdef CUDA_ARRAY
  subroutine gpu_htod_memcpy_2d_async(p_d, pitch_d, p_h, pitch_h, isize, jsize, idx) bind(c, name='gpu_htod_memcpy_2d_async')

    use cudafor, only: cudaMemcpy2DAsync, cudaMemcpyHostToDevice, c_devptr, cuda_stream_kind
    use iso_c_binding, only: c_ptr, c_size_t

    implicit none

    type(c_devptr), value :: p_d
    type(c_ptr), value :: p_h
    integer(c_size_t) :: pitch_d, pitch_h, isize, jsize
    integer :: idx

    integer :: s
    integer :: cudaResult

    s = stream_from_index(idx)

    cudaResult = cudaMemcpy2DAsync(p_d, pitch_d, p_h, pitch_h, isize, jsize, cudaMemcpyHostToDevice, cuda_streams(s))

  end subroutine gpu_htod_memcpy_2d_async
#endif

#ifdef CUDA_ARRAY
  subroutine gpu_dtoh_memcpy_2d_async(p_h, pitch_h, p_d, pitch_d, isize, jsize, idx) bind(c, name='gpu_dtoh_memcpy_2d_async')

    use cudafor, only: cudaMemcpy2DAsync, cudaMemcpyDeviceToHost, c_devptr, cuda_stream_kind
    use iso_c_binding, only: c_ptr, c_size_t

    implicit none

    type(c_devptr), value :: p_d
    type(c_ptr), value :: p_h
    integer(c_size_t) :: pitch_d, pitch_h, isize, jsize
    integer :: idx

    integer :: s
    integer :: cudaResult

    s = stream_from_index(idx)

    cudaResult = cudaMemcpy2DAsync(p_h, pitch_h, p_d, pitch_d, isize, jsize, cudaMemcpyDeviceToHost, cuda_streams(s))

  end subroutine gpu_dtoh_memcpy_2d_async
#endif


  subroutine gpu_htod_memprefetch_async(p, sz, idx) bind(c, name='gpu_htod_memprefetch_async')

    use cudafor, only: cudaMemPrefetchAsync, c_devptr
    use iso_c_binding, only: c_size_t

    implicit none

    type(c_devptr) :: p
    integer(c_size_t) :: sz
    integer :: idx

    integer :: s
    integer :: cudaResult

    s = stream_from_index(idx)

    cudaResult = cudaMemPrefetchAsync(p, sz, cuda_device_id, cuda_streams(s))

  end subroutine gpu_htod_memprefetch_async



  subroutine gpu_dtoh_memprefetch_async(p, sz, idx) bind(c, name='gpu_dtoh_memprefetch_async')

    use cudafor, only: cudaMemPrefetchAsync, c_devptr, cudaCpuDeviceId
    use iso_c_binding, only: c_size_t

    implicit none

    type(c_devptr) :: p
    integer(c_size_t) :: sz
    integer :: idx

    integer :: s
    integer :: cudaResult

    s = stream_from_index(idx)

    cudaResult = cudaMemPrefetchAsync(p, sz, cudaCpuDeviceId, cuda_streams(s))

  end subroutine gpu_dtoh_memprefetch_async



  subroutine gpu_synchronize() bind(c, name='gpu_synchronize')

    use cudafor, only: cudaDeviceSynchronize

    implicit none

    integer :: cudaResult

    cudaResult = cudaDeviceSynchronize()

  end subroutine gpu_synchronize

  subroutine gpu_synchronize_stream(idx) bind(c, name='gpu_synchronize_stream')

    use cudafor, only: cudaStreamSynchronize

    implicit none

    integer :: cudaResult
    integer :: s, idx

    s = stream_from_index(idx)

    cudaResult = cudaStreamSynchronize(cuda_streams(s))

  end subroutine gpu_synchronize_stream



  subroutine mem_advise_set_preferred(p, sz, device) bind(c, name='mem_advise_set_preferred')

    use cudafor, only: c_devptr, cudaMemAdvise, cudaMemAdviseSetPreferredLocation
    use iso_c_binding, only: c_size_t, c_int

    type(c_devptr) :: p
    integer(c_size_t) :: sz
    integer(c_int) :: device

    integer :: cudaResult

    cudaResult = cudaMemAdvise(p, sz, cudaMemAdviseSetPreferredLocation, device)

  end subroutine mem_advise_set_preferred

  subroutine cpu_malloc_pinned(x, sz) bind(c, name='cpu_malloc_pinned')

    use cudafor, only: cudaHostAlloc, c_ptr, cudaHostAllocDefault
    use iso_c_binding, only: c_size_t

    implicit none

    type(c_ptr) :: x
    ! TODO: cudaHostAlloc can only take sz as integer(kind=4) 
    ! not integer(kind=c_size_t) 
    ! but sz passed to cpu_malloc_pinned is of type size_t
    ! integer(c_size_t) :: sz
    integer(kind=4) :: sz

    integer :: cudaResult

    cudaResult = cudaHostAlloc(x, sz, cudaHostAllocDefault)

  end subroutine cpu_malloc_pinned

    ! put a timer with name t_name and ID id
    ! id should be in the range 1:max_cuda_timer
    subroutine timer_take(t_name, id)
        implicit none
        integer, intent(in) :: id
        character(len=10), intent(in) :: t_name
        if (timer_initialized(id) .eq. .false.) then
            timer_name(id) = t_name
            elapsed_time(id) = 0.0
            n_calls(id) = 0
            timer_initialized(id) = .true.
        endif
    end subroutine timer_take

    subroutine timer_start(id)
        use cudafor, only: cudaEventCreate, cudaEventRecord
        implicit none
        integer :: id, cuda_result
        cuda_result = cudaEventCreate(event_start(id))
        cuda_result = cudaEventCreate(event_stop(id))
        cuda_result = cudaEventRecord(event_start(id), 0)
        n_calls(id) = n_calls(id) + 1
    end subroutine timer_start

    subroutine timer_stop(id)
        use cudafor, only: cudaEventRecord, cudaEventDestroy, & 
            cudaEventSynchronize, cudaEventElapsedTime
        use amrex_fort_module, only: amrex_real
        implicit none
        integer :: id, cuda_result
        real :: local_time
        cuda_result = cudaEventRecord(event_stop(id), 0)
        cuda_result = cudaEventSynchronize(event_stop(id))
        cuda_result = cudaEventElapsedTime(local_time, event_start(id), event_stop(id))
        cuda_result = cudaEventDestroy(event_start(id))
        cuda_result = cudaEventDestroy(event_stop(id))
        elapsed_time(id) = elapsed_time(id) + local_time/1000
    end subroutine timer_stop

    ! returned time is in second
    subroutine get_cuda_time(id, time) bind(c, name='get_cuda_time')
        use amrex_fort_module, only: amrex_real
        implicit none
        integer, intent(in) :: id
        real(amrex_real), intent(out) :: time
        time = elapsed_time(id)
    end subroutine get_cuda_time

    subroutine get_cuda_num_calls(id, n) bind(c, name='get_cuda_num_calls')
        use amrex_fort_module, only: amrex_real
        implicit none
        integer, intent(in) :: id
        integer, intent(out) :: n
        n= n_calls(id)
    end subroutine get_cuda_num_calls

    subroutine get_stream(id, pStream) bind(c, name='get_stream')
        implicit none
        integer, intent(in) :: id
        integer(kind=cuda_stream_kind),intent(out) :: pStream
        integer :: s

        s = stream_from_index(id)
        pStream = cuda_streams(s)
    end subroutine get_stream

end module cuda_module
