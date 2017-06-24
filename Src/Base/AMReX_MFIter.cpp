
#include <cstring>
#include <stdlib.h>


#include <AMReX_MFIter.H>
#include <AMReX_FabArray.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Device.H>
#include <AMReX_MultiFab.H>

namespace amrex {

MFIter::MFIter (const FabArrayBase& fabarray_, 
		unsigned char       flags_)
    :
    fabArray(fabarray_),
    tile_size((flags_ & Tiling) ? FabArrayBase::mfiter_tile_size : IntVect::TheZeroVector()),
    flags(flags_),
    index_map(nullptr),
    local_index_map(nullptr),
    tile_array(nullptr),
    local_tile_index_map(nullptr),
    num_local_tiles(nullptr)
{
    Initialize();
}

MFIter::MFIter (const FabArrayBase& fabarray_, 
		bool                do_tiling_)
    :
    fabArray(fabarray_),
    tile_size((do_tiling_) ? FabArrayBase::mfiter_tile_size : IntVect::TheZeroVector()),
    flags(do_tiling_ ? Tiling : 0),
    index_map(nullptr),
    local_index_map(nullptr),
    tile_array(nullptr),
    local_tile_index_map(nullptr),
    num_local_tiles(nullptr)
{
    Initialize();
}

MFIter::MFIter (const FabArrayBase& fabarray_, 
		const IntVect&      tilesize_, 
		unsigned char       flags_)
    :
    fabArray(fabarray_),
    tile_size(tilesize_),
    flags(flags_ | Tiling),
    index_map(nullptr),
    local_index_map(nullptr),
    tile_array(nullptr),
    local_tile_index_map(nullptr),
    num_local_tiles(nullptr)
{
    Initialize();
}

MFIter::MFIter (const BoxArray& ba, const DistributionMapping& dm, unsigned char flags_)
    :
    m_fa(new FabArray<FArrayBox>(ba, dm, 1, 0, MFInfo().SetAlloc(false))),
    fabArray(*m_fa),
    tile_size((flags_ & Tiling) ? FabArrayBase::mfiter_tile_size : IntVect::TheZeroVector()),
    flags(flags_),
    index_map(nullptr),
    local_index_map(nullptr),
    tile_array(nullptr),
    local_tile_index_map(nullptr),
    num_local_tiles(nullptr)
{
    Initialize();
}

MFIter::MFIter (const BoxArray& ba, const DistributionMapping& dm, bool do_tiling_)
    :
    m_fa(new FabArray<FArrayBox>(ba, dm, 1, 0, MFInfo().SetAlloc(false))),
    fabArray(*m_fa),
    tile_size((do_tiling_) ? FabArrayBase::mfiter_tile_size : IntVect::TheZeroVector()),
    flags(do_tiling_ ? Tiling : 0),
    index_map(nullptr),
    local_index_map(nullptr),
    tile_array(nullptr),
    local_tile_index_map(nullptr),
    num_local_tiles(nullptr)
{
    Initialize();
}


MFIter::MFIter (const BoxArray& ba, const DistributionMapping& dm,
                const IntVect& tilesize_, unsigned char flags_)
    :
    m_fa(new FabArray<FArrayBox>(ba, dm, 1, 0, MFInfo().SetAlloc(false))),
    fabArray(*m_fa),
    tile_size(tilesize_),
    flags(flags_ | Tiling),
    index_map(nullptr),
    local_index_map(nullptr),
    tile_array(nullptr),
    local_tile_index_map(nullptr),
    num_local_tiles(nullptr)
{
    Initialize();
}

#ifdef CUDA
MFIter::MFIter (const FabArrayBase& fabarray_, 
                const MFIterRegister& mfi_reg,
		unsigned char       flags_)
    :
    fabArray(fabarray_),
    tile_size((flags_ & Tiling) ? FabArrayBase::mfiter_tile_size : IntVect::TheZeroVector()),
    flags(flags_),
    index_map(nullptr),
    local_index_map(nullptr),
    tile_array(nullptr),
    local_tile_index_map(nullptr),
    num_local_tiles(nullptr)
{
    Initialize();
    // send all fab data registered in MFIterRegister to device
    mfi_reg.allFabToDevice();
}
#endif


MFIter::~MFIter ()
{
#if BL_USE_TEAM
    if ( ! (flags & NoTeamBarrier) )
	ParallelDescriptor::MyTeam().MemoryBarrier();
#endif
    // releaseDeviceData();
}

void 
MFIter::Initialize ()
{
    if (flags & SkipInit) {
	return;
    }
    else if (flags & AllBoxes)  // a very special case
    {
	index_map    = &(fabArray.IndexArray());
	currentIndex = 0;
	beginIndex   = 0;
	endIndex     = index_map->size();
    }
    else
    {
	const FabArrayBase::TileArray* pta = fabArray.getTileArray(tile_size);
	
	index_map            = &(pta->indexMap);
	local_index_map      = &(pta->localIndexMap);
	tile_array           = &(pta->tileArray);
	local_tile_index_map = &(pta->localTileIndexMap);
	num_local_tiles      = &(pta->numLocalTiles);

	{
	    int rit = 0;
	    int nworkers = 1;
#ifdef BL_USE_TEAM
	    if (ParallelDescriptor::TeamSize() > 1) {
		if ( tile_size == IntVect::TheZeroVector() ) {
		    // In this case the TileArray contains only boxes owned by this worker.
		    // So there is no sharing going on.
		    rit = 0;
		    nworkers = 1;
		} else {
		    rit = ParallelDescriptor::MyRankInTeam();
		    nworkers = ParallelDescriptor::TeamSize();
		}
	    }
#endif

	    int ntot = index_map->size();
	    
	    if (nworkers == 1)
	    {
		beginIndex = 0;
		endIndex = ntot;
	    }
	    else
	    {
		int nr   = ntot / nworkers;
		int nlft = ntot - nr * nworkers;
		if (rit < nlft) {  // get nr+1 items
		    beginIndex = rit * (nr + 1);
		    endIndex = beginIndex + nr + 1;
		} else {           // get nr items
		    beginIndex = rit * nr + nlft;
		    endIndex = beginIndex + nr;
		}
	    }
	}
	
#ifdef _OPENMP
	int nthreads = omp_get_num_threads();
	if (nthreads > 1)
	{
	    int tid = omp_get_thread_num();
	    int ntot = endIndex - beginIndex;
	    int nr   = ntot / nthreads;
	    int nlft = ntot - nr * nthreads;
	    if (tid < nlft) {  // get nr+1 items
		beginIndex += tid * (nr + 1);
		endIndex = beginIndex + nr + 1;
	    } else {           // get nr items
		beginIndex += tid * nr + nlft;
		endIndex = beginIndex + nr;
	    }	    
	}
#endif

	currentIndex = beginIndex;

	typ = fabArray.boxArray().ixType();
    }
}

Box 
MFIter::tilebox () const
{ 
    BL_ASSERT(tile_array != 0);
    Box bx((*tile_array)[currentIndex]);
    if (! typ.cellCentered())
    {
	bx.convert(typ);
	const Box& vbx = validbox();
	const IntVect& Big = vbx.bigEnd();
	for (int d=0; d<BL_SPACEDIM; ++d) {
	    if (typ.nodeCentered(d)) { // validbox should also be nodal in d-direction.
		if (bx.bigEnd(d) < Big[d]) {
		    bx.growHi(d,-1);
		}
	    }
	}
    }
    return bx;
}

Box
MFIter::tilebox (const IntVect& nodal) const
{
    BL_ASSERT(tile_array != 0);
    Box bx((*tile_array)[currentIndex]);
    const IndexType new_typ {nodal};
    if (! new_typ.cellCentered())
    {
	bx.setType(new_typ);
	const Box& valid_cc_box = amrex::enclosedCells(validbox());
	const IntVect& Big = valid_cc_box.bigEnd();
	for (int d=0; d<BL_SPACEDIM; ++d) {
	    if (new_typ.nodeCentered(d)) { // validbox should also be nodal in d-direction.
		if (bx.bigEnd(d) == Big[d]) {
		    bx.growHi(d,1);
		}
	    }
	}
    }
    return bx;
}

Box
MFIter::nodaltilebox (int dir) const 
{ 
    BL_ASSERT(dir < BL_SPACEDIM);
    BL_ASSERT(tile_array != 0);
    Box bx((*tile_array)[currentIndex]);
    bx.convert(typ);
    const Box& vbx = validbox();
    const IntVect& Big = vbx.bigEnd();
    int d0, d1;
    if (dir < 0) {
	d0 = 0;
	d1 = BL_SPACEDIM-1;
    } else {
	d0 = d1 = dir;
    }
    for (int d=d0; d<=d1; ++d) {
	if (typ.cellCentered(d)) { // validbox should also be cell-centered in d-direction.
	    bx.surroundingNodes(d);
	    if (bx.bigEnd(d) <= Big[d]) {
		bx.growHi(d,-1);
	    }
	}
    }
    return bx;
}

// Note that a small negative ng is supported.
Box 
MFIter::growntilebox (int ng) const 
{
    Box bx = tilebox();
    if (ng < -100) ng = fabArray.nGrow();
    const Box& vbx = validbox();
    for (int d=0; d<BL_SPACEDIM; ++d) {
	if (bx.smallEnd(d) == vbx.smallEnd(d)) {
	    bx.growLo(d, ng);
	}
	if (bx.bigEnd(d) == vbx.bigEnd(d)) {
	    bx.growHi(d, ng);
	}
    }
    return bx;
}

Box
MFIter::grownnodaltilebox (int dir, int ng) const
{
    BL_ASSERT(dir < BL_SPACEDIM);
    Box bx = nodaltilebox(dir);
    if (ng < -100) ng = fabArray.nGrow();
    const Box& vbx = validbox();
    for (int d=0; d<BL_SPACEDIM; ++d) {
	if (bx.smallEnd(d) == vbx.smallEnd(d)) {
	    bx.growLo(d, ng);
	}
	if (bx.bigEnd(d) >= vbx.bigEnd(d)) {
	    bx.growHi(d, ng);
	}
    }
    return bx;
}

void
MFIter::operator++ () {

    ++currentIndex;

    // releaseDeviceData();

}

void
MFIter::releaseDeviceData() {
    if (Device::inDeviceLaunchRegion()) {
	for (int i = 0; i < registered_fabs.size(); ++i)
	    registered_fabs[i]->toHost(registered_fabs_indices[i]);
	registered_fabs.clear();
	registered_fabs_indices.clear();
    }
}

MFGhostIter::MFGhostIter (const FabArrayBase& fabarray)
    :
    MFIter(fabarray, (unsigned char)(SkipInit|Tiling))
{
    Initialize();
}

void
MFGhostIter::Initialize ()
{
    int rit = 0;
    int nworkers = 1;
#ifdef BL_USE_TEAM
    if (ParallelDescriptor::TeamSize() > 1) {
	rit = ParallelDescriptor::MyRankInTeam();
	nworkers = ParallelDescriptor::TeamSize();
    }
#endif

    int tid = 0;
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_num_threads();
    if (nthreads > 1)
	tid = omp_get_thread_num();
#endif

    int npes = nworkers*nthreads;
    int pid = rit*nthreads+tid;

    BoxList alltiles;
    Array<int> allindex;
    Array<int> alllocalindex;

    for (int i=0; i < fabArray.IndexArray().size(); ++i) {
	int K = fabArray.IndexArray()[i];
	const Box& vbx = fabArray.box(K);
	const Box& fbx = fabArray.fabbox(K);

	const BoxList& diff = amrex::boxDiff(fbx, vbx);
	
	for (BoxList::const_iterator bli = diff.begin(); bli != diff.end(); ++bli) {
	    BoxList tiles(*bli, FabArrayBase::mfghostiter_tile_size);
	    int nt = tiles.size();
	    for (int it=0; it<nt; ++it) {
		allindex.push_back(K);
		alllocalindex.push_back(i);
	    }
	    alltiles.catenate(tiles);
	}
    }

    int n_tot_tiles = alltiles.size();
    int navg = n_tot_tiles / npes;
    int nleft = n_tot_tiles - navg*npes;
    int ntiles = navg;
    if (pid < nleft) ntiles++;

    // how many tiles should we skip?
    int nskip = pid*navg + std::min(pid,nleft);
    BoxList::const_iterator bli = alltiles.begin();
    for (int i=0; i<nskip; ++i) ++bli;

    lta.indexMap.reserve(ntiles);
    lta.localIndexMap.reserve(ntiles);
    lta.tileArray.reserve(ntiles);

    for (int i=0; i<ntiles; ++i) {
	lta.indexMap.push_back(allindex[i+nskip]);
	lta.localIndexMap.push_back(alllocalindex[i+nskip]);
	lta.tileArray.push_back(*bli++);
    }

    currentIndex = beginIndex = 0;
    endIndex = lta.indexMap.size();

    lta.nuse = 0;
    index_map       = &(lta.indexMap);
    local_index_map = &(lta.localIndexMap);
    tile_array      = &(lta.tileArray);
}

/*
 * member functions for MFIterRegister
 */
void MFIterRegister::closeRegister() {
    int nBox = m_mf_v[0]->local_size();
    int nFabArray = m_mf_v.size();
    BL_ASSERT(nFabArray != 0);
    /*
     * In each Box, we might have multiple FArrayBox.data();
     * What's stored in the buffer (in order):
     * dt, dx, dy, dz: (4 * sizeof(amrex::Real) bytes)
     * nBox, nFabArray (2 * sizeof(int) bytes + 2 * sizeof(int) bytes for memory alingment)
     * Box1.loVect, Box1.hiVect, Box2.loVect, Box2.hiVect, ... (6 * nBox * sizeof(int) bytes)
     * For box1 (corresponding to fab1), we need:
     *      pt to MultiFab1.fab1.data(), pt to MultiFab2.fab1.data() ... (nFabArray * sizeof(void*) bytes)
     * For box2 (corresponding to fab2), we need:
     *      pt to MultiFab1.fab2.data(), pt to MultiFab2.fab2.data() ... (nFabArray * sizeof(void*) bytes)
     * For box3 ...
     * ...
     *
     * So the total nubmer of bytes is as below
     */ 
    std::size_t sz = 4 * sizeof(amrex::Real) + (4 + 6 * nBox) * sizeof(int) + nBox * nFabArray * sizeof(void*);
    cpu_malloc_pinned(&buffer, &sz);
    // write data to the buffer
    amrex::Real* real_num = static_cast<amrex::Real*>(buffer);
    real_num[0] = dt;
    real_num[1] = dx;
    real_num[2] = dy;
    real_num[3] = dz;
    void* pos_int = static_cast<char*>(buffer) + 4 * sizeof(amrex::Real);
    int* int_ptr = static_cast<int*>( pos_int );
    int_ptr[0] = nBox;
    int_ptr[1] = nFabArray;
    // not used
    // int_ptr[2] = ;
    // int_ptr[3] = ;
    
    int_ptr = int_ptr + 4;
    void* pos_data = static_cast<char*>(buffer) + 4 * sizeof(amrex::Real) + (4 + 6 * nBox) * sizeof(int);
    amrex::Real** device_data_ptrs = static_cast<amrex::Real**>(pos_data);
    // TODO: don't need to construct iterator?
    {
    int i = 0;
    for (MFIter mfi(*m_mf_v[0]); mfi.isValid(); ++ mfi) {
        const Box& bx = mfi.validbox();
        int pos = i * 6;  
        int_ptr[pos+0] = bx.loVect()[0];
        int_ptr[pos+1] = bx.loVect()[1];
#if (BL_SPACEDIM == 3)
        int_ptr[pos+2] = bx.loVect()[2];
#endif
        int_ptr[pos+3] = bx.hiVect()[0];
        int_ptr[pos+4] = bx.hiVect()[1];
#if (BL_SPACEDIM == 3)
        int_ptr[pos+5] = bx.hiVect()[2];
#endif
        for (int jmultifabs = 0; jmultifabs < nFabArray; ++jmultifabs) {
            // device_data_ptrs[i * nFabArray + jmultifabs] = (*m_mf_v[jmultifabs])[mfi].devicePtr(); 
            device_data_ptrs[i * nFabArray + jmultifabs] = (*m_mf_v[jmultifabs])[mfi].dataPtr(); 
        }
        ++i;
    }
    }

    gpu_malloc(&buffer_d, &sz);

    // send buffer to device
    // TODO: how to create different iter_id here for different MFIter
    gpu_htod_memcpy_async(buffer_d, buffer, &sz, &mfIter_id);
}

void MFIterRegister::printInfo() {
    amrex::Print() << "Print information in MFIter::buffer ..." << std::endl;
    char* data_real = static_cast<char*>(buffer);
    amrex::Real dt, dx, dy;
    std::memcpy(&dt, data_real                        , sizeof(amrex::Real));
    std::memcpy(&dx, data_real +   sizeof(amrex::Real), sizeof(amrex::Real));
    std::memcpy(&dy, data_real + 2*sizeof(amrex::Real), sizeof(amrex::Real));
#if (BL_SPACEDIM == 3)
    amrex::Real dz;
    std::memcpy(&dz, data_real + 3*sizeof(amrex::Real), sizeof(amrex::Real));
#endif
    amrex::Print() << "dt: " << dt << std::endl;
    amrex::Print() << "dx: " << dx << std::endl;
    amrex::Print() << "dy: " << dy << std::endl;
#if (BL_SPACEDIM == 3)
    amrex::Print() << "dz: " << dz << std::endl;
#endif
    char* data_int = data_real + 4 * sizeof(amrex::Real);
    int nb, nmfab;
    std::memcpy(&nb,    data_int              , sizeof(int));
    std::memcpy(&nmfab, data_int + sizeof(int), sizeof(int));
    amrex::Print() << "num of Boxes: " << nb  << std::endl;
    amrex::Print() << "num of MultiFab: " << nmfab  << std::endl;
    amrex::Print() << std::endl;

    data_int = data_int + 4 * sizeof(int);
    for (int i = 0; i < nb; ++i) {
        int pos = i * 6;
        int lox, loy;
        int hix, hiy;
        std::memcpy(&lox, data_int + (pos + 0) * sizeof(int), sizeof(int));
        std::memcpy(&loy, data_int + (pos + 1) * sizeof(int), sizeof(int));
        std::memcpy(&hix, data_int + (pos + 3) * sizeof(int), sizeof(int));
        std::memcpy(&hiy, data_int + (pos + 4) * sizeof(int), sizeof(int));
#if (BL_SPACEDIM == 3)
        int loz, hiz;
        std::memcpy(&loz, data_int + (pos + 2) * sizeof(int), sizeof(int));
        std::memcpy(&hiz, data_int + (pos + 5) * sizeof(int), sizeof(int));
#endif
        amrex::Print() << "Box: " << i << std::endl;
        amrex::Print() << "lo: " << "(" << lox << "," << loy 
#if (BL_SPACEDIM == 3)
            << "," << loz
#endif
            << ")" << std::endl;
        amrex::Print() << "hi: " << "(" << hix << "," << hiy 
#if (BL_SPACEDIM == 3)
            << "," << hiz
#endif
            << ")" << std::endl;
        amrex::Print() << std::endl;
    }

    char* data_pointer = data_real + 4 * sizeof(amrex::Real) + (4 + 6 * nb) * sizeof(int);
    // void* pos_data = static_cast<char*>(buffer) + 4 * sizeof(amrex::Real) + (4 + 6 * 4) * sizeof(int);
    // amrex::Real** device_data_ptrs = static_cast<amrex::Real**>(pos_data);

    for (int i = 0; i < nb; ++i) {
        amrex::Print() << "Box: " << i << std::endl;
        for (int j = 0; j < nmfab; ++j) {
            char* address;
            std::memcpy(&address, data_pointer + (i*nmfab+j)*sizeof(char*), sizeof(char*));
            // size_t address;
            // std::memcpy(&address, data_pointer + (i*nmfab+j)*sizeof(void*), sizeof(int));
            amrex::Print() << "GPU memory address of data array " << j << ":" << (size_t) address << std::endl;;
        }
        amrex::Print() << std::endl;
    }
}

void MFIterRegister::allFabToDevice() const {
    for (std::vector<MultiFab*>::const_iterator it = m_mf_v.begin(); it != m_mf_v.end(); ++it) {
        MultiFab& mf = **it;
	for ( MFIter mfi(mf); mfi.isValid(); ++mfi ) {
	    const int idx = mfi.LocalIndex();
            mf[mfi].toDevice(idx);
	}
    }
}

}
