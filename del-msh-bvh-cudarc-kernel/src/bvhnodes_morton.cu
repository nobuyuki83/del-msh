#include "mat4_col_major.h"

__device__
uint32_t device_ExpandBits(uint32_t v)
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__
unsigned int device_MortonCode(float x, float y, float z)
{
  auto ix = (uint32_t)fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
  auto iy = (uint32_t)fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
  auto iz = (uint32_t)fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
  //  std::cout << std::bitset<10>(ix) << " " << std::bitset<10>(iy) << " " << std::bitset<10>(iz) << std::endl;
  ix = device_ExpandBits(ix);
  iy = device_ExpandBits(iy);
  iz = device_ExpandBits(iz);
  //  std::cout << std::bitset<30>(ix) << " " << std::bitset<30>(iy) << " " << std::bitset<30>(iz) << std::endl;
  return ix * 4 + iy * 2 + iz;
}

__device__
int device_Delta(int i, int j, const unsigned int* sortedMC, int nMC)
{
  if ( j<0 || j >= nMC ){ return -1; }
  return __clz(sortedMC[i] ^ sortedMC[j]);
}

__device__
int2 device_MortonCode_DeterminRange(
    const unsigned int* sortedMC,
    int nMC,
    int imc)
{
  if( imc == 0 ){ return make_int2(0,nMC-1); }
  // ----------------------
  const std::uint32_t mc0 = sortedMC[imc-1];
  const std::uint32_t mc1 = sortedMC[imc+0];
  const std::uint32_t mc2 = sortedMC[imc+1];
  if( mc0 == mc1 && mc1 == mc2 ){ // for hash value collision
    int jmc=imc+1;
    while (jmc<nMC-1){
      jmc += 1;
      if( sortedMC[jmc] != mc1 ) {
        return make_int2(imc,jmc-1);
      }
    }
    return make_int2(imc,jmc);
  }
  int d = device_Delta(imc, imc + 1, sortedMC, nMC) - device_Delta(imc, imc - 1, sortedMC, nMC);
  d = d > 0 ? 1 : -1;

  //compute the upper bound for the length of the range
  const int delta_min = device_Delta(imc, imc - d, sortedMC, nMC);
  int lmax = 2;
  while (device_Delta(imc, imc + lmax*d, sortedMC, nMC)>delta_min)
  {
    lmax = lmax * 2;
  }

  //find the other end using binary search
  int l = 0;
  for (int t = lmax / 2; t >= 1; t /= 2)
  {
    if (device_Delta(imc, imc + (l + t)*d, sortedMC, nMC)>delta_min)
    {
      l = l + t;
    }
  }
  int j = imc + l*d;

  int2 range = make_int2(-1,-1);
  if (imc <= j) { range.x = imc; range.y = j; }
  else { range.x = j; range.y = imc; }
  return range;
}

__device__
int device_MortonCode_FindSplit(
    const unsigned int* sortedMC,
    unsigned int iMC_start,
    unsigned int iMC_last)
{
  //return -1 if there is only
  //one primitive under this node.
  if (iMC_start == iMC_last) { return -1; }

  // ------------------------------
  const int common_prefix = __clz(sortedMC[iMC_start] ^ sortedMC[iMC_last]);

  //handle duplicated morton code
  if (common_prefix == 32 ){ return iMC_start; } // sizeof(std::uint32_t)*8

  // Use binary search to find where the next bit differs.
  // Specifically, we are looking for the highest object that
  // shares more than commonPrefix bits with the first one.
  const std::uint32_t mcStart = sortedMC[iMC_start];
  int iMC_split = iMC_start; // initial guess
  int step = iMC_last - iMC_start;
  do
  {
    step = (step + 1) >> 1; // exponential decrease
    const int newSplit = iMC_split + step; // proposed new position
    if (newSplit < iMC_last){
      const unsigned int splitCode = sortedMC[newSplit];
      int splitPrefix = __clz(mcStart ^ splitCode);
      if (splitPrefix > common_prefix){
        iMC_split = newSplit; // accept proposal
      }
    }
  }
  while (step > 1);
  return iMC_split;
}

extern "C" {

__global__
void tri2cntr(
  float* tri2cntr,
  uint32_t num_tri,
  const uint32_t* tri2vtx,
  const float* vtx2xyz)
{
  const unsigned int i_tri = blockDim.x * blockIdx.x + threadIdx.x;
  if (i_tri >= num_tri) return;
  //
  const uint32_t* node2vtx = tri2vtx + i_tri*3;
  const float ratio = 1.0/3.0;
  for(int i_dim=0;i_dim<3;++i_dim){
    float sum = 0.;
    for(uint32_t i_node=0;i_node<3;++i_node){
        sum += vtx2xyz[node2vtx[i_node]*3+i_dim];
    }
    tri2cntr[i_tri*3+i_dim] = sum * ratio;
  }
}

__global__
void vtx2morton(
    uint32_t num_vtx,
    const float* vtx2xyz,
    const float* transform_cntr2uni,
    uint32_t* vtx2morton)
{
  const unsigned int i_vtx = blockDim.x * blockIdx.x + threadIdx.x;
  if (i_vtx >= num_vtx) return;
  //
  const float* xyz0 = vtx2xyz + i_vtx * 3;
  auto xyz1 = mat4_col_major::transform_homogeneous(transform_cntr2uni, xyz0);
  // printf("%f %f %f\n", xyz1[0], xyz1[1], xyz1[2]);
  assert(xyz1[0]>-1.0e-7f && xyz1[0]<1.0f+1.e-7f);
  assert(xyz1[1]>-1.0e-7f && xyz1[1]<1.0f+1.e-7f);
  assert(xyz1[2]>-1.0e-7f && xyz1[2]<1.0f+1.e-7f);
  uint32_t mc = device_MortonCode(xyz1[0], xyz1[1], xyz1[2]);
  vtx2morton[i_vtx] = mc;
}

__global__
void kernel_MortonCode_BVHTopology(
    const uint32_t nMC,
    uint32_t* dNodeBVH,
    const uint32_t *dSortedMC,
    const uint32_t *dSortedId)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= nMC-1) return;
  const unsigned int ini = idx;
  const unsigned int nni = nMC-1;
  // -------------------------------
  const int2 range = device_MortonCode_DeterminRange(dSortedMC,nMC,ini);
  const int isplit = device_MortonCode_FindSplit(dSortedMC,range.x,range.y);
  // printf("%d --> %d %d  %d\n",ini, range.x, range.y, isplit);
  // -------------------------------
  if( range.x == isplit ){
    const unsigned int inlA = nni+isplit;
    dNodeBVH[ini*3+1] = inlA;
    dNodeBVH[inlA*3] = ini;
    dNodeBVH[inlA*3+1] = dSortedId[isplit];
    dNodeBVH[inlA*3+2] = UINT_MAX;
  }
  else{
    const unsigned int iniA = isplit;
    dNodeBVH[ini*3+1] = iniA;
    dNodeBVH[iniA*3] = ini;
  }
  // ----
  if( range.y == isplit+1 ){
    const unsigned int inlB = nni+isplit+1;
    dNodeBVH[ini*3+2] = inlB;
    dNodeBVH[inlB*3] = ini;
    dNodeBVH[inlB*3+1] = dSortedId[isplit+1];
    dNodeBVH[inlB*3+2] = UINT_MAX;
  }
  else{
    const unsigned int iniB = isplit+1;
    dNodeBVH[ini*3+2] = iniB;
    dNodeBVH[iniB*3] = ini;
  }
  if (idx == 0) {
    dNodeBVH[0] = UINT_MAX;
  }
}


}


