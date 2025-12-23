
namespace mortons {

__device__
uint32_t device_expand_bits2(uint32_t x)
{
    x = (x | (x << 8)) & 0x00ff00ff;
    x = (x | (x << 4)) & 0x0f0f0f0f;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

__device__
uint32_t device_expand_bits3(uint32_t v)
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__
uint32_t device_morton_code2(float x, float y)
{
  auto ix = (uint32_t)fmin(fmax(x * 65536.0f, 0.0f), 65535.0f);
  auto iy = (uint32_t)fmin(fmax(y * 65536.0f, 0.0f), 65535.0f);
  //  std::cout << std::bitset<10>(ix) << " " << std::bitset<10>(iy) << " " << std::bitset<10>(iz) << std::endl;
  ix = device_expand_bits2(ix);
  iy = device_expand_bits2(iy);
  //  std::cout << std::bitset<30>(ix) << " " << std::bitset<30>(iy) << " " << std::bitset<30>(iz) << std::endl;
  return ix * 2 + iy;
}

__device__
uint32_t device_morton_code3(float x, float y, float z)
{
  auto ix = (uint32_t)fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
  auto iy = (uint32_t)fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
  auto iz = (uint32_t)fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
  //  std::cout << std::bitset<10>(ix) << " " << std::bitset<10>(iy) << " " << std::bitset<10>(iz) << std::endl;
  ix = device_expand_bits3(ix);
  iy = device_expand_bits3(iy);
  iz = device_expand_bits3(iz);
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

}