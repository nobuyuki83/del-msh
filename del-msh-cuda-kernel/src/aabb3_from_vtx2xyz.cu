
__device__
void device_AtomicMaxFloat(float * const address, const float value)
{
  if ( *address >= value ) { return; }
  int * const address_as_i = (int *)address;
  int old = * address_as_i, assumed;
  do {
    assumed = old;
    if (__int_as_float(assumed) >= value) { break; }
    old = atomicCAS(address_as_i, assumed, __float_as_int(value));
  } while (assumed != old);
}

__device__
void device_AtomicMinFloat(float * const address, const float value)
{
  if ( *address <= value ) { return; }
  int * const address_as_i = (int *)address;
  int old = * address_as_i, assumed;
  do {
    assumed = old;
    if (__int_as_float(assumed) <= value) { break; }
    old = atomicCAS(address_as_i, assumed, __float_as_int(value));
  } while (assumed != old);
}


extern "C" {

__global__
void kernel_MinMax_TPB256(
    float *d_minmax,
    const float *d_XYZ,
    unsigned int np)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int s_idx = threadIdx.x;
  const unsigned int BLOCK = 256;
  assert(blockDim.x == BLOCK);
  assert( blockDim.y == 3 && blockIdx.y == 0 );
  unsigned int idy = threadIdx.y;
  if( idx >= np ){ return; }
  // ---------------
  // put value in shared memory
  __shared__ float s_XYZ[BLOCK][3];
  s_XYZ[s_idx][idy] = d_XYZ[idx*3+idy];
  __syncthreads();
  if( s_idx == 0 ) {
    // computing min/max inside shared memory
    float vmin = s_XYZ[0][idy];
    float vmax = s_XYZ[0][idy];
    int ns = BLOCK;
    if( blockDim.x * (blockIdx.x+1) > np ) {
      ns = np - blockDim.x * blockIdx.x;
    }
    for(int is=0;is<ns;++is){
      if( s_XYZ[is][idy] < vmin ){ vmin = s_XYZ[is][idy]; }
      if( s_XYZ[is][idy] > vmax ){ vmax = s_XYZ[is][idy]; }
    }
    // computing mim/max across blocks
    device_AtomicMinFloat(d_minmax+idy+0,vmin);
    device_AtomicMaxFloat(d_minmax+idy+3,vmax);
  }
}

}