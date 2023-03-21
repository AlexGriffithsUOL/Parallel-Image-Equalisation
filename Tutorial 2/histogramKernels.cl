#include "Constants.h"

__kernel void histogramMaker(__global int* restrict in,
    __global int* restrict bins,
    uint count) {
    // store a local copy of the histogram to avoid read-accumulate-writes
    // to global memory
    __attribute__((register)) int bins_local[K_NUM_BINS];

    // initialize the local bins
#pragma unroll
    for (uint i = 0; i < K_NUM_BINS; i++) {
        bins_local[i] = 0;
    }

    // compute the histogram
#pragma ii 1
    for (uint i = 0; i < count; i++) {
        bins_local[in[i] % K_NUM_BINS]++;
    }

    // write back the local copy to global memory
#pragma unroll
    for (uint i = 0; i < K_NUM_BINS; i++) {
        bins[i] = bins_local[i];
    }
}