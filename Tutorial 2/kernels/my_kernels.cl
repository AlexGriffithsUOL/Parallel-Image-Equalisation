//#include "Constants.h"
#define K_NUM_BINS 256

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



kernel void translateByLookup(global const unsigned int* A, global unsigned int* B, global const unsigned int* correspondingArr) {
	int id = get_global_id(0);
	B[id] = correspondingArr[(unsigned int)(A[id])];
}


kernel void normaliseHistogram(__global const unsigned int* A, __global float* B, unsigned int imgSize) {
	int id = get_global_id(0);
	B[id] = ((float)A[id] / imgSize);
}

kernel void scaleTo255(__global float* A, __global unsigned int* B) {
	int id = get_global_id(0);
	B[id] = (unsigned int)(A[id] * 255);
}



kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0); int t;
	// Up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0)
			A[id] += A[id - stride];
		barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	}
	// Down-sweep
	if (id == 0) A[N - 1] = 0; // Exclusive scan
	barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	for (int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; // Reduce
			A[id - stride] = t; // Move
		}
		barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	}
}


kernel void scan_hs(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;
	for (int stride = 1; stride < N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];
		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
		C = A; A = B; B = C; // swap A & B between steps
	}
}

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
kernel void scan_add(__global const int* A, global uint* B,local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int* scratch_3;//used for buffer swap
	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *=2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}