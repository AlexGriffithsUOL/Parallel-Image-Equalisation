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

kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//this is just a copy operation, modify to filter out the individual colour channels
	B[id] = A[id];
}

kernel void translateByLookup(global const uchar* A, global uchar* B, global int* correspondingArr) {
	int id = get_global_id(0);
	B[id] = correspondingArr[A[id]];
	barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void createHistogram(global const unsigned int* A, global const unsigned int* B, global int* C) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width * height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y * width; + c * image_size; //global id in 1D space       Remove all but using size

	int key = (int)A[id];
	++C[key];
	barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void busterBrown(global const unsigned int* A, global int* C) {
	int id = get_global_id(0);

	C[id] = A[id];

	if (id > 0) {
		C[id + 1] += C[id];
	}
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