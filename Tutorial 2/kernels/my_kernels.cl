#define K_NUM_BINS 256
//Need to pass another variable called binsize
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
		bins_local[in[i] % K_NUM_BINS]++;  //K num bins swapped for binnumber
	}

	// write back the local copy to global memory
#pragma unroll
	for (uint i = 0; i < K_NUM_BINS; i++) { //Knumbins swap for bin number
		bins[i] = bins_local[i]; 
	}
}

//a very simple histogram implementation
kernel void hist_simple(global const unsigned int* A, global unsigned int* H, unsigned int binSize) {
	int id = get_global_id(0);
	unsigned int bin_index = A[id]/binSize;
	atomic_inc(&H[bin_index]);
}



kernel void translateByLookup(global const unsigned int* A, global unsigned int* B, global const unsigned int* correspondingArr, int binSize) {
	int id = get_global_id(0);
	float binNumber = (float)(A[id]) / binSize;
	B[id] = correspondingArr[(unsigned int)binNumber];
}


kernel void normaliseHistogram(__global const unsigned int* A, __global float* B, unsigned int imgSize) {
	int id = get_global_id(0);
	B[id] = ((float)A[id] / imgSize);
}

kernel void scaleTo255(__global float* A, __global unsigned int* B) {
	int id = get_global_id(0);
	B[id] = (unsigned int)(A[id] * 255);
}



kernel void scan_bl(global uint* A) {
	int id = get_global_id(0);
	int N = get_global_size(0); 
	int t;
	// Up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0)
			A[id] += A[id - stride];
		//barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	}
	// Down-sweep
	if (id == 0) A[N - 1] = 0; // Exclusive scan
	//barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	for (int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			t = A[id];
			//A[id] += A[id - stride]; // Reduce
			printf("%d\n", A[id]);
			A[id - stride] = t; // Move
		}
		//barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	}
}

kernel void scan_parallel_naive_SAT(global const uint* input, global uint* output, const uint n)
{
	__global uint* temp = output;
	int pout = 1, pin = 0;

	int thread = get_local_id(0);
	temp[pout * n + thread] = (thread > 0) ? input[thread - 1] : 0;

	barrier(CLK_LOCAL_MEM_FENCE);
	for (int offset = 1; offset < n; offset *= 2)
	{
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;

		barrier(CLK_LOCAL_MEM_FENCE);
		temp[pout * n + thread] = temp[pin * n + thread];
		if (thread >= offset)
			temp[pout * n + thread] += temp[pin * n + thread - offset];
	}

	output[thread] = temp[pout * n + thread]; // write output
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






















/*
__kernel void histogram(__global const uchar* data,
	__global uint* histogram,
	const uint data_size, const uint group_size)
{
	// Allocate shared memory for the local histogram
	__local uint local_histogram[256];
	for (uint i = 0; i < 256; i++) {
		local_histogram[i] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Each work-group will process a subset of the data
	const uint group_id = get_group_id(0);
	const uint local_id = get_local_id(0);
	const uint global_id = get_global_id(0);

	// Compute the range of data elements to be processed by this work-group
	const uint start_index = group_id * group_size;
	const uint end_index = start_index + group_size;

	// Add padding to the input data array to handle boundary cases
	__local uchar local_data[group_size + group_size - 1];
	for (uint i = local_id; i < group_size + group_size - 1; i += group_size) {
		if (start_index + i - group_size / 2 < data_size) {
			local_data[i] = data[start_index + i - group_size / 2];
		}
		else {
			local_data[i] = 0;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Process the data elements in this range and update the local histogram
	for (uint i = local_id + group_size / 2; i < group_size + group_size / 2 && start_index + i - group_size / 2 < data_size; i += group_size) {
		const uchar bin = local_data[i];
		local_histogram[bin]++;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Combine the local histograms from all work-groups and write the final histogram
	if (local_id == 0) {
		for (uint i = 0; i < 256; i++) {
			uint bin_count = local_histogram[i];
			for (uint j = 1; j < get_num_groups(0); j++) {
				bin_count += local_histogram[i + j * group_size];
			}
			histogram[i] = bin_count;
		}
	}
}*/