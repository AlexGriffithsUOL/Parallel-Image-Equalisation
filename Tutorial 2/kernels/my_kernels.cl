#define K_NUM_BINS 256
//Need to pass another variable called binsize
__kernel void histogramMaker(__global uint* input, __global uint* bins, uint count, local uint* bins_local, uint numBins) {
	for (uint i = 0; i < numBins; i++) { 
		bins_local[i] = 0;
	}
	
	
	for (uint i = 0; i < count; i++) {
		bins_local[input[i] % numBins]++;  //K num bins swapped for binnumber
	}
	

	for (uint i = 0; i < numBins; i++) { //Knumbins swap for bin number
		bins[i] = bins_local[i]; 
	}
}

//a very simple histogram implementation
kernel void hist_simple(global const unsigned int* A, global unsigned int* H, local uint* temp, unsigned int binSize) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	local uint scratch_1;
	temp[id] = A[id];
	scratch_1 = temp[id];
	unsigned int bin_index = scratch_1/binSize;
	atomic_inc(&H[bin_index]);
}



kernel void translateByLookup(global uint* A, global const uint* correspondingArr, const uint binSize) {
	int id = get_global_id(0);
	float binNumber = (float)(A[id]) / binSize;
	A[id] = correspondingArr[(unsigned int)binNumber];
}


kernel void normaliseHistogram(__global const unsigned int* A, __global float* B, unsigned int imgSize) {
	int id = get_global_id(0);
	B[id] = ((float)A[id] / imgSize);
}

kernel void normaliseHistogramFaster(__global float* A, local float* B, unsigned int imgSize) {
	int id = get_global_id(0);
	B[id] = (float)A[id];
	B[id] = (B[id] / imgSize);
	A[id] = B[id];
}

kernel void scaleTo255(__global float* A, __global unsigned int* B) {
	int id = get_global_id(0);
	B[id] = (unsigned int)(A[id] * 255);
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
kernel void scan_add(global const int* A, global uint* B,local uint* scratch_1, local uint* scratch_2) {
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


//Blelloch basic exclusive scan
kernel void scan_bl(global unsigned int* A) {
	unsigned int id = get_global_id(0);
	unsigned int N = get_global_size(0);
	unsigned int t;

	//up-sweep
	for (unsigned int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N - 1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (unsigned int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;      //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}