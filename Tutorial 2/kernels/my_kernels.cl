//Histogram creator (adapted code from workshops)
kernel void hist_simple(global const unsigned int* A, global unsigned int* H, local uint* temp, unsigned int binSize) { //Takes in an input vector (the image) A, a buffer to read (H), a local temp buffer and the bin size
	int id = get_global_id(0); //Gets the id
	int lid = get_local_id(0); //Gets the thread id
	local uint scratch_1; //Local buffer used to create a faster calculation time
	temp[lid] = A[id]; //Writes global data to local
	scratch_1 = temp[lid]; //Translates to a local unsigned integer 
	unsigned int bin_index = scratch_1/binSize; //Calculates the bin
	atomic_inc(&H[bin_index]); //Atomic incrementations (not as efficient yet somehow the fastest so far)
}

//Translation via lookup table
kernel void translateByLookup(global uint* A, global const uint* correspondingArr, const uint binSize) { //Takes in an input vector (A), a lookup table (correspondingArr) and a bin size
	int id = get_global_id(0); //Gets the id
	float binNumber = (float)(A[id]) / binSize; //Recasts the pixel to a float to divide
	A[id] = correspondingArr[(unsigned int)binNumber]; //Casts back to integer to round down so that it can be written to the global vector
}

//Normalisation
kernel void normaliseHistogram(__global const unsigned int* A, __global float* B, unsigned int imgSize) { //Take an input vector (A), a readable buffer (B), and the image size
	int id = get_global_id(0); //Gets the id
	B[id] = ((float)A[id] / imgSize); //Recasts to float to 
}

//Scaling
kernel void scaleTo255(__global float* A, __global unsigned int* B) { //Takes an input vector (A) and a reading vector (B)
	int id = get_global_id(0); //Gets id
	B[id] = (unsigned int)(A[id] * 255); //Recasts A after scaling to round it down and to handle 16-bit
}

//Hillis-Steele scan (double buffered)
kernel void scan_add(global const int* A, global uint* B, local uint* scratch_1, local uint* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int* scratch_3;//used for buffer swap
	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
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
	unsigned int id = get_global_id(0); //Gets the id
	unsigned int N = get_global_size(0); //Gets the size
	unsigned int t; //Temporary variable created

	//up-sweep
	for (unsigned int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0)
			A[id] += A[id - stride]; //Goes through the data in strides

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N - 1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (unsigned int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			t = A[id]; //Reads A[id] into t
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;      //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

//Development code
kernel void normaliseHistogramFaster(__global float* A, local float* B, unsigned int imgSize) { //Takes an input vector (A), a local float vector (B), and an image size
	int id = get_global_id(0); //Gets the id
	B[id] = (float)A[id]; //Recasts A to float to allow for faster normalisation 
	B[id] = (B[id] / imgSize); //Divides to normalise 
	A[id] = B[id]; //Writes back to the global histogram
}

//Naive SAT scan (Code found: )
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

//Hillis-Steele scan
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

//Code found: 
//Intel provided code adapted for variable bin sizes.
__kernel void histogramMaker(__global uint* input, __global uint* bins, uint count, local uint* bins_local, uint numBins) { //Takes input, bins, data size count, local bins for faster operation, number of bins
	for (uint i = 0; i < numBins; i++) { //Initialises local bins to 0
		bins_local[i] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE); //Synchronisation

	for (uint i = 0; i < count; i++) { //Increments through data
		bins_local[input[i] % numBins]++;  //Put data in appropriate bins
	}
	barrier(CLK_LOCAL_MEM_FENCE); //Syncs the kernels again

	for (uint i = 0; i < numBins; i++) {
		bins[i] = bins_local[i]; //Writes local data back to the global bins
	}
}