#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

//#include "AOCLUtils/aocl_utils.h"
#include "CL/opencl.h"

#include "constants.h"

//using namespace aocl_utils;

static const char* kernel_name = "histogram";
static const char* aocx_name = "histogram";

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_kernel kernel;
static cl_program program;
static cl_int status;

static void freeResources() {
    if (kernel)
        clReleaseKernel(kernel);
    if (program)
        clReleaseProgram(program);
    if (queue)
        clReleaseCommandQueue(queue);
    if (context)
        clReleaseContext(context);
}

void cleanup() {
    freeResources();
}

void context_error_callback(const char* errinfo, const void* private_info, size_t cb, void* user_data) {
    printf("Error message in callback: %s\n", errinfo);
}

int main(int argc, char* argv[]) {
    // parse command line args
    unsigned int count = 1000000;
    if (argc > 1) {
        count = atoi(argv[1]);
    }

    int* in_h = (int*)malloc(count * sizeof(int));
    int* bins_h = (int*)malloc(K_NUM_BINS * sizeof(int));
    int* bins_ref_h = (int*)malloc(K_NUM_BINS * sizeof(int));

    for (size_t i = 0; i < count; i++) {
        in_h[i] = rand() % 100;
    }

    for (size_t i = 0; i < K_NUM_BINS; i++) {
        bins_ref_h[i] = 0;
    }

    for (size_t i = 0; i < count; i++) {
        bins_ref_h[in_h[i] % K_NUM_BINS]++;
    }

    cl_uint num_platforms;
    cl_uint num_devices;

    if (!setCwdToExeDir()) {
        return false;
    }

#ifdef FPGA_EMULATOR
    platform = findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
#else 
    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
#endif

    if (platform == NULL) {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        return -1;
    }

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, num_devices);
    checkError(status, "Failed clGetDeviceIDs.");

    context = clCreateContext(0, num_devices, &device, &context_error_callback,
        NULL, &status);
    checkError(status, "Failed clCreateContext.");

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed clCreateCommandQueue.");

    std::string binary_file = getBoardBinaryFile(aocx_name, device);
    program = createProgramFromBinary(context, binary_file.c_str(), &device, num_devices);
    checkError(status, "Failed clCreateProgramWithBinary.");

    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed clBuildProgram.");

    kernel = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed clCreateKernel.");

    cl_mem in_d = clCreateBuffer(context, CL_MEM_READ_WRITE, count * sizeof(int), NULL, &status);
    checkError(status, "clCreateBuffer in_d");
    cl_mem bins_d = clCreateBuffer(context, CL_MEM_READ_WRITE, K_NUM_BINS * sizeof(int), NULL, &status);
    checkError(status, "clCreateBuffer bins_d");

    status = clEnqueueWriteBuffer(queue, in_d, CL_TRUE, 0, count * sizeof(int), in_h, 0, NULL, NULL);
    checkError(status, "clEnqueueWriteBuffer failed for in_d");

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_d);
    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bins_d);
    checkError(status, "Failed to set kernel arg 1");
    status = clSetKernelArg(kernel, 2, sizeof(cl_uint), (void*)&count);
    checkError(status, "Failed to set kernel arg 2");

    status = clEnqueueTask(queue, kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel.");

    checkError(status, "clFinish failed");

    status = clEnqueueReadBuffer(queue, bins_d, CL_TRUE, 0, K_NUM_BINS * sizeof(int), bins_h, 0, NULL, NULL);
    checkError(status, "clEnqueueReadBuffer failed");

    int passed = 1;
    for (int i = 0; i < K_NUM_BINS; i++) {
        if (bins_h[i] != bins_ref_h[i]) {
            passed = 0;
        }
    }

    if (passed) {
        printf("PASSED\n");
    }
    else {
        printf("FAILED\n");
    }

    clReleaseMemObject(in_d);
    clReleaseMemObject(bins_d);

    freeResources();
    free(in_h);
    free(bins_h);
    free(bins_ref_h);

    return passed;
}