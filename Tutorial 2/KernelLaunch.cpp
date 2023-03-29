/*#include "Utils.h"
#include <vector>
#include "CL/cl2.hpp"

void printKernelInfo(unsigned int WT, unsigned int ET, unsigned int RT, string kernelType) {
	cout << "Displaying " << kernelType.c_str() << " Kernel profiling:" << "\n";
	cout << "	Kernel Writing Time: " << ((float)WT / 1000000) << "ms\n";
	cout << "	Kernel Execution Time: " << ((float)ET / 1000000) << "ms\n";
	cout << "	Kernel Reading Time: " << ((float)RT / 1000000) << "ms\n";
	cout << "	Total Kernel Time: " << ((float)(WT + ET + RT) / 1000000) << "ms\n";
}

void KernelLaunch(string kernelName,
				  vector<cl::Buffer> writeBufferArr,
				  cl::Buffer readingBufferArr,
				  vector<unsigned int> dataArr,
				  vector<unsigned int> sizes,
				  vector<cl::Event> eventArr,
				  string operationName,
				  cl::Program program,
				  cl::CommandQueue queue) {
	
	for (int i = 0; i < writeBufferArr.size(); ++i) {
		queue.enqueueWriteBuffer(writeBufferArr[i], CL_TRUE, 0, sizes[i], &dataArr[i], NULL, &eventArr[0]);
	}
	cl::Kernel kernel = cl::Kernel(program, kernelName.c_str());
	
	for (int i = 0; i < writeBufferArr.size(); ++i) {
		kernel.setArg(i, writeBufferArr[i]);
	}
	kernel.setArg(writeBufferArr.size(), readingBufferArr);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(sizes[0]), cl::NullRange, NULL, &eventArr[1]);

	queue.enqueueReadBuffer(readingBufferArr, CL_TRUE, 0, sizes[sizes.size() - 1], &dataArr[dataArr.size() - 1], NULL, &eventArr[2]);

	/*printKernelInfo(eventArr[0].getProfilingInfo<CL_PROFILING_COMMAND_END>() - eventArr[0].getProfilingInfo<CL_PROFILING_COMMAND_START>(),
		eventArr[1].getProfilingInfo<CL_PROFILING_COMMAND_END>() - eventArr[1].getProfilingInfo<CL_PROFILING_COMMAND_START>(),
		eventArr[2].getProfilingInfo<CL_PROFILING_COMMAND_END>() - eventArr[2].getProfilingInfo<CL_PROFILING_COMMAND_START>(),
		operationName.c_str());*/
//}*/