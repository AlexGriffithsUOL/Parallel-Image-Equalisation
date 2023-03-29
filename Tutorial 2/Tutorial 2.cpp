#include "MappingOperationsSerial.h"
#include "ReadingWriting.h"
#include "UserInputHandling.h"

#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"
#include "CL/cl2.hpp"


using namespace cimg_library;

void printKernelInfo(unsigned int WT, unsigned int ET, unsigned int RT, string kernelType) { //Takes in the parameters of the times from the kernel profiling and the kernelType to print it
	cout << "\nDisplaying " << kernelType.c_str() << " Kernel profiling:" << "\n";
	cout << "	Kernel Writing Time: " << ((float)WT / 1000000) << "ms\n"; //Converts the nanoseconds into milliseconds
	cout << "	Kernel Execution Time: " << ((float)ET/ 1000000) << "ms\n";
	cout << "	Kernel Reading Time: " << ((float)RT / 1000000)<< "ms\n";
	cout << "	Total Kernel Time: " << ((float)(WT + ET + RT) / 1000000) << "ms\n";
}

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "-p| select platform\n";
	std::cerr << "-d| select device\n";
	std::cerr << "-l| list all platforms and devices\n";
	std::cerr << "-f| input image file (default: test.pgm)\n";
	std::cerr << "-h| print this message\n";
}

int main(int argc, char** argv) {
	int platformID = 3; //Default platform for my computer, can be changed with arguments below

	int deviceID = 0; //Default device for my computer, can be changed with arguments below

	string imageFilename = "test_large.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platformID = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { deviceID = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << "\n"; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { imageFilename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		//Read image in from the filename, can be defined by user above
		CImg<unsigned int> inputImage(imageFilename.c_str());

		//Colour images are converted to grey
		inputImage._spectrum = 1;

		//Create the context
		cl::Context context = GetContext(platformID, deviceID);		

		//Print out the platform and device
		std::cout << "Running on " << GetPlatformName(platformID) << ", " << GetDeviceName(platformID, deviceID) << std::endl;

		//Create the command queue
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE); //Profiling enabled to allow for kernel profiling

		//Create the kernel sources
		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");

		//Create the kernel program object
		cl::Program program(context, sources);

		//Create events for profiling
		cl::Event timingW; //Writing to buffers
		cl::Event timingE; //Executing kernels
		cl::Event timingR; //Reading from buffers


		//Build and debug kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << "\n";
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << "\n";
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << "\n";
			throw err;
		}

		/*----Start of program----*/
		//Printing the image size
		cout << "Total size image size: " << inputImage.size() << "\n";
		//Ask for user input
		cout << "Select a bin size from below:" << "\n";

		//Gets the highest value in the image
		int value = inputImage.max();

		//Gets the maximum number of bins available
		int maximumPO2 = ceil(log2(value));
		cout << "Max val: " << maximumPO2 << "\n";

		//For loop finds the next highest power of 2 that can be used, allows for 8, 16, etc bit iamges
		for (int i = 0; i < (maximumPO2+ 1); ++i) { //1 over the max power of 2 to write the correct number of options
			cout << i << "| " << (pow(2, i)) << "\n";  //Prints the powers of 2 that can be used
		}
		
		//Get the total size of the input image
		unsigned int totalSize = inputImage.size();



		/*----Get user input----*/
		//Unsigned int used to store input
		int userinput = checkInputType(0, maximumPO2);
		//Bin number set to the power of 2 of the user input (for example if option 2, the set to 2^2 (4))
		int binNumber = pow(2, userinput);
		//BinSize is equivalent to the width of the bin to be used as a divisor
		unsigned int binSize = (pow(2, ceil(log2(value))) / (binNumber));
		cout << "Number of bins: " << binNumber << "\n";
		cout << "Bin width: " << binSize << "\n";
		/*----End----*/



		//Display The Initial Picture
		cout << "\nDisplaying original image...\n\n";
		inputImage.display("Original Image", true, 0, true);

		/*----Create initial image histogram----*/
		//Create vectors to store data
		vector<unsigned int> histogram(binNumber); //Bin number used to setup the appropriate histogram size

		//Create buffers
		cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY, inputImage.size() * sizeof(unsigned int)); //unsigned int used to handle higher bit images
		cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, (histogram.size()) * sizeof(unsigned int));
		
		//Writing to buffer
		queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, inputImage.size() * sizeof(unsigned int), &inputImage.data()[0], NULL, &timingW); //Writes the image into a buffer, calls writing profiler event

		//Setting up program for kernels
		cl::Kernel kernel = cl::Kernel(program, "hist_simple"); //Sets the program name to the correct one

		//Setting up kernel arguments
		kernel.setArg(0, inputImageBuffer); //Buffer is set as an argument to be read into the kernel
		kernel.setArg(1, histogramBuffer); //Buffer is set as an argument to be read back from the kernel calculations
		kernel.setArg(2, cl::Local(histogram.size() * sizeof(unsigned int) / binNumber)); //Bin number is used as a divisor to prevent 16 bit images from using too many resource
		kernel.setArg(3, binSize); //Used as a divisor to keep the bins continous - note it is not the number but the size, i.e. width of the bin

		//Launch correct number of Kernels
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange, NULL, &timingE); //Launches the kernels, using the correct sizing, calls execution profiler event

		//Reading the buffer return
		queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(unsigned int), &histogram.data()[0], NULL, &timingR); //Pointer is used to point to the start of the histogram vector to read data back into
																																			 //Call reading profiler event
		//Display profiling times
		cout << "Histogram created" << "\n"; //Writes to show the the historam has been created
		printKernelInfo(timingW.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingW.getProfilingInfo<CL_PROFILING_COMMAND_START>(), //Writing timing event
						timingE.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingE.getProfilingInfo<CL_PROFILING_COMMAND_START>(), //Execution timing event
						timingR.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingR.getProfilingInfo<CL_PROFILING_COMMAND_START>(), //Reading timing event
						"Histogram");

		//Show Graph of the image
		CImg<unsigned int> histogr(histogram.data(), binNumber); //Image of the data created
		histogr.display_graph("Histogram", 3, 1, "Bin number", 0, 0, "Number of pixels", 0, 0, true); //Histogram graph displayed
		/*----End----*/



		//Display the options
		cout << "Select a scan: " << "\n";
		cout << "0| Blelloch" << "\n";
		cout << "1| Hillis-Steele" << "\n";

		//Get user input for the selection of a seperate scan
		bool input = checkInputType(0, 1); //Function that returns the input after performing error handling
		


		/*----Create a cumulative histogram*/
		//Part 2: Cumulative histogram
		//Create vectors to store data
		vector<unsigned int>cumulHist(histogram.size());

		//Create buffers
		cl::Buffer cumulativeHistogramBuffer(context, CL_MEM_READ_WRITE, histogram.size() * sizeof(unsigned int)); //Based off of histogram size for consistancy with bins

		//Create the correct kernel depending on the selection
		if (input) //If statement to swap between user input
		{
			cout << "Hillis-Steele selected.\n";

			//Setup kernel with appropriate program
			kernel = cl::Kernel(program, "scan_add"); //Hillis-Steel program

			//Setting up kernel arguments
			kernel.setArg(0, histogramBuffer); //Histogram added to be altered
			kernel.setArg(1, cumulativeHistogramBuffer); //Buffer created to read from
			kernel.setArg(2, cl::Local((sizeof(unsigned int)))); //Unsigned int used as it allows for 16-bit by not overloading the memory of the device
			kernel.setArg(3, cl::Local((sizeof(unsigned int)))); //Empty local buffers made for privatisation
		}
		else
		{
			
			cout << "Blelloch selected.\n";
			
			//Setup kernel with appropriate program
			kernel = cl::Kernel(program, "scan_bl"); //Blelloch scan program

			//Swap buffers
			cumulativeHistogramBuffer = histogramBuffer; //Buffers changed to allow for program to alter historam data instead of empty buffer 

			//Setting up kernel arguments
			kernel.setArg(0, cumulativeHistogramBuffer); //Only requires one buffer
		}

		//Launch correct number of Kernels
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange, NULL, &timingE); //Launches the kernels to do the calculations

		//Reading the data back from the buffer
		queue.enqueueReadBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(unsigned int), &cumulHist.data()[0], NULL, &timingR); //Reads the buffer back into the vector

		//Display profiling times
		cout << "Histogram cumulated" << "\n";
		printKernelInfo(timingW.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingW.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			timingE.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingE.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			timingR.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingR.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			"Cumulative Histogram");

		//Display cumulative histogram
		CImg<unsigned int> cumulativeHistogramImage(cumulHist.data(), binNumber);
		cumulativeHistogramImage.display_graph("Cumulative Histogram", 3, 1, "Bin number", 0, 0, "Number of pixels", 0, 0, true);
		/*----End----*/



		/*----Create a Normalised Historam----*/
		//Create vectors to store data
		vector<float>normHist(cumulHist.size());

		//Create buffers
		cl::Buffer normalisedHistogramBuffer(context, CL_MEM_READ_WRITE, histogram.size() * sizeof(float)); //Based off of histogram size for consistancy with bins

		//Writing to buffer
		queue.enqueueWriteBuffer(normalisedHistogramBuffer, CL_TRUE, 0, normHist.size() * sizeof(float), &normHist.data()[0], NULL, &timingW);

		//Setup kernel with appropriate program
		kernel = cl::Kernel(program, "normaliseHistogram");

		//Setting up kernel arguments
		kernel.setArg(0, cumulativeHistogramBuffer);
		kernel.setArg(1, normalisedHistogramBuffer);  //Used to read back the final image
		kernel.setArg(2, (int)(inputImage.size())); //Sets up the argument corresponding to the kernel function in my_kernels.cl

		//Launch correct number of Kernels
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(cumulHist.size()), cl::NullRange, NULL, &timingE); //Gets the range in the devices for the kernels

		//Reading the data back from the buffer
		queue.enqueueReadBuffer(normalisedHistogramBuffer, CL_TRUE, 0, cumulHist.size()*sizeof(float), &normHist.data()[0], NULL, &timingR); //Reads the output buffer back from the device

		//Display profiling times
		cout << "Histogram normalised" << "\n";
		printKernelInfo(timingW.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingW.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			timingE.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingE.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			timingR.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingR.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			"Normalised Histogram");

		//Display normalised histogram
		CImg<float> normhistogr(normHist.data(), binNumber);
		normhistogr.display_graph("Normalised Histogram", 3, 1, "Bin number", 0, 0, "Number of pixels", 0, 0, true);
		/*----End----*/
		


		/*----Create a scaled histogram----*/
		//Create vectors to store data
		vector<unsigned int> scaleHist(normHist.size());  //Based on previous histogram size for continuity

		//Create buffers
		cl::Buffer scaleHistogramBuffer(context, CL_MEM_READ_WRITE, normHist.size() * sizeof(unsigned int));

		//Setup kernel with appropriate program
		kernel = cl::Kernel(program, "scaleTo255"); //Sets the kernel program to scale the histogram

		//Setting up kernel arguments
		kernel.setArg(0, normalisedHistogramBuffer);  //Used to alter the data
		kernel.setArg(1, scaleHistogramBuffer);  //Used to read back the scaled data

		//Launch correct number of Kernels
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(normHist.size()), cl::NullRange, NULL, &timingE); //Launches the correct number of kernels for the calculations

		//Reading the data back from the buffer
		queue.enqueueReadBuffer(scaleHistogramBuffer, CL_TRUE, 0, scaleHist.size()*sizeof(unsigned int), &scaleHist.data()[0], NULL, &timingR);  //Reads to the starting pointer of the scaled histogram

		//Display profiling times
		cout << "Histogram scaled" << "\n";
		printKernelInfo(timingW.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingW.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			timingE.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingE.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			timingR.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingR.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			"Scaled Histogram");

		//Display scaled histogram
		CImg<unsigned int> scalehistogr(scaleHist.data(), binNumber);
		scalehistogr.display_graph("Scaled Histogram", 3, 1, "Bin number", 0, 0, "Number of pixels", 0, 0, true);
		/*----End----*/
		
		

		/*----Contrast  via lookup table----*/
		//Create vectors to store data
		vector<unsigned int> outputImageVector(inputImage.size()); //Vector created to store image

		//Setup kernel with appropriate program
		kernel = cl::Kernel(program, "translateByLookup"); //Sets the kernel program to translate it from a lookup table

		//Setting up kernel arguments
		kernel.setArg(0, inputImageBuffer);  //Used to pass in the values to convert
		kernel.setArg(1, scaleHistogramBuffer);  //Used as a look up table
		kernel.setArg(2, binSize);  //Used as a divisor to put in appropriate bins

		//Launch correct number of Kernels
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange, NULL, &timingE); //Launches the correct range of kernels to process image

		//Reading the data back from the buffer
		queue.enqueueReadBuffer(inputImageBuffer, CL_TRUE, 0, inputImage.size() * sizeof(unsigned int), &outputImageVector.data()[0], NULL, &timingR); //Reads buffer starting from the pointer at the vector data start
		queue.flush(); //Clear queue, not necessary as it is handled by the garbage collector in the library however done to be sure of no memory leakage

		//Display profiling times
		cout << "Image contrasted" << "\n";
		printKernelInfo(timingW.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingW.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			timingE.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingE.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			timingR.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingR.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			"Contrasting Via Lookup Table");

		//Display final parallel result
		CImg<unsigned int> outputImage(outputImageVector.data(), inputImage.width(), inputImage.height(), inputImage.depth(), 1); //Image created from the data
		outputImage.display("Parallel Kernel Output", false, 0, true); //Displays final parallel image
		/*----End----*/



		//Equalisation in serial for comparison
		CImg<unsigned char> serialImage = returnRGBMap(inputImage); //Image created, serial operations called. Will be slow
		serialImage.display("Serial Output For Comparison (Maximum Bins Only)", false, 0, true); //Displays the image

		//Programs end
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << "\n"; //Catches OpenCL errors
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << "\n"; //Catches CImg errors
	}
	return 0;
}