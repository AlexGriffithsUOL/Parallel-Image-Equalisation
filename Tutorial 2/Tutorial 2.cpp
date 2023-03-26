#include "LibraryImport.h"
#include "MappingOperationsSerial.h"
#include "Printing.h"
#include <cstdint>
//#include "Constants.h"
#define K_NUM_BINS 256

//#define MAX_SOURCE_SIZE (0x100000)

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "-p| select platform\n";
	std::cerr << "-d| select device\n";
	std::cerr << "-l| list all platforms and devices\n";
	std::cerr << "-f| input image file (default: test.pgm)\n";
	std::cerr << "-h| print this message\n";
}

int main(int argc, char** argv) {
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test_large.ppm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned int> inputImage(image_filename.c_str());
		inputImage._spectrum = 1;

		cl::Context context = GetContext(platform_id, device_id);		
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");
		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		/*----Start of program----*/
		//Printing the image size
		cout << "Total size image size: " << inputImage.size() << "\n";
		//Ask for user input
		cout << "Select a bin size from below:" << "\n";

		//Gets the highest value in the image
		int value = inputImage.max();

		//For loop finds the next highest power of 2 that can be used, allows for 8, 16, etc bit iamges
		for (int i = 0; i < (ceil(log2(value))+ 1); ++i) {
			cout << i << "| " << (pow(2, i)) << "\n";  //Prints the powers of 2 that can be used
		}

		unsigned int totalSize = inputImage.size();
		/*----Get user input----*/
		//Unsigned int used to store input
		unsigned int userinput;
		//Cin called to request input
		cin >> userinput;
		//Bin number set to the power of 2 of the user input (for example if option 2, the set to 2^2 (4))
		int binNumber = pow(2, userinput);
		//BinSize is equivalent to the width of the bin to be used as a divisor
		unsigned int binSize = pow(2, ceil(log2(value))) / (binNumber);


		/*----Create initial image histogram----*/
		//Create buffers for image and histogram
		cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY, inputImage.size() * sizeof(unsigned int)); //unsigned int used to handle higher bit images
		cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, (binNumber) * sizeof(unsigned int));
		
		//Writing to buffer
		queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, inputImage.size() * sizeof(unsigned int), &inputImage.data()[0]); //Writes the image into a buffer

		//Setting up program for kernels
		cl::Kernel kernel = cl::Kernel(program, "hist_simple"); 

		//Setting up kernel arguments
		kernel.setArg(0, inputImageBuffer);
		kernel.setArg(1, histogramBuffer);
		kernel.setArg(2, binSize); //Used as a divisor to keep the bins continous - note it is not the number but the size, i.e. width of the bin

		cl::Event prof_event;
		//Launching kernels
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange, NULL, &prof_event);
		//write time
		//Execution time
		//transfer time
		//total time

		//Reading the buffer return
		vector<unsigned int> histogram(binNumber); //Bin number used to 
		queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(unsigned int), &histogram.data()[0]); //Pointer is used to point to the start of the histogram vector to read data back into
		/*----End----*/
		cout << "   - Kernel time taken: " << (prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / inputImage.size() << "ms\n";

		
		/*----Create a cumulative histogram*/
		//Buffer Creatopm
		cl::Buffer cumulativeHistogramBuffer(context, CL_MEM_READ_WRITE, histogram.size() * sizeof(unsigned int)); //Based off of histogram size for consistancy with bins

		//Write the buffer to the command queue
		queue.enqueueWriteBuffer(histogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(unsigned int), &histogram.data()[0]);

		//Setup kernel
		kernel = cl::Kernel(program, "scan_add"); //Program changed

		//Set kernel arguments
		kernel.setArg(0, histogramBuffer);
		kernel.setArg(1, cumulativeHistogramBuffer);
		kernel.setArg(2, cl::Local(histogram.size() * sizeof(unsigned int)));
		kernel.setArg(3, cl::Local(histogram.size() * sizeof(unsigned int))); //Empty local buffers made for privatisation
		
		//Launch kernels and execute profiling
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange, NULL, &prof_event);

		//Reading the data back from the buffer
		vector<int>cumHist(binNumber);
		queue.enqueueReadBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(unsigned int), &cumHist.data()[0]);

		//Displaying the kernel execution time
		cout << "   - Kernel time taken: " << (prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / histogram.size() << "ms\n";
		/*----End----*/

		/*----Create a Normalised Historam----*/
		cl::Buffer normalisedHistogramBuffer(context, CL_MEM_READ_WRITE, cumHist.size() * sizeof(float));

		queue.enqueueWriteBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, cumHist.size() * sizeof(unsigned int), &cumHist.data()[0]);

		kernel = cl::Kernel(program, "normaliseHistogram");

		kernel.setArg(0, cumulativeHistogramBuffer);
		kernel.setArg(1, normalisedHistogramBuffer);
		kernel.setArg(2, (int)(inputImage.size())); //Sets up the argument corresponding to the kernel function in my_kernels.cl

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(cumHist.size()), cl::NullRange, NULL, &prof_event); //Gets the range in the devices for the kernels

		vector<float>normHist(cumHist.size());
		queue.enqueueReadBuffer(normalisedHistogramBuffer, CL_TRUE, 0, cumHist.size()*sizeof(float), &normHist.data()[0]); //Reads the output buffer back from the device
		cout << "   - Kernel time taken: " << (prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / histogram.size() << "ms\n";
		/*----End----*/
		
		/*----Create a scaled histogram----*/
		//Create a buffer to hold the scaled histogram, based off of the size of the previous to keep the sizes continous with the bins
		cl::Buffer scaleHistogramBuffer(context, CL_MEM_READ_WRITE, normHist.size() * sizeof(unsigned int));

		//Write the normalised histogram buffer to be used to read in data
		queue.enqueueWriteBuffer(normalisedHistogramBuffer, CL_TRUE, 0, normHist.size() * sizeof(float), &normHist.data()[0]);

		//Set up the kernel with the correct program
		kernel = cl::Kernel(program, "scaleTo255");

		//Set arguments for the kernels
		kernel.setArg(0, normalisedHistogramBuffer);  //Used to altering the data
		kernel.setArg(1, scaleHistogramBuffer);  //Used to read back the scaled data

		//Launch the kernels
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(normHist.size()), cl::NullRange, NULL, &prof_event);

		//Read back the data
		vector<unsigned int> scaleHist(normHist.size());  //Based on previous histogram size for continuity
		queue.enqueueReadBuffer(scaleHistogramBuffer, CL_TRUE, 0, scaleHist.size()*sizeof(unsigned int), &scaleHist.data()[0]);  //Reads to the starting pointer of the scaled histogram
		cout << "   - Kernel time taken: " << (prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / histogram.size() << "ms\n";
		/*----End----*/
		

		/*----Contrast  via lookup table----*/
		//Create a buffer to hold the input image to read data
		cl::Buffer outputImageBuffer(context, CL_MEM_READ_WRITE, inputImage.size() * sizeof(unsigned int));
		
		//Write the buffer, using the correct size, sizing is based off of previous size of vector
		queue.enqueueWriteBuffer(scaleHistogramBuffer, CL_TRUE, 0, scaleHist.size() * sizeof(unsigned int), &scaleHist.data()[0]);

		//Set kernel program
		kernel = cl::Kernel(program, "translateByLookup");

		//Set kernel arguments
		kernel.setArg(0, inputImageBuffer);  //Used to pass in the values to convert
		kernel.setArg(1, outputImageBuffer);  //Used to read back the final image
		kernel.setArg(2, scaleHistogramBuffer);  //Used as a look up table
		kernel.setArg(3, binSize);  //Used as a divisor to put in appropriate bins

		//Launch Kernels
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange);

		//Store Data
		vector<unsigned int> outputImageVector(inputImage.size()); //Vector created to store image
		queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, inputImage.size() * sizeof(unsigned int), &outputImageVector.data()[0]); //Reads buffer starting from the pointer at the vector data start
		queue.flush(); //Clear queue, not necessary as it is handled by the library however done to be sure of no memory leakage
		/*----End----*/




		/*----Displaying images and graphs----*/
		//Display The Initial Picture
		inputImage.display("Original Image", true, 0, true);;


		CImg<unsigned int> histogr(histogram.data(), binNumber);
		histogr.display_graph("Histogram", 3, 1, "Bin number", 0, 0, "Number of pixels", 0, 0, true);
		CImg<unsigned int> cuumhistogr(cumHist.data(), binNumber);
		cuumhistogr.display_graph("Cumulative Histogram", 3, 1, "Bin number", 0, 0, "Number of pixels", 0, 0, true);
		CImg<float> normhistogr(normHist.data(), binNumber);
		normhistogr.display_graph("Normalised Histogram", 3, 1, "Bin number", 0, 0, "Number of pixels", 0, 0, true);
		CImg<unsigned int> scalehistogr(scaleHist.data(), binNumber);
		scalehistogr.display_graph("Scaled Histogram", 3, 1, "Bin number", 0, 0, "Number of pixels", 0, 0, true);


		//Display Final Parallel Result
		CImg<unsigned char> outputImage(outputImageVector.data(), inputImage.width(), inputImage.height(), inputImage.depth(), 1);
		outputImage.display("Parallel Kernel Output", true, 0, true);

		//Equalisation in serial for comparison
		std::vector<int> serialImageVector = returnRGBMap(inputImage);
		CImg<unsigned char> serialImage = historamEqualiseSerial(serialImageVector, inputImage);
		serialImage.display("Serial Output For Comparison", true, 0, true);
		/*----End dispay----*/

		/*----Ending program----*/
		//Program End
		return 0;
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}
	return 0;
}