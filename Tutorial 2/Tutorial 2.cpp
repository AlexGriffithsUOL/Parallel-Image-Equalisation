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

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test(1).pgm";

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
		CImg<unsigned char> display_image(image_filename.c_str());
		inputImage._spectrum = 1;
		CImgDisplay disp_input(display_image, "input");

		cl::Context context = GetContext(platform_id, device_id);		
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		cl::CommandQueue queue(context);
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

		//Start
		cout << "Total size image size: " << inputImage.size() << "\n";
		cout << "Select a bin size from below:" << "\n";

		int value = inputImage.max();
		for (int i = 0; i < (ceil(log2(value))+ 1); ++i) {
			cout << i << "| " << (pow(2, i)) << "\n";
		}

		unsigned int userinput;
		cin >> userinput;
		int binNumber = pow(2, userinput);
		//unsigned int binSize = (pow(2,userinput));
		unsigned int binSize = pow(2, ceil(log2(value))) / (binNumber);



		cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY, inputImage.size() * sizeof(unsigned int));
		cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, (binNumber) * sizeof(unsigned int)); //should be the same as input image
		queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, inputImage.size() * sizeof(unsigned int), &inputImage.data()[0]);
		cl::Kernel kernel = cl::Kernel(program, "hist_simple");
		kernel.setArg(0, inputImageBuffer);
		kernel.setArg(1, histogramBuffer);
		kernel.setArg(2, binSize);
		

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange);
		vector<unsigned int> histogram(binNumber);
		queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(unsigned int), &histogram.data()[0]);



		cout << "histogram" << histogram.size() << "\n";
		for (unsigned int i = 0; i < histogram.size(); i++) {
			cout << i << ": " << histogram[i] << "\n";
		}




		cl::Buffer cumulativeHistogramBuffer(context, CL_MEM_READ_WRITE, histogram.size() * sizeof(unsigned int));
		queue.enqueueWriteBuffer(histogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(unsigned int), &histogram.data()[0]);
		kernel = cl::Kernel(program, "scan_add");
		kernel.setArg(0, histogramBuffer);
		kernel.setArg(1, cumulativeHistogramBuffer);
		kernel.setArg(2, cl::Local(histogram.size() * sizeof(unsigned int)));
		kernel.setArg(3, cl::Local(histogram.size() * sizeof(unsigned int)));
		
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange);
		vector<int>cumHist(binNumber);
		queue.enqueueReadBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(unsigned int), &cumHist.data()[0]);
		queue.flush();

		cout << "Cumulative histogram\n";
		for (unsigned int i = 0; i < cumHist.size(); i++) {
			cout << i << ": " << cumHist[i] << "\n";
		}


		cl::Buffer normalisedHistogramBuffer(context, CL_MEM_READ_WRITE, cumHist.size() * sizeof(float));
		queue.enqueueWriteBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, cumHist.size() * sizeof(unsigned int), &cumHist.data()[0]);
		kernel = cl::Kernel(program, "normaliseHistogram");
		kernel.setArg(0, cumulativeHistogramBuffer);
		kernel.setArg(1, normalisedHistogramBuffer);
		kernel.setArg(2, (int)(inputImage.size())); //Sets up the argument corresponding to the kernel function in my_kernels.cl

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(cumHist.size()), cl::NullRange); //Gets the range in the devices for the kernels
		vector<float>normHist(cumHist.size());
		queue.enqueueReadBuffer(normalisedHistogramBuffer, CL_TRUE, 0, cumHist.size()*sizeof(float), &normHist.data()[0]); //Reads the output buffer back from the device

		cout << "Normalised histogram\n";
		for (unsigned int i = 0; i < normHist.size(); i++) {
			cout << i << ": " << normHist[i] << "\n";
		}

		cl::Buffer scaleHistogramBuffer(context, CL_MEM_READ_WRITE, normHist.size() * sizeof(unsigned int));
		queue.enqueueWriteBuffer(normalisedHistogramBuffer, CL_TRUE, 0, normHist.size() * sizeof(float), &normHist.data()[0]);
		kernel = cl::Kernel(program, "scaleTo255");
		kernel.setArg(0, normalisedHistogramBuffer);
		kernel.setArg(1, scaleHistogramBuffer);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(normHist.size()), cl::NullRange);
		vector<unsigned int> scaleHist(normHist.size());
		queue.enqueueReadBuffer(scaleHistogramBuffer, CL_TRUE, 0, scaleHist.size()*sizeof(unsigned int), &scaleHist.data()[0]);

		cout << "Scaled histogram\n";
		for (int i = 0; i < scaleHist.size(); i++) {
			cout << i << ": " << scaleHist[i] << "\n";
		}

		//Contrast  via lookup table
		cl::Buffer outputImageBuffer(context, CL_MEM_READ_WRITE, inputImage.size() * sizeof(unsigned int));
		//queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, inputImage.size() * sizeof(unsigned int), &inputImage.data()[0]);
		queue.enqueueWriteBuffer(scaleHistogramBuffer, CL_TRUE, 0, scaleHist.size() * sizeof(unsigned int), &scaleHist.data()[0]);
		kernel = cl::Kernel(program, "translateByLookup");
		kernel.setArg(0, inputImageBuffer);
		kernel.setArg(1, outputImageBuffer);
		kernel.setArg(2, scaleHistogramBuffer);
		kernel.setArg(3, binSize);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange);
		vector<unsigned int> outputImageVector(inputImage.size());
		queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, inputImage.size() * sizeof(unsigned int), &outputImageVector.data()[0]);
		queue.flush();

		//Display Final Parallel Result
		CImg<unsigned char> output_image(outputImageVector.data(), inputImage.width(), inputImage.height(), inputImage.depth(), 1);
		//CImgDisplay disp_output(output_image, "Kernel output");
		output_image.display();




		//Equalisation in serial for comparison
		std::vector<int> tempArr = returnRGBMap(inputImage);
		CImg<unsigned char> newer = historamEqualiseSerial(tempArr, inputImage);
		CImgDisplay oioi(newer, "Serial output");

		

		int huh;

		cin >> huh;
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