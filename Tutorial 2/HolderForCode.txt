#include "LibraryImport.h"
#include "MappingOperationsSerial.h"
#include "Printing.h"
//#include "Constants.h"
#define K_NUM_BINS 256

//#define MAX_SOURCE_SIZE (0x100000)

using namespace cimg_library;

int getBinsize(int width, int height) { //I can't believe this is all to come up with even bins :)
	std::vector<int> arr;
	int n = width;

	for (int i = 1; i <= n; ++i) {
		if (n % i == 0)
			arr.push_back(i);
	}

	cout << "The bin size is: " << arr.size() << "\n";
	cout << "The bin numbers are: ";

	for (int i = 0; i <= arr.size() - 1; ++i) {
		std::cerr << arr[i] << " ";
	}
	cout << "\n";
	bool checker = false;
	int binSize = 0;

	while (!checker) {
		cout << "Enter a number from above: ";
		binSize = 0;
		try {
			cin >> binSize;;
			if (cin.fail()) { throw(std::invalid_argument("Input was not a valid number, please enter a valid integer above.")); }
			cin.clear();
			cin.ignore();
			if (std::binary_search(arr.begin(), arr.end(), binSize)) { checker = true; }
			else { throw(binSize); }
		}
		catch (int size) { cout << "Element is not in the array.\n"; }
		catch (std::invalid_argument& e) {
			cout << e.what() << endl;
			cin.clear();
			cin.ignore();
			binSize = 0;
		}
		catch (...) {
			cout << "Error detected please enter a valid number.\n";
			cin.clear();
			cin.ignore();
		};
	}

	cout << "Bin size is: " << binSize << "\n";
	return binSize;
}

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
		int totalSize = inputImage.size();
		cout << "Total size: " << totalSize << "\n";

		cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY, inputImage.size() * sizeof(unsigned int));
		cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, K_NUM_BINS * sizeof(unsigned int)); //should be the same as input image
		queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, inputImage.size() * sizeof(unsigned int), &inputImage.data()[0]);
		cl::Kernel kernel = cl::Kernel(program, "histogramMaker");
		kernel.setArg(0, inputImageBuffer);
		kernel.setArg(1, histogramBuffer);
		kernel.setArg(2, (unsigned int)inputImage.size());
		

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange);
		vector<int> histogram(K_NUM_BINS);
		queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(unsigned int), &histogram.data()[0]);


		cl::Buffer cumulativeHistogramBuffer(context, CL_MEM_READ_WRITE, histogram.size() * sizeof(unsigned int));
		queue.enqueueWriteBuffer(histogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(unsigned int), &histogram.data()[0]);
		kernel = cl::Kernel(program, "scan_add");
		kernel.setArg(0, histogramBuffer);
		kernel.setArg(1, cumulativeHistogramBuffer);
		kernel.setArg(2, cl::Local(histogram.size() * sizeof(unsigned int)));
		kernel.setArg(3, cl::Local(histogram.size() * sizeof(unsigned int)));
		
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(histogram.size()), cl::NullRange);
		vector<int>cumHist(K_NUM_BINS);
		queue.enqueueReadBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(unsigned int), &cumHist.data()[0]);
		queue.flush();


		cl::Buffer normalisedHistogramBuffer(context, CL_MEM_READ_WRITE, cumHist.size() * sizeof(float));
		queue.enqueueWriteBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, cumHist.size()*sizeof(unsigned int), &cumHist.data()[0]);
		kernel = cl::Kernel(program, "normaliseHistogram");
		kernel.setArg(0, cumulativeHistogramBuffer);
		kernel.setArg(1, normalisedHistogramBuffer);
		kernel.setArg(2, totalSize); //Sets up the argument corresponding to the kernel function in my_kernels.cl

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
		queue.enqueueWriteBuffer(scaleHistogramBuffer, CL_TRUE, 0, scaleHist.size() * sizeof(unsigned int), &scaleHist.data()[0]);
		kernel = cl::Kernel(program, "translateByLookup");
		kernel.setArg(0, inputImageBuffer);
		kernel.setArg(1, outputImageBuffer);
		kernel.setArg(2, scaleHistogramBuffer);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange);
		vector<unsigned int> outputImageVector(inputImage.size());
		queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, inputImage.size() * sizeof(unsigned int), &outputImageVector.data()[0]);
		

		//Display Final Parallel Result
		CImg<unsigned char> output_image(outputImageVector.data(), inputImage.width(), inputImage.height(), inputImage.depth(), 1);
		CImgDisplay disp_output(output_image, "Kernel output");




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