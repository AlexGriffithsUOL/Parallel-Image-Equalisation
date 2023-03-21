#include "LibraryImport.h"
#include "MappingOperationsSerial.h"
#include "Printing.h"
#include "Constants.h"

#define MAX_SOURCE_SIZE (0x100000)

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
		CImg<unsigned int> image_input(image_filename.c_str());
		image_input._spectrum = 1;
		CImgDisplay disp_input(image_input, "input");

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
		int totalSize = image_input.width() * image_input.height(); //change to output_image below or image_input above
		cout << "Total size: " << totalSize << "\n";
		cout << "Image size: " << image_input.size() << "\n"; //change to output_image below or image_input above

		


		cl::Buffer dev_input_image(context, CL_MEM_READ_ONLY, image_input.size() * sizeof(unsigned int));
		cl::Buffer hist_buffer(context, CL_MEM_READ_WRITE, K_NUM_BINS * sizeof(unsigned int)); //should be the same as input image
		queue.enqueueWriteBuffer(dev_input_image, CL_TRUE, 0, image_input.size() * sizeof(unsigned int), &image_input.data()[0]);
		cout << "Number of pixels in image: " << image_input.size() << endl;
		cl::Kernel kernel = cl::Kernel(program, "histogramMaker");
		kernel.setArg(0, dev_input_image);
		kernel.setArg(1, hist_buffer);
		kernel.setArg(2, (unsigned int)image_input.size());
		

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		vector<int> histogram(K_NUM_BINS);
		queue.enqueueReadBuffer(hist_buffer, CL_TRUE, 0, K_NUM_BINS * sizeof(unsigned int), &histogram.data()[0]);

		cout << "Histogram" << endl;
		for (unsigned int i = 0; i < histogram.size(); i++) {
			cout << i << ": " << histogram[i] << endl;
		}





		//Equalisation in serial for comparison
		std::vector<int> tempArr = returnRGBMap(image_input);
		CImg<unsigned char> newer = historamEqualiseSerial(tempArr, image_input);
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