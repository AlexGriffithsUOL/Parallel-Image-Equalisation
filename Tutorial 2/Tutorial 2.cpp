#include "LibraryImport.h"
#include "MappingOperationsSerial.h"
#include "Printing.h"

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
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//3.2 Load & build the device code
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

		//SECOND START OF USER CODE
		//Start
		//Creating histogram and normalising it
		std::map<int, int> tempMap;

		int totalSize = image_input.width() * image_input.height(); //change to output_image below or image_input above
		cout << "Total size: " << totalSize << "\n";
		cout << "Image size: " << image_input.size() << "\n"; //change to output_image below or image_input above

		//Vectorise data
		std::vector<int> vectorisedImage = vectoriseData(image_input); //change to output_image below or image_input above

		//Create map
		tempMap = createHistogram(vectorisedImage);
		print_map(tempMap);

		//Cumulative Histogram
		tempMap = createCumulativeHistogram(tempMap);

		//Normalise
		map<int, float> floatMap;
		floatMap = createFloatHistogram(tempMap, totalSize);

		//Turn into corresponding RGB values
		tempMap = createRGBMap(floatMap);

		vector<int>tempArr = vectoriseData(tempMap);
		//End
		//SECOND END OF USER CODE










		//Part 4 - device operations

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
		//comment out if not working
		cl::Buffer lookUpTable(context, CL_MEM_READ_ONLY, tempArr.size()*sizeof(int));//

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		//comment out if not working
		queue.enqueueWriteBuffer(lookUpTable, CL_TRUE, 0, tempArr.size()*sizeof(int), &tempArr[0]);// ampersand at the end bit points to the memory location of the tempArray element 0,
		/* buffer
		   CL_TRUE preserves the location in memory
		   0 is the offset
		   tempArr.size()*sizeof(int) is the number of bytes to write in or expect
		   &tempArr[0] refers a pointer to the first element of the tempArr so it knows where to read from */
		

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "histogramEqualisation");
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_image_output);
		//comment out if not working
		kernel.setArg(2, lookUpTable); //Sets up the argument corresponding to the kernel function in my_kernels.cl

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange); //Gets the range in the devices for the kernels

		vector<unsigned char> output_buffer(image_input.size());

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]); //Reads the output buffer back from the device

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "Kernel output");



		//Custom User kernel
		vector<int>histoVector(255);
		int ar(255);
		cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, histoVector.size());
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		cl::Kernel histogramKernel = cl::Kernel(program, "countToHistogram");
		histogramKernel.setArg(0, dev_image_input);
		histogramKernel.setArg(1, histogramBuffer);

		queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);

		queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, histoVector.size(), &ar);

		print_vector(histoVector);

		//ORIGINAL START OF USER CODE
	
		//ORIGINAL END OF USER CODE


		//CImg<unsigned char> newer = historamEqualiseSerial(tempMap, image_input);
		//CImgDisplay oioi(newer, "Serial output");
		int huh = 0;
		cout << "End of program - any key to close ";
		cin >> huh;
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}
	return 0;
} 


////SORT OUT BUFFER