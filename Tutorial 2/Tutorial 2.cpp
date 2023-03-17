#include "LibraryImport.h"
#include "MappingOperationsSerial.h"
#include "Printing.h"

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
		CImg<unsigned char> image_input(image_filename.c_str());
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





		//Custom User kernel
		const int LIST_SIZE = 256; //List size is set to the array size.
		int* A = (int*)malloc(sizeof(int) * image_input.size());
		int* B = (int*)malloc(sizeof(int) * LIST_SIZE);

		for (int i = 0; i < image_input.size(); i++) { //Remove this 
			A[i] = image_input._data[i];
		} 





		cl::Buffer a_mem_obj(context, CL_MEM_READ_ONLY, image_input.size() * sizeof(int)); 
		cl::Buffer b_mem_obj(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int));
		cl::Buffer c_mem_obj(context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int));

		queue.enqueueWriteBuffer(a_mem_obj, CL_TRUE ,0, image_input.size() * sizeof(int), A); 
		queue.enqueueWriteBuffer(b_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), B);

		cl::Kernel kernel = cl::Kernel(program, "createHistogram");
		kernel.setArg(0, a_mem_obj);
		kernel.setArg(1, b_mem_obj);
		kernel.setArg(2, c_mem_obj);

		size_t global_item_size = image_input.size(); // Process the entire lists
		size_t local_item_size = 128; // Divide work items into groups of 64

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_item_size), cl::NDRange( local_item_size)); //Gets the range in the devices for the kernels

		// Read the memory buffer C on the device to the local variable C
		int* C = (int*)malloc(sizeof(int) * LIST_SIZE);
		queue.enqueueReadBuffer(c_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), &C[0]); //Reads the output buffer back from the device



		//Translate C to vector
		vector<int>newTempArr;
		int total = 0;
		for (int i = 0; i < LIST_SIZE; i++) {
			printf("%d - %d\n", i, C[i]);
			total += C[i];
			newTempArr.push_back(C[i]);//Add to array
		}
		cout << "Total size: " << total << "\n";

		//Clean up
		queue.flush();
		queue.finish();
		free(A);
		free(B);
		free(C);

	
		
		

		//Create cumulative histogram
		A = (int*)malloc(sizeof(int) * LIST_SIZE);
		B = (int*)malloc(sizeof(int) * LIST_SIZE);

		for (int i = 0; i < newTempArr.size(); i++) { //Remove this 
			A[i] = newTempArr[i];
			B[i] = 0;
		}





		a_mem_obj = cl::Buffer (context, CL_MEM_READ_WRITE, LIST_SIZE * sizeof(int));
		b_mem_obj = cl::Buffer (context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int));

		queue.enqueueWriteBuffer(a_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), A);
		queue.enqueueWriteBuffer(b_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), B);

		kernel = cl::Kernel(program, "scan_hs");
		kernel.setArg(0, a_mem_obj);
		kernel.setArg(1, b_mem_obj);

		global_item_size = LIST_SIZE; // Process the entire lists
		local_item_size = 64; // Divide work items into groups of 64

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_item_size), cl::NDRange(local_item_size)); //Gets the range in the devices for the kernels

		// Read the memory buffer C on the device to the local variable C
		C = (int*)malloc(sizeof(int) * LIST_SIZE);
		queue.enqueueReadBuffer(b_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), &C[0]); //Reads the output buffer back from the device

		//Translate C to vector
		/**/vector<int>newertempArr;
		total = 0;
		for (int i = 0; i < LIST_SIZE; i++) {
			printf("%d - %d\n", i, C[i]);
			total += C[i];
			newertempArr.push_back(C[i]);//Add to array
		}
		cout << "Total size: " << total << "\n";

		//Clean up
		queue.flush();
		queue.finish();
		free(A);
		free(B);
		free(C);
		

		


		//Contrast  via lookup table
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
		cl::Buffer lookUpTable(context, CL_MEM_READ_ONLY, newTempArr.size() * sizeof(int));

		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueWriteBuffer(lookUpTable, CL_TRUE, 0, newTempArr.size() * sizeof(int), &newTempArr[0]);// ampersand at the end bit points to the memory location of the tempArray element 0,

		kernel = cl::Kernel(program, "translateByLookup");
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_image_output);
		kernel.setArg(2, lookUpTable); //Sets up the argument corresponding to the kernel function in my_kernels.cl

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange); //Gets the range in the devices for the kernels

		vector<unsigned char> output_buffer(image_input.size());
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]); //Reads the output buffer back from the device

		queue.flush();
		queue.finish();



		//Equalisation in serial for comparison
		std::vector<int> tempArr = returnRGBMap(image_input);
		CImg<unsigned char> newer = historamEqualiseSerial(tempArr, image_input);
		CImgDisplay oioi(newer, "Serial output");

		//Display Final Parallel Result
		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "Kernel output");

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