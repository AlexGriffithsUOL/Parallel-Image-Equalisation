#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include "Utils.h"
#include "CImg.h"
#include <stdexcept>

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

int HSVtoRGB(float H, float S, float V) {
	if (H > 360 || H < 0 || S>100 || S < 0 || V>100 || V < 0) {
		cout << "The givem HSV values are not in valid range" << endl;
		return -1;
	}
	float s = S / 100;
	float v = V / 100;
	float C = s * v;
	float X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
	float m = v - C;
	float r, g, b;
	if (H >= 0 && H < 60) {
		r = C, g = X, b = 0;
	}
	else if (H >= 60 && H < 120) {
		r = X, g = C, b = 0;
	}
	else if (H >= 120 && H < 180) {
		r = 0, g = C, b = X;
	}
	else if (H >= 180 && H < 240) {
		r = 0, g = X, b = C;
	}
	else if (H >= 240 && H < 300) {
		r = X, g = 0, b = C;
	}
	else {
		r = C, g = 0, b = X;
	}
	int R = (r + m) * 255;
	int G = (g + m) * 255;
	int B = (b + m) * 255;
	return R;
}

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

void print_map(std::map<int, int> const& m) {
	cout << "\n NEW MAP ######################\n";
	for (auto const& pair : m) {
		cout << "|" << pair.first << "| " << pair.second << "\n";
	}
}

void print_map(std::map<int, float > const& m) {
	cout << "\n NEW MAP ######################\n";
	for (auto const& pair : m) {
		cout << "|" << pair.first << "| " << pair.second << "\n";
	}
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

		//Part 4 - device operations

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size());

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "identity");
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_image_output);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);

		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		//CImgDisplay disp_output(output_image, "output");


		//Creating histogram and normalising it
		std::vector<int> line = {};
		std::map<int, int> tempMap;

		int totalSize = output_image.width() * output_image.height();
		cout << "Total size: " << totalSize << "\n";
		cout << "Image size: " << output_image.size() << "\n";

		for (int i = 0; i < totalSize; ++i) {
			line.push_back(int(output_image._data[i]));
		}

		for (int i = 0; i < line.size(); ++i) {
			++tempMap[line[i]];
		}

		int total = 0;
		for (int i = 0; i < tempMap.size(); ++i) {
			tempMap[i] = tempMap[i] + total;
			total = tempMap[i];
		}

		std::map<int, float> floatMap;
		for (int i = 0; i < tempMap.size(); ++i) {
			floatMap[i] = HSVtoRGB(0, 0, float(tempMap[i]) / float(output_image.width() * output_image.height()) * 100);//*255.0;
		}
		print_map(floatMap);

		CImg<unsigned char> newImg(1024, 683, 1, 1, 0);
		cout << newImg.size() << "\n";
		for (int i = 0; i < newImg.size(); ++i) {
			int key = output_image._data[i];
			newImg._data[i] = floatMap[key];
		}
		newImg.save("allBlackTest.pgm");
		CImgDisplay local(newImg, "Woi");

		int huh = 0;
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