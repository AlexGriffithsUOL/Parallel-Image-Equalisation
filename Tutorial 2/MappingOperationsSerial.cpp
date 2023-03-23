#include "MappingOperationsSerial.h"
#include "ConversionSerial.h"
#pragma once

using namespace std;

std::vector<int> vectoriseData(CImg<unsigned char> img) {
	std::vector<int> vectorisedData;
	for (int i = 0; i < img.size(); ++i) {
		vectorisedData.push_back(img._data[i]);
	}
	return vectorisedData;
}

std::vector<int>vectoriseData(std::map<int, int> img) {
	std::vector<int> vectorisedData;
	for (int i = 0; i < img.size(); ++i) {
		vectorisedData.push_back(img[i]);
	}
	return vectorisedData;
}

//Create map
std::map<int, int> createHistogram(std::vector<int> vectorData) {
	std::map<int, int> assignedMap;
	for (int i = 0; i < vectorData.size(); ++i) {
		++assignedMap[vectorData[i]];
	}
	return assignedMap;
}

//Cumulative Histogram
std::map<int, int> createCumulativeHistogram(std::map<int, int> assignedMap) {
	int total = 0;
	for (int i = 0; i < assignedMap.size(); ++i) {
		assignedMap[i] = assignedMap[i] + total;
		total = assignedMap[i];
	}
	return assignedMap;
}

//Turn into float Map
std::map<int, float> createFloatHistogram(std::map<int, int> assignedMap, int imageSize) {
	std::map<int, float> floatMap;
	for (int i = 0; i < assignedMap.size(); ++i) {
		floatMap[i] = float(assignedMap[i]) / imageSize;
	}
	return floatMap;
}

//Turn into RGB Values
std::map<int, int> createRGBMap(std::map<int, float> assignedMap) {
	std::map<int, int> newMap;
	for (int i = 0; i < assignedMap.size(); ++i) {
		newMap[i] = HSVtoRGB(0.0, 0.0, (assignedMap[i] * 100));//*255.0;
	}
	return newMap;
}

std::vector<int> returnRGBMap(CImg<unsigned char> inputImage) {
	int totalSize = inputImage.width() * inputImage.height() * inputImage.spectrum();
	std::map<int, int> newMap;
	//Vectorise data
	std::vector<int> vectorisedImage = vectoriseData(inputImage); //change to output_image below or image_input above

	//Create map
	newMap = createHistogram(vectorisedImage);

	//Cumulative Histogram
	newMap = createCumulativeHistogram(newMap);

	//Normalise
	std::map<int, float> floatMap;
	floatMap = createFloatHistogram(newMap, totalSize);

	//Turn into corresponding RGB values
	newMap = createRGBMap(floatMap);

	std::vector<int>tempArr = vectoriseData(newMap);
	return tempArr;
}


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