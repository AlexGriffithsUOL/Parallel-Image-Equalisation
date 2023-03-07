#include "MappingOperationsSerial.h"
#include "ConversionSerial.h"
#pragma once



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