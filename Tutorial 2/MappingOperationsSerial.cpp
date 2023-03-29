#include "MappingOperationsSerial.h"
#include "ReadingWriting.h"
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

//Vectoris the data
std::vector<int>vectoriseData(std::map<int, int> img) { //Takes in the map of the image
	std::vector<int> vectorisedData; //Temporary variable to store the vector
	for (int i = 0; i < img.size(); ++i) { //For loop  to write the image data into a vector
		vectorisedData.push_back(img[i]); //Writing the data to the vector
	}
	return vectorisedData; //Returning vector
}

//Create map
std::map<int, int> createHistogram(std::vector<int> vectorData) { //Vector data passed into to remap it
	std::map<int, int> assignedMap; //Creates a local map to store the data
	for (int i = 0; i < vectorData.size(); ++i) {
		++assignedMap[vectorData[i]]; //Increments using the vector data
	}
	return assignedMap; //Returns
}

//Cumulative Histogram
std::map<int, int> createCumulativeHistogram(std::map<int, int> assignedMap) { //Map passed in to pass data
	int total = 0; //Local running total to hold the cumulative sum
	for (int i = 0; i < assignedMap.size(); ++i) {
		assignedMap[i] = assignedMap[i] + total; //Current value at the index + running total to give the cumulative sum
		total = assignedMap[i]; //Reassigns the map to the total
	}
	return assignedMap; //Returns cumulative histogram
}

//Turn into float Map
std::map<int, float> createFloatHistogram(std::map<int, int> assignedMap, int imageSize) { //Float map create by parsing the assigned map and dividing the image size
	std::map<int, float> floatMap; //Local map to store data
	for (int i = 0; i < assignedMap.size(); ++i) {
		floatMap[i] = float(assignedMap[i]) / imageSize; //Divisor with a recast to float to allow for normalised values
	}
	return floatMap; //Returns map
}

//Turn into RGB Values
std::map<int, int> createRGBMap(std::map<int, float> assignedMap) { //Argument is a float map as normalisation returns decimal values
	std::map<int, int> newMap; //Local map to store data
	for (int i = 0; i < assignedMap.size(); ++i) {
		newMap[i] = HSVtoRGB(0.0, 0.0, (assignedMap[i] * 100)); //Calls external function to convert grey in HSV to RGB
	}
	return newMap; //Returns new map
}

CImg<unsigned char>returnRGBMap(CImg<unsigned char> inputImage) { //Takes the image and returns the contrasted image through all previous functions
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

	CImg<unsigned char> newImg = historamEqualiseSerial(tempArr, inputImage);
	
	return newImg;
}