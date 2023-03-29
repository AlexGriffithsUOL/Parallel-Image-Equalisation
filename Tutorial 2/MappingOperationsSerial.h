#include <vector>
#include <map>
#include "CImg.h"
#pragma once

using namespace cimg_library;
//Data Vectorisation
std::vector<int> vectoriseData(CImg<unsigned char>);
std::vector<int> vectoriseData(std::map<int, int>);

//Create map
std::map<int, int> createHistogram(std::vector<int>);

//Cumulative Histogram
std::map<int, int> createCumulativeHistogram(std::map<int, int>);

//Turn into float Map
std::map<int, float> createFloatHistogram(std::map<int, int> assignedMap, int imageSize);

//Turn into RGB Values
std::map<int, int> createRGBMap(std::map<int, float> assignedMap);

//Full run of the functions
CImg<unsigned char> returnRGBMap(CImg<unsigned char> inputImage);