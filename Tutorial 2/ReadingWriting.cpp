#include "ReadingWriting.h"
/*----This file is purely for testing and comparison----*/
using namespace cimg_library;
using namespace std;

//Serial histogram equalisation
CImg<unsigned char> historamEqualiseSerial(std::vector<int> readableMap, CImg<unsigned char> origImage) { //Take in a lookup table and the original image.
	CImg<unsigned char> newImg(origImage.width(), origImage.height(), 1, 1, 0); //Create a new image
	for (int i = 0; i < newImg.size(); ++i) { //For loop to go through each element
		int key = origImage._data[i]; //Key created from data
		newImg._data[i] = readableMap[key]; //Data used in lookup table to translate to new image.
	}
	return newImg; //Return image
}