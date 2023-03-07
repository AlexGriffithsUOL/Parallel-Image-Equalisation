#include "ReadingWriting.h"

using namespace cimg_library;
using namespace std;

/*CImg<unsigned char> historamEqualiseSerial(std::map<int, int> readableMap, CImg<unsigned char> origImage) {
	CImg<unsigned char> newImg(origImage.width(), origImage.height(), 1, 1, 0);
	for (int i = 0; i < newImg.size(); ++i) {
		int key = origImage._data[i];
		newImg._data[i] = readableMap[key];
	}
	return newImg;
}*/

CImg<unsigned char> historamEqualiseSerial(std::vector<int> readableMap, CImg<unsigned char> origImage) {
	CImg<unsigned char> newImg(origImage.width(), origImage.height(), 1, 1, 0);
	for (int i = 0; i < newImg.size(); ++i) {
		int key = origImage._data[i];
		newImg._data[i] = readableMap[key];
	}
	return newImg;
}