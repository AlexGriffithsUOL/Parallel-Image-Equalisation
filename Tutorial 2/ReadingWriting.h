#include <iostream>
#include "CImg.h"
#include <map>
#pragma once

using namespace cimg_library;
CImg<unsigned char> historamEqualiseSerial(std::map<int, int> readableMap, CImg<unsigned char> origImage);