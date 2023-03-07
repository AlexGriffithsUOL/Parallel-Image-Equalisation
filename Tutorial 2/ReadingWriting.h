#include <iostream>
#include "CImg.h"
#include <vector>
#pragma once

using namespace cimg_library;
CImg<unsigned char> historamEqualiseSerial(std::vector<int> readableMap, CImg<unsigned char> origImage);