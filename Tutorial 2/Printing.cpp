#include "Printing.h"
#include <iostream>
#include <map>
#include <vector>

#pragma once

using namespace std;

void print_map(map<int, int> const& m) {
	for (auto const& pair : m) {
		cout << "|" << pair.first << "| " << pair.second << "\n";
	}
}

void print_map(map<int, float> const& m) {
	for (auto const& pair : m) {
		cout << "|" << pair.first << "| " << pair.second << "\n";
	}
}

void print_vector(std::vector<int> vectorData) {
	for (int i = 0; i < vectorData.size(); ++i) {
		cout << i << " - " << vectorData[i] << "\n";
	}
}

void print_vector(std::vector<unsigned char> vectorData) {
	for (int i = 0; i < vectorData.size(); ++i) {
		cout << int(vectorData[i]) << "\n";
	}
}