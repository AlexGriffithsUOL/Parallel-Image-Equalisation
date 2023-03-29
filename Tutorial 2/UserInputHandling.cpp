#include "UserInputHandling.h"

using namespace std;
int checkInputType(int lowerR, int upperR) {
	int input;
	while (true) {
		if (!(cin >> input))
		{
			if (cin.bad()) {
				cin.clear();
				cin.ignore();
				cout << "Input is not of the correct type, please enter a positive integer value.\n";
			}
			else if (cin.eof()) {
				cin.clear();
				cin.ignore();
				cout << "Input is in an incorrect format, please enter a positive integer value.\n";
			}
			else {
				cin.clear();
				cin.ignore();
				cout << "Unknown error, enter a posiive integer value.\n";
			}
		}
		else {
			if (checkInputRange(lowerR, upperR, input)) { return input; break; }
		}
	}
}

bool checkInputRange(int lowerR, int upperR, int input) 
{
	if (input >= lowerR  && input < (upperR + 1)) {
		cout << "Valid input selected : " << input << "\n";
		return true;
	}
	else {
		cin.clear();
		cin.ignore();
		cout << "Value out of range, select a value from " << lowerR << " to " << upperR << "\n";
		return false;
	}
}