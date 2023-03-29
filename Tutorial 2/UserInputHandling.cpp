#include "UserInputHandling.h"

using namespace std;

int checkInputType(int lowerR, int upperR) { //Upper and lower bounds passed in for later range check
	int input;
	while (true) { //While loop to wait until broken
		if (!(cin >> input)) //Checks against the type of input provided
		{
			if (cin.bad()) { //Checks for bad flag in cin
				cin.clear(); //Clears to stop skipping over input 
				cin.ignore(); //Ignores flag incase that does not work
				cout << "Input is not of the correct type, please enter a positive integer value.\n"; //Prints out error message
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
		else { //When user input is correct type it checks range
			if (checkInputRange(lowerR, upperR, input)) { return input; break; } //Returns the input then breaks the loop to be sure nothing has gone wrong
		}
	}
}

bool checkInputRange(int lowerR, int upperR, int input) { //Upper and lower bounds passed in with input to allow for a check
	if (input >= lowerR  && input < (upperR + 1)) { //Checks if inputs between the ranges
		cout << "Valid input selected : " << input << "\n"; //Confirms valid input
		return true; //Returns the boolean to true so that the input can be returned
	}
	else {
		cin.clear(); //Clears the cin flags when an error occurs
		cin.ignore(); //Ignores the error with cin
		cout << "Value out of range, select a value from " << lowerR << " to " << upperR << "\n"; //Reiterates the correct range that the user can input
		return false; //Returns a false check
	}
}