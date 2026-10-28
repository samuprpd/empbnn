#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "model.hpp"

auto model = Classic_nn();

int main () {
	// Format cout
	std::cout << std::fixed << std::setprecision(4);
	std::cout << model.test_data.shape[0] << "\n";

	// Load the model test data and weights
	model.load_test_data("data.bin");
	model.load_weights("weights.bin");

	for (u32 i = 0; i < model.test_data.shape[0]; i++) {
		auto v = model.get_data(i);
		model.forward(v);
	}
}
