#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "model_tail.hpp"
#include "model_head.hpp"

auto tail = Bnn_tail();
auto head = Bnn_head();

int main () {
	// Set random seed
	Tensorlib::Bnn::rand_seed = time(NULL);

	// Format cout
	std::cout << std::fixed << std::setprecision(4);
	std::cout << tail.mc_samples << " " << tail.test_data.shape[0] << "\n";

	// Load the model test data and weights
	tail.load_test_data("data.bin");
	tail.load_weights("tail_weights.bin");
	head.load_weights("head_weights.bin");

	for (u32 i = 0; i < tail.test_data.shape[0]; i++) {
		auto v = tail.get_data(i);
		tail.forward(v);
		auto x = tail.output();
		head.forward(x);
	}
}
