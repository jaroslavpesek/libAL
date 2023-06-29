#include <gtest/gtest.h>

#include <memory>
#include <armadillo>

#include <libal/uncertainty.hpp>

/**
 * @brief      Tests the method LCUncertainty, 3 classes.
 */
TEST(uncertainty_strategies, LCUncertainty) {
	arma::mat samples = arma::mat({
		{0.1, 0.2, 0.7},
		{0.3, 0.4, 0.3},
		{0.5, 0.2, 0.3},
		{0.1, 0.2, 0.7},
		{0.3, 0.4, 0.3},
		{0.5, 0.2, 0.3},
		{0.1, 0.2, 0.7},
		{0.3, 0.4, 0.3},
		{0.5, 0.2, 0.3},
		{0.1, 0.2, 0.7},
		{0.3, 0.4, 0.3},
		{0.5, 0.2, 0.3},
		{0.1, 0.2, 0.7},
		{0.3, 0.4, 0.3},
		{0.5, 0.2, 0.3},
		{0.1, 0.2, 0.7},
		{0.3, 0.4, 0.3},
		{0.5, 0.2, 0.3},
		{0.1, 0.2, 0.7},
		{0.3, 0.4, 0.3},
		{0.5, 0.2, 0.3}
	}).t();
	arma::uvec expected = {1, 4, 7, 10, 13, 16, 19};
	std::shared_ptr<LCUncertainty> strategy = std::make_shared<LCUncertainty>(7);
	arma::uvec actual = strategy->select(samples);
	EXPECT_EQ(expected.size(), actual.size());
	for (int i = 0; i < expected.size(); i++) {
		EXPECT_TRUE(std::find(actual.begin(), actual.end(), expected[i]) != actual.end());
	}
}

/**
 * @brief      Tests the method LCUncertainty, 33 classes.
 */
TEST(uncertainty_strategies, LCUncertainty_33_classes) {
	// generate a matrix of 100 samples with 33 classes
	arma::mat samples = arma::randu(33, 100);
	// normalize the columns so it make 1 in sum
	samples.each_col([](arma::vec &v) { v /= arma::sum(v); });
	std::shared_ptr<LCUncertainty> strategy = std::make_shared<LCUncertainty>(10);
	arma::uvec actual = strategy->select(samples);
	EXPECT_EQ(10, actual.size());
}

