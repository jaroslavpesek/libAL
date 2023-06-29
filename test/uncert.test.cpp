#include <gtest/gtest.h>

#include <memory>
#include <armadillo>

#include <libal/queryAL.hpp>
#include <libal/uncertainty.hpp>

TEST(uncertainty_strategies, LCUncertainty) {
	arma::mat samples = {
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
	};
	arma::uvec expected = {2, 5, 8, 11, 14, 17, 20};
	std::shared_ptr<LCUncertainty> strategy = std::make_shared<LCUncertainty>(7);
	arma::uvec actual = strategy->select(samples);
	ASSERT_EQ(expected.n_elem, actual.n_elem);
	for (int i = 0; i < expected.n_elem; i++) {
		ASSERT_EQ(expected(i), actual(i));
	}
}
