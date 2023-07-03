#ifndef LIBAL_FARTHEST_SAMPLING_HPP
#define LIBAL_FARTHEST_SAMPLING_HPP

#include <armadillo>
#include <memory>
#include <mlpack.hpp>

#define NUMBER_OF_PROJECTIONS 10
#define NUMBER_OF_ELEMENTS_STORED 3
#define FURTHEST_NEIGHBOURS 5

class FarthestSampling {
public:
	FarthestSampling() {}
	explicit FarthestSampling(int n) : m_n(n) {}
	arma::uvec select(const arma::mat &labelled, const arma::mat &unlabelled) {
		mlpack::DrusillaSelect<> ds(labelled, NUMBER_OF_PROJECTIONS, NUMBER_OF_ELEMENTS_STORED);
		arma::Mat<size_t> neighbours;
		arma::mat distances;
		ds.Search(unlabelled, FURTHEST_NEIGHBOURS, neighbours, distances);
		arma::uvec indices =  arma::sort_index(distances, "descend");
		return indices.head(m_n);
	}
private:
	int m_n;
};



#endif // LIBAL_FARTHEST_SAMPLING_HPP
