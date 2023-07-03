#include <armadillo>

#ifndef LIBAL_QBC_HPP
#define LIBAL_QBC_HPP

template <typename Model>
class BaseQBC {
public:
	BaseQBC() {}
	explicit BaseQBC(int n) : n(n) {}
	virtual arma::uvec select(const arma::mat &unlabelled, const arma::mat &labelled) = 0;
protected:
	int n;
};

template <typename Model>
class KLDiverganceQBC: public BaseQBC<Model> {

};

#endif // LIBAL_QBC_HPP
