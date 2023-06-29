#ifndef LIBAL_UNCERTAINTY_HPP
#define LIBAL_UNCERTAINTY_HPP

#include <armadillo>
#include <memory>

class BaseUncertainty {
public:
	BaseUncertainty() {}
	BaseUncertainty(int n) : n(n) {}
	virtual arma::uvec select(const arma::mat &samples) = 0;
protected:
	int n;
};

class LCUncertainty: public BaseUncertainty {
public:
	using BaseUncertainty::BaseUncertainty;
	/**
	 * @brief      Selects the most uncertain samples from the given sample matrix with least
	 * 				confidence measure.
	 *
	 * @param      samples  The samples matrix where each column is a sample and each row is
	 * 						a class probability as output of model's predict_proba method.
	 *
	 * @return     The indices of the selected samples from the given samples matrix. Size could
	 * 			   be less than n if the number of samples in the given samples matrix is less
	 * 			   than n.
	 */
	arma::uvec select(const arma::mat &samples) override {
		arma::mat uncertainty = 1 - arma::max(samples, 0);
		arma::uvec indices = arma::sort_index(uncertainty, "descend");
		return indices.head(n);
	}
};

class MCUncertainty: public BaseUncertainty {
public:
	using BaseUncertainty::BaseUncertainty;
	/**
	 * @brief      Selects the most uncertain samples from the given sample matrix with margin of
	 * 				confidence measure.
	 *
	 * @param      samples  The samples matrix where each column is a sample and each row is
	 * 						a class probability as output of model's predict_proba method.
	 *
	 * @return     The indices of the selected samples from the given samples matrix. Size could
	 * 			   be less than n if the number of samples in the given samples matrix is less
	 * 			   than n.
	 */
	arma::uvec select(const arma::mat &samples) override
	{
		arma::mat sorted = arma::sort(samples, "descend", 0);
		arma::mat margin = sorted.row(0) - sorted.row(1);
		arma::uvec indices = arma::sort_index(margin, "descend");
		return indices.head(n);
	}
};

class EUncertainty: public BaseUncertainty {
public:
	using BaseUncertainty::BaseUncertainty;
	/**
	 * @brief      Selects the most uncertain samples from the given sample matrix with entropy
	 * 				measure.
	 *
	 * @param      samples  The samples matrix where each column is a sample and each row is
	 * 						a class probability as output of model's predict_proba method.
	 *
	 * @return     The indices of the selected samples from the given samples matrix. Size could
	 * 			   be less than n if the number of samples in the given samples matrix is less
	 * 			   than n.
	 */
	arma::uvec select(const arma::mat &samples) override
	{
		arma::mat entropy = -arma::sum(samples % arma::log(samples), 0);
		arma::uvec indices = arma::sort_index(entropy, "descend");
		return indices.head(n);
	}
};

class RUncertainty: public BaseUncertainty {
public:
	using BaseUncertainty::BaseUncertainty;
	/**
	 * @brief      Selects the most uncertain samples from the given sample matrix with ratio
	 * 				measure.
	 *
	 * @param      samples  The samples matrix where each column is a sample and each row is
	 * 						a class probability as output of model's predict_proba method.
	 *
	 * @return     The indices of the selected samples from the given samples matrix. Size could
	 * 			   be less than n if the number of samples in the given samples matrix is less
	 * 			   than n.
	 */
	arma::uvec select(const arma::mat &samples) override
	{
		arma::mat sorted = arma::sort(samples, "descend", 0);
		arma::mat ratio = sorted.row(0) / sorted.row(1);
		arma::uvec indices = arma::sort_index(ratio, "descend");
		return indices.head(n);
	}
};

#endif // LIBAL_UNCERTAINTY_HPP
