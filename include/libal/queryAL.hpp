#ifndef LIBAL_QUERYAL_HPP
#define LIBAL_QUERYAL_HPP

#include <memory>
#include <armadillo>

template <typename Strategy>
class QueryAL {
private:
    std::shared_ptr<Strategy> m_strategy;
    int m_buffer_size;
    int m_n_features;
    arma::mat m_samples_buffer;

public:
    QueryAL(
        std::shared_ptr<Strategy> s,
        int b,
        int f
    ) : m_strategy(s), m_buffer_size(b), m_n_features(f) {}

	/**
	 * @brief      Fill the buffer with the given samples. If the buffer is full, the size is going
	 * 				to be exceeded unlimitedly. It is the caller's responsibility to be responsible.
	 * 				The buffer is implemented as a matrix where each column is a sample and
	 * 				each row is a feature.
	 *
	 * @param      samples  The samples matrix where each column is a sample and each row is a
	 * 						feature.
	 * @exception  std::invalid_argument  If the number of features in the given samples matrix
	 * 						is not equal to the number of features in the buffer.
	 */
    void feed(arma::mat & newSamples) {
		if (newSamples.n_rows != m_n_features) {
			throw std::invalid_argument(
				"The number of features in the given samples matrix is not equal to the number"
				" of features in the buffer.");
		}
		m_samples_buffer.insert_cols(m_samples_buffer.n_cols, newSamples);
	}

    arma::mat select() {
        m_strategy->select(m_samples_buffer);
    }

    arma::mat get(arma::mat &samples) {
        feed(samples);
        return select();
    }

    void flush() {
        m_samples_buffer.set_size(m_n_features, m_buffer_size);
    }

    void resize(int n, int b) {
        m_n_features = n;
        m_buffer_size = b;
        m_samples_buffer.resize(m_n_features, m_buffer_size);
    }

    ~QueryAL() {}
};

#endif
