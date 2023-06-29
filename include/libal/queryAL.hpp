#include <memory>
#include <armadillo>

template <typename Strategy>
class QueryAL {
private:
    std::shared_ptr<Strategy> strategy;
    int buffer_size;
    int n_features;
    arma::mat samples_buffer;

public:
    QueryAL(
        std::shared_ptr<Strategy> s,
        int b,
        int f
    ) : model(m), annotator(a), strategy(s), buffer_size(b), n_features(f) {}

    bool Feed(arma::mat);

    arma::mat Select() {
        strategy->Select();
    }

    arma::mat Get(arma::mat samples) {
        Feed(samples);
        return Select();
    }

    void Flush() {
        samples_buffer.set_size(n_features, buffer_size);
    }

    void Resize(int n, int b) {
        n_features = n;
        buffer_size = b;
        samples_buffer.resize(n_features, buffer_size);
    }






    ~QueryAL() {}
};
