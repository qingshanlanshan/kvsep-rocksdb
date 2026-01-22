
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <cassert>

class UniformIntGenerator {
public:
    UniformIntGenerator(int min, int max, int seed = 0)
        : dist_(min, max), engine_(seed) 
    {
        assert(min <= max && "min should be less than or equal to max");

        // uniform distribution for integer sampling
        dist_ = std::uniform_int_distribution<int>(min, max);
    }

    int next() {
        return dist_(engine_);
    }

private:
    std::uniform_int_distribution<int> dist_;
    std::mt19937_64 engine_;
};



class ZipfGenerator {
public:
    // s = skew parameter (s > 0)
    ZipfGenerator(int start, int end, double s, int seed = 0)
        : start_(start), end_(end), s_(s), dist_(0.0, 1.0), engine_(seed)
    {
        assert(start <= end);
        assert(s > 0.0);

        int n = end - start + 1;
        cdf_.resize(n);

        // Compute normalization constant = 1 / H_{N,s}
        double harmonic = 0.0;
        for (int i = 1; i <= n; i++) {
            harmonic += 1.0 / std::pow(i, s_);
        }
        double norm = 1.0 / harmonic;

        // Build CDF
        double cumulative = 0.0;
        for (int i = 1; i <= n; i++) {
            cumulative += norm / std::pow(i, s_);
            cdf_[i - 1] = cumulative;
        }
        cdf_[n - 1] = 1.0; // ensure exact endpoint
    }

    int next() {
        double u = dist_(engine_);
        // Binary search over the CDF
        int idx = binary_search(u);
        return start_ + idx;
    }

private:
    int start_, end_;
    double s_;
    std::vector<double> cdf_;
    std::uniform_real_distribution<double> dist_;
    std::mt19937_64 engine_;

    int binary_search(double u) {
        int l = 0, r = (int)cdf_.size() - 1;
        while (l < r) {
            int mid = (l + r) / 2;
            if (cdf_[mid] >= u)
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }
};
