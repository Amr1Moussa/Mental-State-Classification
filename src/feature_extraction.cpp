#include "../includes/feature_extraction.h"
#include <cmath>
#include <numeric>

using namespace std;

double mean(const vector<double>& data) {
    double sum = accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double variance(const vector<double>& data, double data_mean) {
    double var_sum = 0.0;
    for (auto val : data) {
        double diff = val - data_mean;
        var_sum += diff * diff;
    }
    return var_sum / data.size();
}

double skewness(const vector<double>& data, double data_mean, double data_std) {
    double skew_sum = 0.0;
    for (auto val : data) {
        double diff = val - data_mean;
        skew_sum += pow(diff / data_std, 3);
    }
    return skew_sum / data.size();
}

double kurtosis(const vector<double>& data, double data_mean, double data_std) {
    double kurt_sum = 0.0;
    for (auto val : data) {
        double diff = val - data_mean;
        kurt_sum += pow(diff / data_std, 4);
    }
    return kurt_sum / data.size() - 3.0;
}

double entropy(const vector<double>& data) {
    vector<double> probs(data.size());
    double sum_abs = 0.0;
    for (auto val : data) sum_abs += abs(val);
    if (sum_abs == 0) return 0.0;

    for (size_t i = 0; i < data.size(); ++i) {
        probs[i] = abs(data[i]) / sum_abs;
    }

    double ent = 0.0;
    for (auto p : probs) {
        if (p > 1e-12) ent -= p * log2(p);
    }
    return ent;
}
