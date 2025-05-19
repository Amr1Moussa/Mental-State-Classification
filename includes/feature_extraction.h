#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <vector>

using namespace std;

double mean(const vector<double>& data);
double variance(const vector<double>& data, double data_mean);
double skewness(const vector<double>& data, double data_mean, double data_std);
double kurtosis(const vector<double>& data, double data_mean, double data_std);
double entropy(const vector<double>& data);

#endif // FEATURE_EXTRACTION_H
