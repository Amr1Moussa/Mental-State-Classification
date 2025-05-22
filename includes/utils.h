#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

using namespace std;

bool load_csv(const string& filename, vector<vector<double>>& features, vector<int>& labels, const string& label_header);
void normalize_rows(vector<vector<double>>& data);
vector<double> dct_reduce(const vector<double>& input, int keep_dim);
void standardize_features(vector<vector<double>>& features);
void split_dataset(const vector<vector<double>>& features,
                   const vector<int>& labels,
                   double train_ratio, double val_ratio,
                   vector<vector<double>>& train_features,
                   vector<int>& train_labels,
                   vector<vector<double>>& val_features,
                   vector<int>& val_labels,
                   vector<vector<double>>& test_features,
                   vector<int>& test_labels);



#endif

