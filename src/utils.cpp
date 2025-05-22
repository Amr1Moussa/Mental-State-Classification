#include "../includes/utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>

using namespace std;

bool is_number(const string& s, double& result) {
    try {
        size_t idx;
        result = stod(s, &idx);
        return idx == s.length();
    } catch (...) {
        return false;
    }
}

bool load_csv(const string& filename, vector<vector<double>>& features, vector<int>& labels, const string& label_header) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open CSV file: " << filename << endl;
        return false;
    }

    string line;
    getline(file, line);  // Header
    stringstream header_ss(line);
    vector<string> headers;
    string col;
    int label_index = -1;

    while (getline(header_ss, col, ',')) {
        headers.push_back(col);
        if (col == label_header) {
            label_index = headers.size() - 1;
        }
    }

    if (label_index == -1) {
        cerr << "Label column not found: " << label_header << endl;
        return false;
    }

    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        vector<double> row;
        int col_idx = 0;
        double val;
        int label = -1;

        while (getline(ss, token, ',')) {
            if (col_idx == label_index) {
                label = stoi(token);
            } else {
                if (!is_number(token, val) || isnan(val) || isinf(val)) {
                    val = 0.0;  // Replace NaN/Inf
                }
                row.push_back(val);
            }
            col_idx++;
        }

        if (row.size() > 0 && label >= 0) {
            features.push_back(row);
            labels.push_back(label);
        }
    }

    return true;
}

void normalize_rows(vector<vector<double>>& data) {
    for (auto& row : data) {
        double max_val = 0.0;
        for (double val : row) {
            if (abs(val) > max_val) max_val = abs(val);
        }
        if (max_val > 0) {
            for (double& val : row) {
                val /= max_val;
            }
        }
    }
}


// Function to standardize features (mean=0, std=1) across samples for each feature
void standardize_features(vector<vector<double>>& features) {
    size_t n_samples = features.size();
    if (n_samples == 0) return;
    size_t n_features = features[0].size();

    for (size_t j = 0; j < n_features; ++j) {
        // Compute mean
        double mean = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            mean += features[i][j];
        }
        mean /= n_samples;

        // Compute standard deviation
        double var = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            var += (features[i][j] - mean) * (features[i][j] - mean);
        }
        var /= n_samples;
        double stddev = sqrt(var + 1e-10); // Small epsilon to avoid division by zero

        // Standardize
        for (size_t i = 0; i < n_samples; ++i) {
            features[i][j] = (features[i][j] - mean) / stddev;
        }
    }
}

// 1D DCT Type-II for a single vector
vector<double> dct_reduce(const vector<double>& input, int keep_dim) {
    int N = input.size();
    vector<double> output(keep_dim, 0.0);
    for (int k = 0; k < keep_dim; ++k) {
        double sum = 0.0;
        for (int n = 0; n < N; ++n) {
            sum += input[n] * cos(M_PI * k * (2 * n + 1) / (2.0 * N));
        }
        output[k] = sum * sqrt(2.0 / N);
    }
    output[0] /= sqrt(2.0);  // Normalize first term
    return output;
}


void split_dataset(const vector<vector<double>>& features,
                   const vector<int>& labels,
                   double train_ratio, double val_ratio,
                   vector<vector<double>>& train_features,
                   vector<int>& train_labels,
                   vector<vector<double>>& val_features,
                   vector<int>& val_labels,
                   vector<vector<double>>& test_features,
                   vector<int>& test_labels) {
    size_t N = features.size();
    vector<size_t> indices(N);
    iota(indices.begin(), indices.end(), 0);

    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    size_t train_end = N * train_ratio;
    size_t val_end = train_end + N * val_ratio;

    for (size_t i = 0; i < N; ++i) {
        size_t idx = indices[i];
        if (i < train_end) {
            train_features.push_back(features[idx]);
            train_labels.push_back(labels[idx]);
        } else if (i < val_end) {
            val_features.push_back(features[idx]);
            val_labels.push_back(labels[idx]);
        } else {
            test_features.push_back(features[idx]);
            test_labels.push_back(labels[idx]);
        }
    }
}



// mapping output function
string map_pred(int pred){
    static string name;
    if(pred==0)  name="neutral"; 
    else if(pred==1)  name="relaxed";
    else if(pred==2)  name="stressed"; 
    return name;
}
