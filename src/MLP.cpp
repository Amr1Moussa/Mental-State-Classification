#include "C:\Users\laphouse\Projects\Brain_waves\includes\MLP.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <ctime>
#include <iostream>
#include <stdexcept>

using namespace std;

MLP::MLP(const vector<int>& layers, double learning_rate)
    : layers_(layers), learning_rate_(learning_rate) {
    if (layers.size() < 2) {
        throw invalid_argument("MLP must have at least 2 layers (input and output)");
    }
    srand(static_cast<unsigned>(time(nullptr)));

    int n_layers = layers_.size();
    weights_.resize(n_layers - 1);
    biases_.resize(n_layers - 1);
    weight_grads_accum_.resize(n_layers - 1);
    bias_grads_accum_.resize(n_layers - 1);

    for (int i = 0; i < n_layers - 1; ++i) {
        int rows = layers_[i + 1];
        int cols = layers_[i];

        weights_[i].resize(rows, vector<double>(cols));
        biases_[i].resize(rows);
        weight_grads_accum_[i].resize(rows, vector<double>(cols, 0.0));
        bias_grads_accum_[i].resize(rows, 0.0);

        // Xavier Initialization
        double limit = sqrt(6.0 / (rows + cols));
        for (int r = 0; r < rows; ++r) {
            biases_[i][r] = 0.0;
            for (int c = 0; c < cols; ++c) {
                weights_[i][r][c] = ((double)rand() / RAND_MAX * 2 - 1) * limit;
            }
        }
    }
}

double MLP::sigmoid(double x) {
    // Clip input to prevent overflow in exp
    x = max(-500.0, min(500.0, x));
    return 1.0 / (1.0 + exp(-x));
}

double MLP::sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

vector<double> MLP::softmax(const vector<double>& z) {
    vector<double> res(z.size());
    double max_z = *max_element(z.begin(), z.end());
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        res[i] = exp(z[i] - max_z);
        sum += res[i];
    }
    if (sum == 0.0) {
        throw runtime_error("Softmax sum is zero");
    }
    for (size_t i = 0; i < z.size(); ++i) {
        res[i] /= sum;
    }
    return res;
}

void MLP::forward(const vector<double>& input) {
    if (input.size() != layers_[0]) {
        throw invalid_argument("Input size does not match input layer");
    }

    activations_.clear();
    zs_.clear();
    activations_.push_back(input);

    for (size_t i = 0; i < weights_.size(); ++i) {
        const vector<double>& a_prev = activations_.back();
        int layer_size = layers_[i + 1];
        vector<double> z(layer_size, 0.0);

        for (int j = 0; j < layer_size; ++j) {
            z[j] = biases_[i][j];
            for (size_t k = 0; k < a_prev.size(); ++k) {
                z[j] += weights_[i][j][k] * a_prev[k];
            }
        }

        zs_.push_back(z);
        vector<double> a(layer_size);
        if (i == weights_.size() - 1) {
            a = softmax(z);
        } else {
            for (int j = 0; j < layer_size; ++j) {
                a[j] = sigmoid(z[j]);
            }
        }
        activations_.push_back(a);
    }
}

void MLP::backward(const vector<double>& y_true) {
    if (y_true.size() != layers_.back()) {
        throw invalid_argument("True label size does not match output layer");
    }

    int n_layers = layers_.size();
    vector<vector<double>> deltas(n_layers - 1);

    // Output layer: softmax + cross-entropy
    const vector<double>& a_last = activations_.back();
    deltas[n_layers - 2].resize(layers_.back());
    for (int j = 0; j < layers_.back(); ++j) {
        deltas[n_layers - 2][j] = a_last[j] - y_true[j];
    }

    // Hidden layers
    for (int l = n_layers - 3; l >= 0; --l) {
        int size = layers_[l + 1];
        deltas[l].resize(size);
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < layers_[l + 2]; ++k) {
                sum += weights_[l + 1][k][j] * deltas[l + 1][k];
            }
            deltas[l][j] = sum * sigmoid_derivative(zs_[l][j]);
        }
    }

    // Accumulate gradients
    for (int l = 0; l < n_layers - 1; ++l) {
        int rows = layers_[l + 1];
        int cols = layers_[l];
        for (int r = 0; r < rows; ++r) {
            bias_grads_accum_[l][r] += deltas[l][r];
            for (int c = 0; c < cols; ++c) {
                weight_grads_accum_[l][r][c] += deltas[l][r] * activations_[l][c];
            }
        }
    }
}

void MLP::update_weights(int batch_size) {
    for (size_t l = 0; l < weights_.size(); ++l) {
        for (size_t r = 0; r < weights_[l].size(); ++r) {
            biases_[l][r] -= (learning_rate_ / batch_size) * bias_grads_accum_[l][r];
            bias_grads_accum_[l][r] = 0.0;
            for (size_t c = 0; c < weights_[l][r].size(); ++c) {
                weights_[l][r][c] -= (learning_rate_ / batch_size) * weight_grads_accum_[l][r][c];
                weight_grads_accum_[l][r][c] = 0.0;
            }
        }
    }
}

double MLP::compute_loss(const vector<double>& y_pred, const vector<double>& y_true) {
    double loss = 0.0;
    for (size_t i = 0; i < y_pred.size(); ++i) {
        loss -= y_true[i] * log(max(y_pred[i], 1e-15)); // Prevent log(0)
    }
    return loss;
}



void MLP::train(const vector<vector<double>>& X,
                const vector<int>& y, int epochs, int batch_size) {
    if (X.size() != y.size()) {
        throw invalid_argument("Input and label sizes do not match");
    }
    if (batch_size <= 0) {
        throw invalid_argument("Batch size must be positive");
    }

    // Initialize gradient accumulators
    int n_layers = layers_.size();
    weight_grads_accum_.resize(n_layers - 1);
    bias_grads_accum_.resize(n_layers - 1);
    for (int l = 0; l < n_layers - 1; ++l) {
        weight_grads_accum_[l].assign(layers_[l + 1], vector<double>(layers_[l], 0.0));
        bias_grads_accum_[l].assign(layers_[l + 1], 0.0);
    }

    // Initialize indices
    vector<size_t> indices(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        indices[i] = i;
    }

    // Initialize random number generator
    random_device rd;
    mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle indices using shuffle
        shuffle(indices.begin(), indices.end(), g);

        double total_loss = 0.0;
        for (size_t i = 0; i < X.size(); i += batch_size) {
            int current_batch_size = min(batch_size, static_cast<int>(X.size() - i));

            // Clear gradients for the batch
            for (auto& wg : weight_grads_accum_) {
                for (auto& row : wg) {
                    fill(row.begin(), row.end(), 0.0);
                }
            }
            for (auto& bg : bias_grads_accum_) {
                fill(bg.begin(), bg.end(), 0.0);
            }

            // Process batch
            for (int j = 0; j < current_batch_size; ++j) {
                size_t idx = indices[i + j];
                forward(X[idx]);

                vector<double> y_onehot(layers_.back(), 0.0);
                if (y[idx] >= layers_.back()) {
                    throw invalid_argument("Label out of range");
                }
                y_onehot[y[idx]] = 1.0;

                backward(y_onehot);
                total_loss += compute_loss(activations_.back(), y_onehot);
            }

            update_weights(current_batch_size);
        }

        if ((epoch + 1) % 10 == 0) {
            double acc = evaluate(X, y);
            cout << "Epoch " << epoch + 1
                      << " | Loss: " << total_loss / X.size()
                      << " | Train Accuracy: " << acc << endl;
        }
    }
}

double MLP::evaluate(const vector<vector<double>>& X,
                     const vector<int>& y) {
    if (X.size() != y.size()) {
        throw invalid_argument("Input and label sizes do not match");
    }

    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        int pred = predict(X[i]);
        if (pred == y[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / X.size();
}

int MLP::predict(const vector<double>& x) {
    forward(x);
    const vector<double>& output = activations_.back();
    return distance(output.begin(), max_element(output.begin(), output.end()));
}