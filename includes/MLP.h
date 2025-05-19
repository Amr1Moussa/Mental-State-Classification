#ifndef MLP_H
#define MLP_H

#include <vector>

using namespace std;

class MLP {
public:
    // Constructor: layers specify the size of each layer (input, hidden, output)
    MLP(const vector<int>& layers, double learning_rate);

    // Train the model on input features X and labels y
    void train(const vector<vector<double>>& X,
               const vector<int>& y, int epochs, int batch_size = 32);

    // Evaluate accuracy on a dataset
    double evaluate(const vector<vector<double>>& X,
                    const vector<int>& y);

    // Predict the class for a single input
    int predict(const vector<double>& x);

private:
    vector<int> layers_;                    // Layer sizes
    double learning_rate_;                       // Learning rate for gradient descent
    vector<vector<vector<double>>> weights_;  // Weight matrices
    vector<vector<double>> biases_;    // Bias vectors
    vector<vector<double>> activations_;  // Activations per layer
    vector<vector<double>> zs_;        // Pre-activation values
    vector<vector<vector<double>>> weight_grads_accum_;  // Accumulated weight gradients
    vector<vector<double>> bias_grads_accum_;  // Accumulated bias gradients

    // Activation functions
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    vector<double> softmax(const vector<double>& z);

    // Forward and backward passes
    void forward(const vector<double>& input);
    void backward(const vector<double>& y_true);
    void update_weights(int batch_size);

    // Compute cross-entropy loss
    double compute_loss(const vector<double>& y_pred, const vector<double>& y_true);
};

#endif // MLP_H