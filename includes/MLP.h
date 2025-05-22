#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>
#include <chrono>  // For timing

using namespace std;

class MLP {
public:
    // Constructor: layers specify the size of each layer (input, hidden, output)
    MLP(const vector<int>& layers, double learning_rate);

    // Original train function (backward compatibility)
    void train(const vector<vector<double>>& X,
               const vector<int>& y, int epochs, int batch_size = 32);

    // Enhanced train function with validation data for learning curves
    void train(const vector<vector<double>>& X_train,
               const vector<int>& y_train, 
               const vector<vector<double>>& X_val,
               const vector<int>& y_val,
               int epochs, int batch_size = 32);

    // Evaluate accuracy on a dataset
    double evaluate(const vector<vector<double>>& X,
                    const vector<int>& y);

    // Predict the class for a single input
    int predict(const vector<double>& x);

    // Learning curve analysis functions
    void save_learning_curves(const string& filename = "learning_curves.csv");
    void get_learning_curves(vector<int>& epochs, 
                            vector<double>& train_losses,
                            vector<double>& train_accuracies, 
                            vector<double>& val_accuracies);
    void print_learning_summary();
    
    // Clear learning history (useful for multiple training sessions)
    void clear_learning_history();

private:
    vector<int> layers_;                    // Layer sizes
    double learning_rate_;                  // Learning rate for gradient descent
    vector<vector<vector<double>>> weights_; // Weight matrices
    vector<vector<double>> biases_;         // Bias vectors
    vector<vector<double>> activations_;    // Activations per layer
    vector<vector<double>> zs_;             // Pre-activation values
    vector<vector<vector<double>>> weight_grads_accum_; // Accumulated weight gradients
    vector<vector<double>> bias_grads_accum_; // Accumulated bias gradients

    // Learning curve data storage
    vector<double> training_losses_;
    vector<double> training_accuracies_;
    vector<double> validation_accuracies_;
    vector<int> epoch_numbers_;

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