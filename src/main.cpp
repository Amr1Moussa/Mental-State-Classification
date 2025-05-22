#include "C:\Users\laphouse\Projects\Brain_waves\includes\utils.h"
#include "C:\Users\laphouse\Projects\Brain_waves\includes\feature_extraction.h"
#include "C:\Users\laphouse\Projects\Brain_waves\includes\MLP.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <string>
#include <sstream> // Added for istringstream
#include <fstream> // Added for file output

using namespace std;

// mapping output function
string map_pred(int pred){
    static string name;
    if(pred==0)  name="neutral"; 
    else if(pred==1)  name="relaxed";
    else if(pred==2)  name="stressed"; 
    return name;
}

int main() {
    vector<vector<double>> features;
    vector<int> labels;

    // Load data
    if (!load_csv("data/mental-state.csv", features, labels, "Label")) {
        cerr << "Failed to load CSV.\n";
        return 1;
    }

    cout << "Loaded " << features.size() << " rows.\n";
    cout << "Feature dim: " << features[0].size() << "\n";

    // Check class distribution
    map<int, int> class_counts;
    for (int label : labels) {
        class_counts[label]++;
    }
    cout << "Class distribution:\n";
    for (const auto& [label, count] : class_counts) {
        cout << "Class " << label << ": " << count << " samples ("
             << 100.0 * count / labels.size() << "%)\n";
    }

    // Store raw features separately
    vector<vector<double>> raw_features = features;

    // Extract statistical features
    for (auto& sample : features) {
        double m = mean(sample);
        double var = variance(sample, m);
        double stddev = sqrt(var);
        double sk = skewness(sample, m, stddev);
        double kurt = kurtosis(sample, m, stddev);
        double ent = entropy(sample);

        sample.push_back(m);
        sample.push_back(var);
        sample.push_back(sk);
        sample.push_back(kurt);
        sample.push_back(ent);
    }
    cout << "Extracted and appended statistical features.\n";

    // Standardize features (feature-wise)
    standardize_features(features);
    cout << "Data standardized.\n";

    // Apply DCT reduction
    int dct_dim = 100; // Increased to retain more information
    for (auto& sample : features) {
        sample = dct_reduce(sample, dct_dim);
    }
    cout << "Applied DCT reduction to " << dct_dim << " features.\n";

    // Initialize MLP
    vector<int> layers = {dct_dim, 128, 64, 32, 3}; 
    double learning_rate = 0.001;              
    int batch_size = 32;                       
    MLP model(layers, learning_rate);

    // Split dataset
    vector<vector<double>> train_features, val_features, test_features;
    vector<int> train_labels, val_labels, test_labels;
    split_dataset(features, labels, 0.7, 0.15,
                  train_features, train_labels,
                  val_features, val_labels,
                  test_features, test_labels);

    cout << "Dataset split:\n";
    cout << "Train: " << train_features.size() << " samples\n";
    cout << "Validation: " << val_features.size() << " samples\n";
    cout << "Test: " << test_features.size() << " samples\n";

    // Save test samples to text file
    ofstream out_file("test_samples.txt");
    if (!out_file.is_open()) {
        cerr << "Failed to open test_samples.txt for writing.\n";
        return 1;
    }
    for (size_t i = 0; i < test_features.size(); ++i) {
        for (double feature : test_features[i]) {
            out_file << feature << " ";
        }
        out_file << test_labels[i] << "\n";
    }
    out_file.close();
    cout << "Saved " << test_features.size() << " test samples to test_samples.txt\n";

    // Interactive menu
    string choice;
    random_device rd;
    mt19937 gen(rd());
    bool model_trained = false;

    while (true) {
        cout << "\nMenu:\n";
        cout << "1. Train the model\n";
        cout << "2. Evaluate on test set\n";
        cout << "3. Test on a random test sample\n";
        cout << "4. Test on user input\n";
        cout << "5. Exit\n";
        cout << "Enter choice (1-5): ";
        getline(cin, choice);

        if (choice == "1") {
            // Option 1: Train
            int epochs = 200;
            int patience = 30;
            double best_val_acc = 0.0;
            int epochs_no_improve = 0;
            cout << "Training MLP...\n";
            for (int epoch = 0; epoch < epochs; ++epoch) {
                model.train(train_features, train_labels, 1, batch_size); // Train 1 epoch
                double val_acc = model.evaluate(val_features, val_labels);
                if((epoch+1)%5==0){
                cout << "Epoch " << epoch + 1 << " | Validation Accuracy: " << val_acc << "\n";}

                // Early stopping
                if (val_acc > best_val_acc) {
                    best_val_acc = val_acc;
                    epochs_no_improve = 0;
                } else {
                    epochs_no_improve++;
                    if (epochs_no_improve >= patience) {
                        cout << "Early stopping at epoch " << epoch + 1 << "\n";
                        break;
                    }
                }
            }
            model_trained = true;
            cout << "Training completed.\n";

        } else if (choice == "2") {
            // Option 2: Evaluate
            if (!model_trained) {
                cout << "Model not trained yet. Please train first.\n";
                continue;
            }
            double test_accuracy = model.evaluate(test_features, test_labels);
            cout << "Test accuracy: " << test_accuracy << "\n";

        } else if (choice == "3") {
            // Option 3: Test on random sample
            if (!model_trained) {
                cout << "Model not trained yet. Please train first.\n";
                continue;
            }
            if (test_features.empty()) {
                cout << "Test set is empty.\n";
                continue;
            }
            uniform_int_distribution<> dis(0, test_features.size() - 1);
            int idx = dis(gen);
            int pred = model.predict(test_features[idx]);
            cout << "Random test sample " << idx << ":\n";
            cout << "True label = " << test_labels[idx] << ", Predicted = " << map_pred(pred) << "\n";

        } else if (choice == "4") {
            // Option 4: Test on student input
            if (!model_trained) {
                cout << "Model not trained yet. Please train first.\n";
                continue;
            }
            cout << "Enter " << dct_dim << " feature values (space-separated):\n";
            vector<double> input(dct_dim);
            string line;
            getline(cin, line);
            if (line.empty()) {
                cout << "Invalid input: Empty input provided.\n";
                continue;
            }
            istringstream iss(line);
            size_t i = 0;
            double val;
            while (i < dct_dim && iss >> val) {
                input[i++] = val;
            }
            if (i != dct_dim || iss.fail()) {
                cout << "Invalid input: Expected " << dct_dim << " numeric values, got " << i << "\n";
                continue;
            }
            int pred = model.predict(input);
            cout << "\nPredicted label = " << map_pred(pred) << "\n";

        } else if (choice == "5") {
            // Exit
            cout << "Exiting.\n";
            break;
        } else {
            cout << "Invalid choice. Please enter 1-5.\n";
        }
    }

    return 0;
}
