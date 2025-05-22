# Mental-State-Classification

A C++ implementation for classifying mental states using EEG datasets and a Multilayer Perceptron (MLP) neural network.

## Overview

This project provides tools for EEG data analysis and mental state classification. It leverages feature extraction techniques and a custom MLP implementation to classify different mental states from brainwave data.

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/Mental-State-Classification.git
cd Mental-State-Classification
```

## Building the Project

### Linux/MacOS (Bash)
```bash
g++ src/feature_extraction.cpp src/utils.cpp src/main.cpp src/MLP.cpp -o eeg_reader
```

### Windows (PowerShell/Command Prompt)
```powershell
g++ src/feature_extraction.cpp src/utils.cpp src/main.cpp src/MLP.cpp -o eeg_reader
```

## Usage

Run the compiled executable:
```bash
./eeg_reader
```

## Features

The application provides a menu-based interface with the following options:

1. **Train the model**: Uses 70% of the dataset for training the MLP
2. **Evaluate on test set**: Tests performance on the remaining 30% of data
3. **Test on a random sample**: Classifies a randomly selected test sample
4. **Test on user input**: Classifies data from a sample provided in test_samples.txt (without labels)
5. **Exit**: Terminate the application

## Dataset

The project works with EEG datasets for mental state classification. Ensure your data is properly formatted according to the expected input format.

## Project Structure

- `src/feature_extraction.cpp`: Implementation of EEG feature extraction algorithms
- `src/utils.cpp`: Utility functions for data handling and processing
- `src/main.cpp`: Main application logic and user interface
- `src/MLP.cpp`: Multilayer Perceptron neural network implementation

## Requirements

- C++ compiler (GCC recommended)
- Basic knowledge of EEG data processing


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
