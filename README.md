# Simplest Artificial Intelligence

I'm curious about how neural networks work, so I recreated one by building the simplest AI model to predict prime numbers. The model uses a 4-layer neural network, and I designed a function to observe the training and testing process at each layer.

## Features

* Fully hand-written matrix and neural network logic
* Support for multiple dense and activation layers
* Simple sigmoid activation
* Basic mean squared error loss function
* Training and inference through CLI
* Visual forward pass with intermediate layer outputs
* Interactive UI with:

  * Training mode
  * Inference mode
  * Evaluation mode
  * Weight visualization and editing

## Architecture

The model consists of:

* Input layer: 10-bit binary representation of a number (0-1023)
* 4 hidden layers, each with 8 neurons
* Output layer: 1 neuron representing probability of primality

Layer structure:

```
Input (10) → Dense(8) → Sigmoid → Dense(8) → Sigmoid → Dense(8) → Sigmoid → Dense(8) → Sigmoid → Dense(1) → Sigmoid
```

## Usage

### Build & Run

```bash
cargo build
cargo run
```

### Menu Options

* **1. Train the model**: Specify number of samples (up to 1024) to train the model from scratch.
* **2. Use the model**: Enter a number between 0-1023 to see its prediction.
* **3. Evaluate performance**: Run the model over all samples and view accuracy with a confusion matrix.
* **4. View model weights**: Print all layer weight matrices.
* **5. Edit model weights**:

  * Manually change a specific weight
  * Randomize all weights
  * Reset all weights to zero
* **6. Quit**: Exit the program.

## File Structure

* `main.rs`: Entry point. Starts interactive CLI.
* `matrix.rs`: Matrix struct and operations.
* `layer.rs`: DenseLayer and ActivationLayer with trait-based abstraction.
* `model.rs`: Model struct for sequential layer management.
* `train.rs`: Training logic and loss computation.
* `dataset.rs`: Generates dataset of numbers \[0, 1023] with primality labels.
* `interface.rs`: Menu-based interactive CLI.
* `loss.rs`: Mean Squared Error loss.
* `activation.rs`: Sigmoid and (unused) ReLU functions.

## Dataset

The dataset contains all numbers from 0 to 1023. Each number is:

* Converted to a 10-bit binary vector

* Labeled `1` if prime, else `0`


---

Built with ❤️ by Ruangsak.