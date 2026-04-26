# Multi Layer Perceptron with Numpy

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.3+-013243?logo=numpy&logoColor=white)](https://numpy.org/)


A modular, from‑scratch implementation of Multi‑Layer Perceptrons using only NumPy. This project provides a flexible framework for building, training, and evaluating custom neural networks, demonstrated through two practical applications: **image compression with autoencoders** and **feature extraction for classification**.

The library implements core components including various activation functions, loss functions, weight initialization strategies, backpropagation with momentum, and batch training. All built without high‑level deep learning frameworks.

## 📌 Key Features

- **Fully connected layers** with configurable input/output dimensions
- **Activation functions**: ReLU, Leaky ReLU, Linear, Sigmoid, Tanh, Softmax
- **Loss functions**: MSE (regression) and Cross‑Entropy (classification)
- **Weight initialization**: zero, uniform, normal, Xavier (normal/uniform), Kaiming (normal/uniform)
- **Optimization**: SGD with configurable learning rate and momentum
- **Training modes**: stochastic, mini‑batch, or full‑batch gradient descent

All components are implemented as modular classes, making it easy to extend or swap components.

## 🏗️ Project Structure

The core neural network implementation is contained in `MLP.py`, which includes:
- `WeightInitializer` - handles different weight initialization schemes
- Activation classes - `ReLU`, `LeakyReLU`, `Linear`, `Sigmoid`, `Tanh`, `Softmax`
- Loss classes - `MSELoss`, `CrossEntropyLoss`
- `LinearLayer` - single fully‑connected layer with weight/bias storage
- `MLP` - main class supporting forward/backward passes, training, and evaluation
- Utility functions - data shuffling, loss plotting, PSNR calculation

Two demonstration notebooks (`autoencoder.ipynb`, `classifier.ipynb`) show end‑to‑end usage for autoencoding image compression and classification tasks.


## 🚀 Usage

### 1. Creating an MLP Model

Instantiate an `MLP` object by specifying layer sizes, activations per layer, loss function, learning rate, momentum, and weight initialization.

Example: autoencoder with one hidden layer

```python
from MLP import MLP, Sigmoid, Linear, ReLU, Softmax

layer_sizes = [64, 32, 64]           # input -> hidden -> output
activations = [Sigmoid(), Linear()]  # activation for hidden layer, output layer uses Linear for reconstruction

model = MLP(
    layer_sizes=layer_sizes,
    activations=activations,
    loss="mse",              # or "cross_entropy"
    lr=0.001,
    momentum=0.5,
    weight_type="xavier"
)
```

### 2. Training

Use the `train()` method with input data `X` and target data `Y`. Matrices should be shaped as `(n_samples, n_features)`.

```python
model.train(
    X_train, Y_train,
    epochs=50,
    batch_size=16,
    shuffle=True,
    test_model=True,
    X_test=X_val,
    Y_test=Y_val
)
```

Training automatically tracks loss per epoch and (optionally) test/validation loss.

### 3. Inference
```python
predictions = model.forward(X_new, grad=False)
```
### 4. Evaluating

Compute loss on a test set:

```python
test_loss = model.test(X_test, Y_test)
```
Plot training curves using the provided helper:

```python
from MLP import plot_metric_over_epoch

plot_metric_over_epoch(
    train_metric_list=model.train_loss_list,
    test_metric_list=model.test_loss_list,
    title="Training and Validation Loss"
)
```


## 🖼️ Use Case 1: Image Compression with Autoencoders

An autoencoder compresses images into a low‑dimensional latent representation and then reconstructs them. The encoder maps input blocks to a smaller hidden layer, and the decoder reconstructs the original block from that compressed code.

**Example configuration** (from the experiments):
- Input: 8×8 image blocks flattened to 64‑dimensional vectors
- Architecture: 64 → hidden_size → 64
- Hidden layer sizes tested: 4, 16, 32 neurons
- Activation: Sigmoid (encoder), Linear (decoder)
- Loss: MSE
- Training: 50 epochs, batch size 16

**Key findings**:
- Larger hidden layers preserve more detail, yielding higher PSNR and lower reconstruction loss.
- Smaller block sizes (e.g., 4×4) lead to better reconstruction but lower compression; 8×8 blocks offer a good trade‑off.
- Momentum (e.g., coefficient 0.5) stabilises training and can improve final reconstruction quality.

The `autoencder.ipynb` notebook contains the full pipeline: block extraction, model training, comparison of hidden layer sizes, block sizes, and momentum effects.

## 🔍 Use Case 2: Feature Extraction and Classification

Here an autoencoder is first trained to reconstruct `Fashion‑MNIST` images (28×28 → 784‑dimensional vectors). Its latent layer is then used as a feature extractor for a separate classifier.

**Stage 1 – Autoencoder training**:
- Architecture tested: 784 → 128 → 784 (single hidden layer)
- Activation: ReLU (faster convergence, reduces vanishing gradients)
- Loss: MSE
- Learning rate: 0.001 (best trade‑off between stability and speed)
- Batch size: 8 or 16 (smoother than pure SGD, faster than full‑batch)

**Stage 2 – Classifier**:
- Input: latent features from the trained autoencoder (128‑dimensional)
- Architecture: 128 → hidden → 10 (Softmax output)
- Loss: Cross‑Entropy
- Hidden layer sizes tested: 32 and 64 neurons (ReLU activation)

**Results**:
- Autoencoder with 128‑unit bottleneck achieved strong reconstruction (PSNR ~17 dB).
- Classifier with 64 hidden neurons reached 85.0% accuracy on Fashion‑MNIST test set; 32 neurons gave 84.8%.
- ReLU activation in the autoencoder led to faster and more stable convergence compared to Sigmoid.

The `classifier.ipynb` notebook walks through data loading (IDX format), train/validation splitting, hyperparameter tuning (hidden size, learning rate, batch size), and final classification evaluation.


## ⚙️ Customisation Examples

**Adding a new activation function** – subclass the base pattern:

```python
class MyActivation:
    def activate(self, x):
        return ...
    def derivative(self, x):
        return ...
```

**Changing weight initialisation** – pass any supported type to the `weight_type` parameter:
"zero", "uniform", "normal", "xavier", "xavier-uniform", "kaiming", "kaiming-uniform"

**Using classification mode** – set `loss="cross_entropy"` and ensure the output layer uses `Softmax` activation.


## 📄 License

This project is licensed under Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
