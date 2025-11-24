import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time

GLOBAL_SEED = 24

rng = np.random.default_rng(GLOBAL_SEED)

def reset_seed(seed = GLOBAL_SEED):
    global rng
    rng = np.random.default_rng(seed)
    
def shuffle_data(arr1, arr2):
    assert len(arr1) == len(arr2), "Arrays must have the same length"
    permutation = rng.permutation(len(arr1))
    reset_seed()
    return arr1[permutation], arr2[permutation]


DEFAULT_WEIGHT_TYPE = 'xavier'
weight_methods = {'zero', 'uniform', 'normal', 'xavier', 'xavier-uniform',
                  'kaiming', 'kaiming-uniform'}

class WeightInitializer:       
    def __init__(self, weight_type):
        if weight_type in weight_methods:
            self.weight_type = weight_type
        else:
            print(f"Warning: Unknown weight initialization type '{weight_type}'. Using default '{DEFAULT_WEIGHT_TYPE}'.")
            self.weight_type = DEFAULT_WEIGHT_TYPE

    def zeros(self, shape):
        return np.zeros(shape)

    def uniform(self, shape, low=-0.5, high=0.5):
        return rng.uniform(low, high, shape)

    def normal(self, shape, mean=0.0, std=0.5):
        return rng.normal(mean, std, shape)

    def xavier(self, shape, fan_in, fan_out):
        std = np.sqrt(2 / (fan_in + fan_out))
        return rng.normal(0, std, shape)
    
    def xavier_uniform(self, shape, fan_in, fan_out):
        limit = np.sqrt(6 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, shape)

    def kaiming(self, shape, fan_in):
        std = np.sqrt(2 / fan_in)
        return rng.normal(0, std, size=shape)

    def kaiming_uniform(self, shape, fan_in):
        limit = np.sqrt(6 / fan_in)
        return rng.uniform(-limit, limit, shape)
    
        
    def initialize_weights(self, shape, n_in, n_out):        
        # Map weight types to their corresponding methods
        weight_methods = {
            'zero': self.zeros,
            'uniform': self.uniform,
            'normal': self.normal,
            'xavier': lambda s: self.xavier(s, n_in, n_out),
            'xavier-uniform': lambda s: self.xavier_uniform(s, n_in, n_out),
            'kaiming': lambda s: self.kaiming(s, n_in),
            'kaiming-uniform': lambda s: self.kaiming_uniform(s, n_in),
        }

        W = weight_methods.get(self.weight_type)(shape)
        reset_seed()
        return W
    
    
# -------------------------
# Activation functions
# -------------------------
class ReLU:
    def activate(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return (x > 0).astype(float)
    
class LeakyReLU:
    def __init__(self, alpha = 0.01):
        self.alpha = alpha
        
    def activate(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)

class Linear:
    def activate(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)
    
class Sigmoid:
    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        s = self.activate(x)
        return s * (1 - s)
    
class Tanh:
    def activate(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.tanh(x)**2

class Softmax:
    def activate(self, x):
        exp_x = np.exp(x - np.max(x))  # stability trick
        return exp_x / np.sum(exp_x)
    
    def derivative(self, x):
        raise ValueError("Softmax derivative should only be used in the output layer (with Cross Entropy Loss).")


# -------------------------
# Loss functions
# -------------------------
class MSELoss:  # Do not use with Softmax activation!!
    def loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def output_delta(self, y_pred, y_true, z, activation):
        return (y_pred - y_true) * activation.derivative(z)  


class CrossEntropyLoss:  # Only use with Softmax activation!!
    def loss(self, y_pred, y_true):
        epsilon = 1e-15  # avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))  # np.mean is used if we're updating batch by batch

    def output_delta(self, y_pred, y_true, z, activation):  # activation should be Softmax
        if isinstance(activation, Softmax) == False:
            raise ValueError("Cross Entropy Loss should be used with Softmax activation in the output layer.")
        
         # Softmax + Cross Entropy has a simplified formula for delta
        return y_pred - y_true  # y_pred is post-activation function!


# -------------------------
# Linear Layer
# -------------------------
class LinearLayer:
    def __init__(self, n_in, n_out, activation, initializer):
        self.shape = (n_in, n_out)
        self.activation = activation
        
        # self.W = np.random.randn(n_in, n_out) * np.sqrt(2/n_in)
        self.W = initializer.initialize_weights(self.shape, n_in, n_out)
        self.b = np.zeros((n_out,))
        
        # store previous weight updates (difference between current weights and the previous weights). used for momentum
        self.previous_weight_update = np.zeros_like(self.W)
        self.previous_bias_update = np.zeros_like(self.b)


    def forward(self, x):
        return x @ self.W + self.b

    def __str__(self):
        return f"Linear Block {self.shape},   Activation: {self.activation.__class__.__name__}\n"




# -------------------------
# MLP with Delta-Based Backprop
# -------------------------
class MLP:
    def __init__(self, layer_sizes, activations, existing_layers=[], loss="mse", lr=0.01, momentum=0.0, weight_type="xavier"):
        self.layers = existing_layers  # for transfer learning or continuing training from existing layers
        
        self.lr = lr
        self.momentum = momentum
        self.loss_fn = CrossEntropyLoss() if loss == "cross_entropy" else MSELoss()

        self.initializer = WeightInitializer(weight_type.strip().lower())
        
        for i in range(len(layer_sizes) - 1):
            activation = activations[i] if i < len(activations) else Linear()
            self.layers.append(
                LinearLayer(layer_sizes[i], layer_sizes[i + 1], activation, self.initializer)
            )
            
        self.train_loss_list = []
        self.test_loss_list = []
    
                
    def forward(self, x, grad=False):
        out = x
        if grad:
            self.z = []  # pre-activation outputs
            self.a = [x]  # post-activation outputs, starting with input

        for layer in self.layers:
            z = layer.forward(out)
            out = layer.activation.activate(z)
            
            if grad:
                self.z.append(z)
                self.a.append(out)
        return out


    def backward(self, batch_X, batch_Y):
        # gradient accumulators - these will store the sum of the gradients for each weight - dLoss/dW
        dW_acc = [np.zeros_like(layer.W) for layer in self.layers]
        db_acc = [np.zeros_like(layer.b) for layer in self.layers]

        batch_size = len(batch_X)
        batch_loss = 0

        # calculate gradients (dW) for each sample in batch and accumulate them
        for x, y in zip(batch_X, batch_Y):
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)

            y_pred = self.forward(x, grad=True)
            batch_loss += self.loss_fn.loss(y_pred, y)

            deltas = [None] * len(self.layers)
            deltas[-1] = self.loss_fn.output_delta(
            y_pred, y, self.z[-1], self.layers[-1].activation
            )

            # find hidden layer deltas
            for i in reversed(range(len(self.layers)-1)):
                W_next = self.layers[i+1].W
                delta_next = deltas[i+1]
                z = self.z[i]
                deltas[i] = (delta_next @ W_next.T) * \
                            self.layers[i].activation.derivative(z)

            # store (accumulate) gradients for each weight
            for i, (layer, delta) in enumerate(zip(self.layers, deltas)):
                x_i = self.a[i]
                dW_acc[i] += x_i.reshape(-1,1) @ delta.reshape(1,-1)  # x * delta
                db_acc[i] += delta.reshape(-1)

        # Average gradients for every weight - divide accumulated gradients by batch size
        dW_acc = [dW / batch_size for dW in dW_acc]
        db_acc = [db / batch_size for db in db_acc]

        # Update - if momentum = 0, this is just standard gradient descent
        for i, layer in enumerate(self.layers):
            weight_update = - (self.lr * dW_acc[i] + self.momentum * layer.previous_weight_update)
            bias_update   = - (self.lr * db_acc[i] + self.momentum * layer.previous_bias_update)
            
            layer.W = layer.W + weight_update
            layer.b = layer.b + bias_update
            
            layer.previous_weight_update = weight_update
            layer.previous_bias_update = bias_update
                  
        return batch_loss


    def train(self, X, Y, epochs=10, print_interval=2, batch_size=1, shuffle=False,
              test_model=False, X_test=None, Y_test=None):
        
        start_time = time.time()  # Start timing the training process
        N = len(X)
        
        if shuffle:
            X, Y = shuffle_data(X, Y)
        
        for epoch in tqdm(range(epochs), desc="Training", unit='Epoch'):  # Loop over epochs
            epoch_loss = 0
            
            for start in range(0, N, batch_size):  # Loop over batches
                end = start + batch_size
                X_batch = X[start:end]
                Y_batch = Y[start:end]                   
                epoch_loss += self.backward(X_batch, Y_batch)  # find gradients after each batch and update weights

            train_loss = epoch_loss / N
            self.train_loss_list.append(train_loss)
            
            if test_model:
                test_loss = self.test(X_test, Y_test)
                self.test_loss_list.append(test_loss)
            
            if (epoch + 1) % print_interval == 0:
                print(f"Epoch {epoch+1:4} / {epochs:4},   Train Loss: {train_loss:8f}", end='')
                if test_model:
                    print(f",   Test Loss: {test_loss:8f}", end='')
                print()
    
        end_time = time.time()  # End timing the training process
        total_time = end_time - start_time
        print(f"Training completed in {total_time:.2f} seconds.")


    def test(self, X, Y):
            loss_sum = 0
            for x, y in zip(X, Y):
                x = x.reshape(1, -1)
                y = y.reshape(1, -1)

                y_pred = self.forward(x, grad=False)
                loss_sum += self.loss_fn.loss(y_pred, y)
            return loss_sum/len(X)
        


    def __str__(self):
        network_architecture = ""
        for i, layer in enumerate(self.layers):
            network_architecture += f"  Layer {i+1:2}: {layer}"
            
        return (
            f"Multi-Layer Perceptron Details:\n"
            f"Input Size: {self.layers[0].shape[0]}\n"
            f"Output Size: {self.layers[-1].shape[-1]}\n"
            f"Architecture:\n{network_architecture}"
            f"Learning Rate: {self.lr}\n"
            f"Momentum: {self.momentum}\n"
            f"Weight Initialization Type: {self.initializer.weight_type}\n"
        )


def plot_metric_over_epoch(train_metric_list, test_metric_list=None, title="", y_label="Mean Loss", figure_size=(8, 5)):
    plt.figure(figsize=figure_size)

    epochs = np.arange(1, len(train_metric_list) + 1)
    plt.plot(epochs, train_metric_list, label='Train', linewidth=2)
    if test_metric_list is not None:
        plt.plot(epochs, test_metric_list, color='salmon', label='Test', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    # plt.tight_layout()
    plt.show()


def plot_train_metrics(metric_lists, labels=None, title="", y_label="Mean Loss", figure_size=(8, 5)):

    plt.figure(figsize=figure_size)
    
    num_curves = len(metric_lists)
    colors = plt.cm.tab10(np.linspace(0, 1, num_curves))  # distinct colors

    for i, error_rates in enumerate(metric_lists):
        epochs = np.arange(1, len(error_rates) + 1)
        label = labels[i] if labels and i < len(labels) else f"Run {i+1}"
        plt.plot(epochs, error_rates, color=colors[i], label=label)

    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    # plt.tight_layout()
    plt.show()



def calculate_psnr(image1, image2):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two grayscale images.

    Parameters:
        image1 (numpy.ndarray): First image with values in [0, 1].
        image2 (numpy.ndarray): Second image with values in [0, 1].

    Returns:
        float: PSNR value in decibels (dB).
    """
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if images are identical
    
    max_pixel_value = 1.0  # Since the images are in [0, 1]
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr