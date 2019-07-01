# MultiLayerPerceptron
A library developed to train deep neural nets written from scratch.

### Dependencies

 - NumPy
 - MatPlotLib (for plotting the metrics)
 
 
### Features

 - Two Optimizers: Adam Stochastic Optimization and Regular Gradient Descent
 - Common activation functions and their derivatives for backpropagation: RELU, ELU, Swish, Leaky RELU, Sigmoid
 - Xavier initialization
 - The final layer maps to probablities using Softmax, and the Cross Entropy is minimized
 - Option for Gradient Clipping
 - Arbitrary number of layers and neurons in each layer
 - Ability to save and reload the neural net for training
 - Utility functions, including:<br> 
                                  - `build_one_hot` that will build a binary encoded matrix from an enumerated classification array<br>
                                  - `map_to_unit` which will map an array to the unit interval [0, 1]<br>
                                  - `split_train_test` can seperate your data into a training and testing batch<br>
                                  

### Example

In `Digits.py`, we train a neural net with three hidden layers to classify 8x8 images of handwritten digits (0-9) using Scikit-learn's Digits dataset. To acquire this data, you can install the library with `pip install scikit-learn` or `conda install scikit-learn`.<br>

In this example, we will train a model for 100 epochs and save it, then evaluate the testing accuracy. Further on, we will reload the model from the directory we saved it in, then train it for 150 more epochs, and observe the improved testing accuracy.<br>

This is the output from the console after having successfully run the script:

```
Epoch   0   Cost   2.58357209223   Accuracy   0.102595797281
Epoch   50   Cost   1.75090436199   Accuracy   0.613720642769
Saving Model...
MLP trained in 2.3542688999999997 seconds
Test batch accuracy: 0.826815642458
Reloaded
Epoch   0   Cost   1.72772859501   Accuracy   0.620519159456
Epoch   50   Cost   0.69413787185   Accuracy   0.862793572311
Epoch   100   Cost   0.372743086693   Accuracy   0.914709517923
MLP trained in 2.549563400000004 seconds
Test batch improved accuracy: 0.91061452514

```

You'll notice two plots appear (one after the other), allowing us to visualize the training entropy-loss and accuracy over the training epochs of the first saved model. <br>

### Accuracy
![alt text](https://github.com/D-Thatcher/MultiLayerPerceptron/blob/master/accuracy_final.png)

### Entropy Loss
![alt text](https://github.com/D-Thatcher/MultiLayerPerceptron/blob/master/entropy_final.png)

### References

* Glorot, Xavier, and Yoshua Bengio. "Understanding the Difficulty of Training Deep Feedforward Neural Networks." 2010, proceedings.mlr.press/v9/glorot10a/glorot10a.pdf.<br>

* Kingma, Diederik P., and Jimmy Ba. “Adam: A Method for Stochastic Optimization.” ArXiv.org, 30 Jan. 2017, arxiv.org/abs/1412.6980.<br>

* Skalski, Piotr. “Let's Code a Neural Network in Plain NumPy.” Towards Data Science, Towards Data Science, 12 Oct. 2018, towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795.

