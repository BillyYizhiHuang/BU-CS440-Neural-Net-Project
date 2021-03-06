"""
Created on Wed Feb 10 21:56:02 2016

@author:Yizhi Huang
        Yuezhou
        Annalisa Chen
        Yingqiao Xiong
"""

import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    """
    This class implements a simple 3 layer neural network.
    """
    
    def __init__(self, input_dim, output_dim, epsilon):
        """
        Initializes the parameters of the neural network to random values
        """
        
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))
        self.epsilon = epsilon
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total loss on the dataset
        """
        
        num_samples = len(X)
        # Do Forward propagation to calculate our predictions
        z = X.dot(self.W) + self.b
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        # Calculate the cross-entropy loss
        cross_ent_err = -np.log(softmax_scores[range(num_samples), y])
        data_loss = np.sum(cross_ent_err)
        return 1./num_samples * data_loss

    
    #--------------------------------------------------------------------------
 
    def predict(self,x):
        """
        Makes a prediction based on current model parameters
        """
        # Do Forward Propagation
        z = x.dot(self.W) + self.b
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return np.argmax(softmax_scores, axis=1)
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y,num_epochs):
        """
        Learns model parameters to fit the data
        """
        num_samples = len(X)
        for i in range(0,num_epochs):
            deltab = 0
            deltaw = 0
            for j in range(0,num_samples):
                z = X[j].dot(self.W) + self.b
                exp_z = np.exp(z)
                output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                if y[j] == 0:
                    ground_truth = np.array([1,0])
                else:
                    ground_truth = np.array([0,1])
                beta = output - ground_truth
                deltab = deltab + beta
                deltaw = deltaw + np.dot(X[j].reshape(2,1), beta)
            self.W = self.W - self.epsilon * deltaw / num_samples
            self.b = self.b - self.epsilon * deltab / num_samples 

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
        
def plot_decision_boundary(pred_func):
    """
    Helper function to print the decision boundary given by model
    """
    # Set min and max values
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#Train Neural Network on
linear = True

#A. linearly separable data
if linear:
    #load data
    X = np.genfromtxt('Desktop/Lab4_Soln/DATA/ToyLinearX.csv', delimiter=',')
    y = np.genfromtxt('Desktop/Lab4_Soln/DATA/ToyLineary.csv', delimiter=',')
    y = y.astype(int)
    #plot data
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
#B. Non-linearly separable data
else:
    #load data
    X = np.genfromtxt('Desktop/Lab4_Soln/DATA/ToyMoonX.csv', delimiter=',')
    y = np.genfromtxt('Desktop/Lab4_Soln/DATA/ToyMoony.csv', delimiter=',')
    y = y.astype(int)
    #plot data
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

input_dim = 2 # input layer dimensionality
output_dim = 2 # output layer dimensionality

# Gradient descent parameters 
epsilon = 0.01
num_epochs = 5000

# Fit model

NN = NeuralNet(input_dim, output_dim, epsilon)
NN.fit(X,y,num_epochs)


# Plot the decision boundary
plot_decision_boundary(lambda x: NN.predict(x))
plt.title("Neural Net Decision Boundary")         
