"""
Created on Wed Feb 10 21:56:02 2016

@author: Yizhi Huang
        Yue Zhou
        Annalisa Chen
        Yingqiao Xiong
"""

import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    """
    This class implements a simple 3 layer neural network.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon):
        """
        Initializes the parameters of the neural network to random values
        """
        
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.W2 = np.random.randn(hidden_dim,output_dim) / np.sqrt(hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, output_dim))
        self.epsilon = epsilon
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total loss on the dataset
        """
        
        num_samples = len(X)
        # Do Forward propagation to calculate our predictions
        z1 = X.dot(self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2
        exp_z = np.exp(z2)
        a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        softmax_scores = a2
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
        z1 = x.dot(self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2
        exp_z = np.exp(z2)
        a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        softmax_scores = a2
        return np.argmax(softmax_scores, axis=1)
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y,num_epochs):
        """
        Learns model parameters to fit the data
        """
        num_samples = len(X)
        for i in range(0,1):
            deltab2 = 0
            deltaw2 = 0
            deltab1 = 0
            deltaw1 = 0
            for j in range(0,num_samples):   
                z1 = X[j].dot(self.W1) + self.b1
                a1 = 1 / (1 + np.exp(-z1)) 
                z2 = a1.dot(self.W2) + self.b2
                exp_z = np.exp(z2)
                output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                if y[j] == 0:
                    ground_truth = np.array([1,0])
                else:
                    ground_truth = np.array([0,1]) 
            
                beta3 = output - ground_truth
                beta2 = np.dot(beta3,self.W2.T) * a1 * (1-a1)
            
                deltab2 += beta3
                deltaw2 += np.dot(a1.T, beta3)
                deltab1 += beta2
                deltaw1 += np.dot(X[j].reshape(2,1), beta2) 
                 
            self.W1 -= self.epsilon * deltaw1 / num_samples
            self.b1 -= self.epsilon * deltab1 / num_samples
            self.W2 -= self.epsilon * deltaw2 / num_samples
            self.b2 -= self.epsilon * deltab2 / num_samples

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
linear = False

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
hidden_dim = 3

# Gradient descent parameters 
epsilon = 0.01
num_epochs = 5000

# Fit model

NN = NeuralNet(input_dim, hidden_dim, output_dim, epsilon)
NN.fit(X,y,num_epochs)


# Plot the decision boundary
plot_decision_boundary(lambda x: NN.predict(x))
plt.title("Neural Net Decision Boundary")         