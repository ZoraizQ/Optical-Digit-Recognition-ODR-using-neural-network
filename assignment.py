# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:00:51 2019

@author: YourAverageSciencePal
"""
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
'''
Depending on your choice of library you have to install that library using pip

'''


'''
Read chapter on neural network from the book. Most of the derivatives,formulas 
are already there.
Before starting this assignment. Familiarize yourselves with np.dot(),
What is meant by a*b in 2 numpy arrays.
What is difference between np.matmul and a*b and np.dot.
Numpy already has vectorized functions for addition and subtraction and even for division
For transpose just do a.T where a is a numpy array 
Also search how to call a static method in a class.
If there is some error. You will get error in shapes dimensions not matched
because a*b !=b*a in matrices
'''

def read_filelines(fileName, split_char='\n'): # file lines excluding any empty line ending
    with open(fileName) as file:  # with automatically closes the file and does exception handling
        filelines = file.read().split(split_char)
        if filelines[-1] in ['', '\n']: # if there was an empty line at the end remove it
            filelines.pop()
    return filelines


class NeuralNetwork():

    @staticmethod
    #note the self argument is missing i.e. why you have to search how to use static methods/functions
    def cross_entropy_loss(y_pred, y_true): # ✅
        '''implement cross_entropy loss error function here
        Hint: Numpy has a sum function already
        Numpy has also a log function
        Remember loss is a number so if y_pred and y_true are arrays you have to sum them in the end
        after calculating -[y_true*log(y_pred)]'''
        return np.sum(-y_true * np.log(y_pred))


    @staticmethod
    def accuracy(y_pred, y_true): # ✅
        '''function to calculate accuracy of the two lists/arrays
        Accuracy = (number of same elements at same position in both arrays)/total length of any array
        Ex-> y_pred = np.array([1,2,3]) y_true=np.array([1,2,4]) Accuracy = 2/3*100 (2 Matches and 1 Mismatch)'''
        total_length = float(len(y_pred))
        # using list comprehension, every match of a y_pred and y_true element contributes a 1 to the numerator (sum of matches)
        return np.sum(y_pred == y_true) / total_length
    

    @staticmethod
    def softmax(x): # ✅
        '''Implement the softmax function using numpy here
        Hint: Numpy sum has a parameter axis to sum across row or column. You have to use that
        Use keepdims=True for broadcasting
        You guys should have a pretty good idea what the size of returned value is.
        '''
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True) #sum across column axis = 1 (since j in formula)
    

    @staticmethod
    def sigmoid(x): # ✅   
        '''Implement the sigmoid function using numpy here
        Sigmoid function is 1/(1+e^(-x))
        Numpy even has a exp function search for it.Eh?
        '''
        return 1/(1 + np.exp(-x))
    

    def __init__(self): # ✅  
        '''Creates a Feed-Forward Neural Network.
        "nodes_per_layer" is a list containing number of nodes in each layer (including input layer)
        "num_layers" is the number of layers in your network 
        "input_shape" is the shape of the image you are feeding to the network
        "output_shape" is the number of probabilities you are expecting from your network'''

        self.num_layers = 3 # includes input layer
        self.nodes_per_layer = [784, 30, 10] # input, hidden, output - neural net size
        self.input_shape = 784 # 28x28=784 since flattened images fed
        self.output_shape = 10 # 10 output neurons and classes
        self.__init_weights(self.nodes_per_layer)


    def __init_weights(self, nodes_per_layer): # ✅ 
        '''Initializes all weights and biases between -1 and 1 using numpy'''
        self.weights_ = []
        self.biases_ = []

        for i,_ in enumerate(nodes_per_layer): # hidden and output layers will have weights and biases only
            if i == 0:
                # skip input layer, it does not have weights/bias
                continue
            # np.random.uniform generates a random uniform distributed matrix within values low=-1 and high=1 in this case 
            weight_matrix = np.random.uniform(-1, 1, size=(nodes_per_layer[i-1], nodes_per_layer[i])) # shape == (last layer, current layer)
            self.weights_.append(weight_matrix)
            bias_vector = np.zeros(nodes_per_layer[i]) # vector of zeros of size of nodes in current layer (each node/neuron has a bias)
            self.biases_.append(bias_vector)
    

    def fit(self, Xs, Ys, epochs, lr=1e-3): # ✅ 
        '''Trains the model on the given dataset for "epoch" number of iterations with step size="lr". 
        Returns list containing loss for each epoch.'''
        history = []
        
        input_data = Xs
        layer_activations = self.forward_pass(input_data) # forward pass returns the neuron activations for the sample input data Xs
        deltas = self.backward_pass(Ys, layer_activations) # backward pass uses the targets (Ys) and activations from forward pass to calculate deltas
        
        # list containing input for all layers
        layer_inputs = input_data + (layer_activations[:-1]) # all 'hidden layer' activations added with input_data (output activations excluded)
        self.weight_update(deltas, layer_inputs, lr) # use results to execute gradient descent to update weights, with lr as step size

        y_pred = self.predict(Xs) # predicts output of last layer for given input data / Xs
        epoch_loss = cross_entropy_loss(y_pred, Ys) # Ys has true/test labels
        history.append(epoch_loss) # loss value for the current epoch / iteration stored in history

        return history
    
    
    def forward_pass(self, input_data):
        '''Executes the feed forward algorithm.
        "input_data" is the input to the network in row-major form
        Returns "activations", which is a list of all layer outputs (excluding input layer of course)
        What is activation?
        In neural network you have inputs(x) and weights(w).
        What is first layer? It is your input right?
        A linear neuron is this: y = w.T*x+b =>T is the transpose operator 
        A sigmoid neuron activation is y = sigmoid(w1.T*x+b1) for 1st hidden layer 
        Now for the last hidden layer the activation y = sigmoid(w2.T*y+b2).
        '''
        activations = []

        w1, b1 = self.weights_[0], self.biases_[0] # weights and biases for 1st hidden layer
        y1 = sigmoid(np.dot(w1, input_data) + b1)
        activations.append(y1) # y1, activation in the 1st hidden layer
        
        for i in range (1, self.num_layers-2): # excluding the first hidden layer (index 0) and last index output layer
            wi, bi = self.weights_[i], self.biases_[i] # if there were more hidden layers than just 1
            yi = sigmoid(np.dot(wi, activations[-1]) + bi) # input is now the previous layer's activations
            activations.append(yi) # calculate activations for all layers in between as well
        
        wo, bo = self.weights_[-1], self.biases_[-1] # last weights and biases belong to output layer
        yo = sigmoid(np.dot(wo, activations[-1]) + bo) # input is now the last hidden layer to get activation for output layer
        activations.append(yo)

        return activations
    

    def backward_pass(self, targets, layer_activations):
        '''Executes the back-propagation algorithm.
        "targets" is the ground truth/labels
        "layer_activations" are the return value of the forward pass step
        Returns "deltas", which is a list containing weight update values for all layers (excluding the input layer of course)
        You need to work on the paper to develop a generalized formulae before implementing this.
        Chain rule and derivatives are the pre-requisite for this part.
        '''
        deltas = []
        
        
        return deltas
            
            
    def weight_update(self, deltas, layer_inputs, lr):
        '''Executes the gradient descent algorithm.
        "deltas" is return value of the backward pass step
        "layer_inputs" is a list containing the inputs for all layers (including the input layer)
        "lr" is the learning rate
        You just have to implement the simple weight update equation. 
        
        '''
        

    def predict(self, Xs):
        '''Returns the model predictions (output of the last layer) for the given "Xs".'''
        predictions = []
        
        return predictions
    

    def evaluate(self, Xs, Ys):
        '''Returns appropriate metrics for the task, calculated on the dataset passed to this method.'''
        pred = self.predict(Xs)
        acc = self.accuracy(Xs, Ys) 
        loss = None 
        return loss,acc
        

    def give_images(self, listDirImages): # ✅  
        '''Returns the images and labels from the listDirImages list after reading
        Hint: Use os.listdir(),os.getcwd() functions to get list of all directories
        in the provided folder. Similarly os.getcwd() returns you the current working
        directory. 
        For image reading use any library of your choice. Commonly used are opencv, pillow but
        you have to install them using pip
        "images" is list of numpy array of images 
        labels is a list of labels you read 
        '''
        images = []
        labels = []

        print("Reading images and labels...")
        cwd = os.getcwd()
        for fdir in os.listdir(listDirImages):
            fdir_path = os.path.join(cwd, listDirImages, fdir)
            if fdir in ["test.txt", "train.txt"]: # ignore file/directory list
                continue

            if fdir == "labels.txt":
                labels = read_filelines(fdir_path)
                continue

            for img in os.listdir(fdir_path): # f is a numbered directory otherwise with images
                img_path = os.path.join(cwd, listDirImages, fdir, img)
                images.append(np.array(Image.open(img_path).getdata())) # reading images and converting to np.array as well

        return images, labels

    def generate_labels(self,labels): # ✅
        '''Returns your labels into one hot encoding array
        labels is a list of labels [0,1,2,3,4,1,3,3,4,1........]
        Ex-> If label is 1 then one hot encoding should be [0,1,0,0,0,0,0,0,0,0]
        Ex-> If label is 9 then one hot encoding should be [0,0,0,0,0,0,0,0,0,1]
        Hint: Use sklearn one hot-encoder to convert your labels into one hot encoding array
        "onehotlabels" is a numpy array of labels. In the end just do np.array(onehotlabels).
        '''
        labels_column = labels.reshape(-1, 1) # as the hot encoder requires a column vector
        onehotlabels = OneHotEncoder().fit_transform(labels_column).toarray() # construct, fit to labels and transform
        return np.array(onehotlabels)


    def save_weights(self,fileName): # ✅  
        '''save the weights of your neural network into a file
        Hint: Search python functions for file saving as a .txt'''
        with open(fileName, 'w') as f: # with, closes the file on read/write finish as loop ends
            for weight in self.weights_:
                f.write("%s\n" % weight)


    def reassign_weights(self,fileName): # ✅
        '''assign the saved weights from the fileName to the network
        Hint: Search python functions for file reading
        '''
        with open(fileName) as file:  # Use file to refer to the file object
            filelines = file.read().split('\n')
            if filelines[-1] == '': # if there was an empty line at the end, remove it
                filelines.pop()
            self.weights_ = np.array(filelines, dtype=np.float32)


    def savePlot(self):
        '''function to plot the execution time versus learning rate plot
        You can edit the parameters passed to the savePlot function'''
        # plt.savefig(fileName)


'''
python3 assignment.py train train.txt train-labels.txt 1.54
python3 assignment.py test test.txt test-labels.txt netWeights.txt
'''
def main(): 
    train_file, test_file, train_labels_file, test_labels_file, net_weights_file = "", "", "", "", ""
    learning_rate = 0.0
    mode = sys.argv[1]

    NN = NeuralNetwork()
    

    if mode == "train":
        print("Reading training data from files... (This might take a while.)")
        train_file = sys.argv[2]
        train_labels_file = sys.argv[3]
        learning_rate = float(sys.argv[4])

        train_filelines_mutated = read_filelines(train_file, ']')
        # remove \n in between with replace and remove [ as well, then split 
        # numbers through spaces in arrays and store np arrays with type as np.uint8 (255 max8 so 8 bit unsigned integer)
        # x_train_map = map(lambda fileline : np.array(fileline.replace('\n', '')[1:].split(), dtype=np.uint8), train_filelines_mutated)
        # x_train = np.array(list(x_train_map), dtype=np.uint8) # converted the map iter to get x_train as well
        # x_train = array of flattened images from test dataset
        # y_train = read_filelines(train_labels_file) # list of strings with all class labels for training dataset

        # x_train, y_train = NN.give_images("Raw+images/train") # optionally could have been done

        # NN.savePlot()
    
    elif mode == "test":
        print("Reading test data from files... (This might take a while.)")
        test_file = sys.argv[2]
        test_labels_file = sys.argv[3]
        net_weights_file = sys.argv[4]
        
        test_filelines_mutated = read_filelines(test_file, ']')
        # x_test_map = map(lambda fileline : np.array(fileline.replace('\n', '')[1:].split(), dtype=np.uint8), test_filelines_mutated)
        # x_test = np.array(list(x_test_map), dtype=np.uint8) 
        # y_test = read_filelines(test_labels_file)

        # x_test, y_test = NN.give_images("Raw+images/test") # optionally could have been done


        if net_weights_file in os.listdir(): # if the net weights file already exists in the current directory
            NN.reassign_weights(net_weights_file) # read previous weights from it

        NN.save_weights(net_weights_file)
        
main()

