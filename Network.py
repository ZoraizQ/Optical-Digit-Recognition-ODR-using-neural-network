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
import time
import pickle

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
because a*b != b*a in matrices
'''

def read_filelines(fileName, split_char='\n'): # file lines excluding any empty line ending
    with open(fileName) as file:  # with automatically closes the file and does exception handling
        filelines = file.read().split(split_char)
        if filelines[-1] in ['', '\n']: # if there was an empty line at the end remove it
            filelines.pop()
    return filelines

def format_time(raw_time):
    return time.strftime("%H:%M:%S", time.gmtime(raw_time))

class NeuralNetwork():

    #note the self argument is missing i.e. why you have to search how to use static methods/functions
    @staticmethod
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
        y_pred, y_true = np.array(y_pred), np.array(y_true)
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
        total_images = Xs.shape[0]

        print(f"Training model for {epochs} epochs...")
        execution_start_time = time.time()
        for e in range(epochs):
            # Xs has multiple images (rows) and Ys has multiple corresponding targets, evaluate them one by one
            
            elapsed_time = time.time()
            for i in range(total_images):
                if i%5000 == 0:
                    elapsed_time = time.time() - elapsed_time
                    print(f"Images processed: {i}, Elapsed time", format_time(elapsed_time), "...")

                input_image = Xs[i].reshape((1,self.input_shape))
                target = Ys[i].reshape((1,self.output_shape))
                # print(input_image.shape)
                layer_activations = self.forward_pass(input_image) # forward pass returns the neuron activations for the sample input image
                deltas = self.backward_pass(target, layer_activations) # backward pass uses the targets and activations from forward pass to calculate deltas
                
                # list containing inputs for all layers (input->hidden + hidden->output) 
                layer_inputs = [input_image, layer_activations[0]] # inputs from input layer and hidden layer(activations) combined 
                self.weight_update(deltas, layer_inputs, lr) # use results to execute gradient descent to update weights, with lr as step size

            epoch_loss, epoch_accuracy = self.evaluate(Xs, Ys)
            history.append(epoch_loss) # loss value for the current epoch / iteration stored in history

            print(f"Epoch Number {e} -------> {epoch_accuracy * total_images}/{total_images} images correctly classified")
            print(f"Accuracy {epoch_accuracy*100} % ------------------- Error {100-epoch_accuracy*100} %")
        
        execution__time = time.time() - execution_start_time
        print(f"Execution time {format_time(execution__time)} (HH:MM:SS)")

        return history
    
    
    def forward_pass(self, input_data): # ✅  
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
        activations = [] # activation = squished weighted sum of activations in previous layer + bias for inactivity threshold

        wh, bh = self.weights_[0], self.biases_[0] # weights and biases for hidden layer
        zh = np.dot(input_data, wh) + bh  # z(L) = wL * a(L-1) + b(L)        
        ah = NeuralNetwork.sigmoid(zh) # a(L)= sigmoid(z(L))  or  y = sigmoid(wh.T*x+bh)
        
        wo, bo = self.weights_[1], self.biases_[1] # last weights and biases belong to output layer
        zo = np.dot(ah, wo) + bo
        ao = NeuralNetwork.sigmoid(zo) # input is now the last hidden layer to get activation for output layer

        activations = [ah, ao]
        return activations
    

    def backward_pass(self, targets, layer_activations): # ✅
        '''Executes the back-propagation algorithm.
        "targets" is the ground truth/labels
        "layer_activations" are the return value of the forward pass step
        Returns "deltas", which is a list containing weight update values for all layers (excluding the input layer of course)
        You need to work on the paper to develop a generalized formulae before implementing this.
        Chain rule and derivatives are the pre-requisite for this part.
        '''
        deltas = [] # gradient vector for weights
        '''
        y = targets
        sensitivity to weights:
            w(L) = weights for layer L, z(L) = w(L)*a(L-1) + b(L) for layer L
            a(L) = activations for layer L (sigmoid(z(L))), and C0 is the cost function = summation of (a(jL)-y)^2
            pC0 / pw(L) = pz(L) / pw(L)  * pa(L) / pz(L) * pC0 / pa(L) using the chain rule
            pC0 / pw(L) = deriv of sigmoid(z(L)) * a(L-1) * 2(aL-y)
            neurons that fire together, wire together - change in cost depends on a(L-1) in last partial
        sensitivity to bias:
            pC0 / pb(L) = deriv of sigmoid(z(L)) * 2(aL-y)
        sensitivity to previous activation:
            pC0 / pa(L-1) = w(L) * deriv of sigmoid(z(L)) * 2(a(L)-y)
            pC0 / pa(L-1) = w(L) * (a(L) * (1 - a(L))) * (a(L)-y)
            summations and indices hidden
        acknowledgements: 3Blue1Brown, neuralnetworksanddeeplearning.com
        '''

        y = targets
        ah, ao = layer_activations[0], layer_activations[1] # activation in (first and only here) hidden layer
        # and activation in output layer
        wo = self.weights_[1]

        # going backwards, output layer first
        # ∂C/∂aLj=(aLj−yj)
        cost_aoj = (ao - y) # error in output layer, depends on firstly 
        # from derivative of cost function wrt activation from a(L-1), w(L), b(L) 
        sigmoid_prime_o = ao * (1 - ao) # derivative of activation a0 OR also sigmoid(z(L))
        # since d sigmoid(x) / d(x) = sigmoid(x) * (1 - sigmoid(x))
        δ_wo = cost_aoj * sigmoid_prime_o # Hadamard product computed since both are arrays - delta (weight update value) for output layer
        
        # for the next hidden layer layer (wl+1)Tδx,l+1) is used to get cost_ahj
        cost_ahj = (np.dot(wo, δ_wo.T)).T
        sigmoid_prime_h = ah * (1 - ah)
        δ_wh = cost_ahj * sigmoid_prime_h
        
        deltas = [δ_wh, δ_wo] # correct order
        return deltas
            
            
    def weight_update(self, deltas, layer_inputs, lr): # ✅
        '''Executes the gradient descent algorithm.
        "deltas" is return value of the backward pass step
        "layer_inputs" is a list containing the inputs for all layers (including the input layer)
        "lr" is the learning rate
        You just have to implement the simple weight update equation. 
        
        '''
        [ah, ao] = layer_inputs
        [δ_wh, δ_wo] = deltas
        
        self.weights_[1] -= lr * np.dot(ao.T, δ_wo) # gradient descent rule for weights
        # update factor => - learning rate * ∑x δ(x,l) * (ax(L−1)).T
        self.biases_[1] -= lr * np.sum(δ_wo)

        self.weights_[0] -= lr * np.dot(ah.T, δ_wh)
        self.biases_[0] -= lr * np.sum(δ_wh)
        
 

    def predict(self, Xs): # ✅
        '''Returns the model predictions (output of the last layer) for the given "Xs".'''
        predictions = []
        for i in range (Xs.shape[0]): # for image i in input data
            # extract the ith row in input data Xs
            current_image = np.array([Xs[i]]) # as a column with a single row vector
            current_prediction = (self.forward_pass(current_image)[-1]).reshape(self.output_shape) 
            # output of last layer (activation from  forward pass) for current sample image is a column vector,
            # so it is flattened to a row vector with reshape == becoming the current prediction
            predictions.append(current_prediction)

        predictions = np.array(predictions) # convert to np array from list
        return predictions
    

    def evaluate(self, Xs, Ys): # ✅  
        '''Returns appropriate metrics for the task, calculated on the dataset passed to this method.'''
        pred = self.predict(Xs)
        # acc = NeuralNetwork.accuracy(Xs, Ys) 
        acc = NeuralNetwork.accuracy(pred.argmax(axis=1), Ys.argmax(axis=1))
        loss = NeuralNetwork.cross_entropy_loss(pred, Ys) # y_pred/pred == predicted labels, Ys == true/test labels
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
        labels_column = np.array(labels).reshape(-1, 1) # as the hot encoder requires a column vector
        onehotlabels = OneHotEncoder().fit_transform(labels_column).toarray() # construct, fit to labels and transform
        return np.array(onehotlabels)


    def save_weights(self,fileName): # ✅  
        '''save the weights of your neural network into a file
        Hint: Search python functions for file saving as a .txt'''
        with open(fileName, 'wb') as outfile: # write binary mode
            pickle.dump(self.weights_, outfile)
        

    def reassign_weights(self,fileName): # ✅
        '''assign the saved weights from the fileName to the network
        Hint: Search python functions for file reading
        '''
        with open(fileName, 'rb') as infile: # read binary mode
            self.weights_ = pickle.load(infile)


    def savePlot(self, execution_time, lr):
        '''function to plot the execution time versus learning rate plot
        You can edit the parameters passed to the savePlot function'''
        # plt.savefig(fileName)


'''
python3 Network.py train train.txt train-labels.txt 1
python3 Network.py test test.txt test-labels.txt netWeights.txt
'''
def main(): 
    train_file, test_file, train_labels_file, test_labels_file, net_weights_file = "", "", "", "", ""
    lr = 1 # picked from the three values (0.01, 0.1, 1) works amazing, moderately large step size required
    mode = sys.argv[1]
    NN = NeuralNetwork()
    np.random.seed(123) # for predictability

    if mode == "train":
        print("Reading training data from files... (This might take a while.)")
        train_file = sys.argv[2]
        train_labels_file = sys.argv[3]
        lr = float(sys.argv[4])

        train_filelines_mutated = read_filelines(train_file, ']')
        # remove \n in between with replace and remove [ as well, then split 
        # numbers through spaces in arrays and store np arrays with type as np.uint8 (255 max8 so 8 bit unsigned integer)
        x_train_map = map(lambda fileline : fileline.replace('\n', '')[1:].split(), train_filelines_mutated)
        x_train = np.array(list(x_train_map), dtype=np.float32) # converted the map iter to get x_train as well
        # x_train = array of flattened images from test dataset
        # x_train, y_train = NN.give_images("Raw+images/train") # optionally could have been done
        x_train /= 255.0 # normalizing images rgb to 0-1

        print("Reading labels...")
        y_train = read_filelines(train_labels_file) # list of strings with all class labels for training dataset
        y_train = NN.generate_labels(y_train) # convert labels to one hot encoded ones
        
        history = NN.fit(x_train, y_train, 2, lr)

        # NN.savePlot("train_plot.jpg")
        NN.save_weights("netWeights.txt")
        
    
    elif mode == "test":
        print("Reading test data from files... (This might take a while.)")
        test_file = sys.argv[2]
        test_labels_file = sys.argv[3]
        net_weights_file = sys.argv[4]
        
        test_filelines_mutated = read_filelines(test_file, ']')
        x_test_map = map(lambda fileline : fileline.replace('\n', '')[1:].split(), test_filelines_mutated)
        x_test = np.array(list(x_test_map), dtype=np.float32) 
        # x_test, y_test = NN.give_images("Raw+images/test") # optionally could have been done
        x_test /= 255.0 # normalizing rgb to 0-1
        
        print("Reading labels...")
        y_test = read_filelines(test_labels_file)
        y_test = NN.generate_labels(y_test)

        if net_weights_file in os.listdir(): # if the net weights file already exists in the current directory
            NN.reassign_weights(net_weights_file) # read previous weights from it

        print("Evaluating tests...")
        loss, accuracy = NN.evaluate(x_test, y_test)
        print(f"Accuracy {accuracy*100} % ------------------- Error {100-accuracy*100} %")


main()

