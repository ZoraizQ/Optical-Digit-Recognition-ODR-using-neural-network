import numpy as np
import sys


def read_filelines(fileName, split_char='\n'): # file lines excluding any empty line ending
    with open(fileName) as file:  # with automatically closes the file and does exception handling
        filelines = file.read().split(split_char)
        if filelines[-1] in ['', '\n']: # if there was an empty line at the end remove it
            filelines.pop()
    return filelines


class BNBClassifier():
    @staticmethod
    def accuracy(y_pred, y_true):
        total_length = float(len(y_pred))
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        return np.sum(y_pred == y_true) / total_length


    def calculate_prior(self): # P(Ci)
        total_size = float(self.data.shape[0])
        patient_type_column = self.data[:, 0]
        self.freq_C1 = np.count_nonzero(patient_type_column)
        self.freq_C2 = total_size - self.freq_C1

        prior_C1 = (self.freq_C1)/(total_size)
        prior_C2 = (self.freq_C2)/(total_size)
        return (prior_C1, prior_C2)


    def calculate_likelihood(self, test_row): #P(xi)_given_Ci
        frequencies_xi_C1 = []
        frequencies_xi_C2 = []

        # traversing column wise in split dataset, and element wise in the test_row
        for j in range(self.rows_C1.shape[1]):
            frequencies_xi_C1 += [(self.rows_C1[:, j] == test_row[j]).sum()] # every time there's a match

        for j in range(self.rows_C2.shape[1]):
            frequencies_xi_C2 += [(self.rows_C2[:, j] == test_row[j]).sum()] 

        likelihood_C1 = np.prod(np.array(frequencies_xi_C1) / self.freq_C1) # divide all individual frequencies by class freq 
        # then product of all as per NB formula
        likelihood_C2 = np.prod(np.array(frequencies_xi_C2) / self.freq_C2)
        return (likelihood_C1, likelihood_C2)


    def fit(self, data):
        '''
        # C1 = class normal (1), C2 = class abnormal (0)
        '''
        self.data = data 
        self.prior_C1, self.prior_C2 = self.calculate_prior()

        # storing class wise split data
        # normal type test rows
        self.rows_C1 = np.array(list(filter(lambda row: row[0] == 1, self.data)))[:, 1:] # first columns excluded
        # abnormal type test rows
        self.rows_C2 = np.array(list(filter(lambda row: row[0] == 0, self.data)))[:, 1:]

        self.data_likelihoods = {}
        for i in range(self.data.shape[0]):
            # store without patient type
            self.data_likelihoods[str(data[i][1:])] = self.calculate_likelihood(data[i][1:])
        # preprocessing and memoization using a dictionary, store likelihoods of training data beforehand

    def classify(self, data):
        '''
        # C1 = class normal (1), C2 = class abnormal (0)
        '''

        rows = data.shape[0]
        pred = np.zeros(rows)
        for i in range(rows): # rows to test
            stringed_row = str(data[i][1:]) # without patient type
            likelihood_C1, likelihood_C2 = 1, 1 

            if stringed_row in self.data_likelihoods: # to get memoized result for performance
                likelihood_C1, likelihood_C2 = self.data_likelihoods[stringed_row]
                # print("read from memo")
            else:
                likelihood_C1, likelihood_C2 = self.calculate_likelihood(data[i][1:])

            C1_p = self.prior_C1 * likelihood_C1
            C2_p = self.prior_C2 * likelihood_C2
            
            if C1_p > C2_p:
                pred[i] = 1

        return BNBClassifier.accuracy(pred, data[:, 0])

# python3 Classifier.py Spect_train.txt Spect_test.txt
def main(): 
    train_file, test_file = "", ""

    print("Reading training and test data from files...")
    train_file = sys.argv[1] 
    test_file = sys.argv[2]

    # SPECT contains only binary values
    # first bit => normal (value of 1) or abnormal (value of 0).
    # the rest 22 => test number the patient failed and which he/she passed.
    train_filelines = read_filelines(train_file)
    train = np.array(list(map(lambda fileline : fileline.split(','), train_filelines)), dtype=np.uint8) # smallest possible dtype to store 0/1 (bool has issues)

    test_filelines = read_filelines(test_file)
    test = np.array(list(map(lambda fileline : fileline.split(','), test_filelines)), dtype=np.uint8)

    classifier = BNBClassifier()

    print(f"Training on {train.shape[0]} data points...")
    classifier.fit(train)
    print("Training complete.")

    print(f"Testing on {test.shape[0]} data points...")
    accuracy = classifier.classify(test)
    print(f"Total Accuracy = {accuracy*100}%")


main()