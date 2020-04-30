import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import time
import pickle


def read_filelines(fileName, split_char='\n'): # file lines excluding any empty line ending
    with open(fileName) as file:  # with automatically closes the file and does exception handling
        filelines = file.read().split(split_char)
        if filelines[-1] in ['', '\n']: # if there was an empty line at the end remove it
            filelines.pop()
    return filelines

def format_time(raw_time):
    return time.strftime("%H:%M:%S", time.gmtime(raw_time))

class BNBClassifier():
    @staticmethod
    def sample_static(x):
        '''

        '''

    def __init__(self): 
        '''

        '''


    def __init_twoweights(self):
       '''

       '''


    def test(self, execution_time, lr):
        '''
        '''
        
# python Classifier.py Spect_train.txt Spect_test.txt
def main(): 
    train_file, test_file = "", ""
    classifier = BNBClassifier()

    print("Reading training and test data from files...")
    train_file = sys.argv[2] 
    test_file = sys.argv[3]

    # SPECT contains only binary values

main()