'''
Goal: For every plural word, see if you can find its singular word in our vocabulary.
Motivation: To get more value out of coref_pronouns.py because otherwise plural vocab words will always be referred to 
with {they, them, their}. 
'''

import spacy
import csv
import json 
import os
import time
import sys
import neuralcoref
from collections import defaultdict
from tabulate import tabulate
from nltk.stem import PorterStemmer

ROOT = '/mnt/data0/lucy/manosphere/'
DLOGS = '/mnt/data0/dtadimeti/manosphere/logs/'
POSTS = ROOT + 'data/submissions/'
LOGS = ROOT + 'logs/'
COMMENTS = ROOT + 'data/comments/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'

def main(): 
    ps = PorterStemmer()
   
    # load vocabulary 
    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                words.append(row['entity'])
    print(len(words))
    
    count = 0
                
    for word in words:
        # check that stemmed version is not the same as before
        # check if the stemmed version is in vocab 
        if word != ps.stem(word) and ps.stem(word) in words:
            count = count + 1
#             print(word + "->" + ps.stem(word))
    print(count)
    
   
                                

        




if __name__ == '__main__':
    main()