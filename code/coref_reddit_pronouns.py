'''
Goal: get pronouns that refer to
a word in our vocabulary.
'''

# dictionary that maps vocab word to the number of clusters it appears in. this will be denominator
# create pronoun sets {she, her, hers} {he, him, his} {they, them, their, theirs}
# create dictionary where key is word and value is list [x, y] with x being num clusters with that word having each pronoun group
# how many clusters with that word have a certain pronoun group

import spacy
import csv
import json
import os
import time
import sys
import neuralcoref
from collections import defaultdict

# ROOT = '/global/scratch/lucy3_li/manosphere/'

# TODO Read from Lucy's data, write to Divya's own log folder
ROOT = '/mnt/data0/lucy/manosphere/'
DLOGS = '/mnt/data0/dtadimeti/manosphere/logs/'
POSTS = ROOT + 'data/submissions/'
LOGS = ROOT + 'logs/'
COMMENTS = ROOT + 'data/comments/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'
BOTS = LOGS + 'reddit_bots.txt'
# SAMPLE_COMMENTS = '/mnt/data0/dtadimeti/manosphere/data/sample_data4'

def get_denoms():
        # for each word
        # for each line
            # for each cluster
                # if cluster contains the word
                    # increment dictionary value
    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                words.append(row['entity'])
    
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    
    f = '2010-04'
    month = f.replace('RC_', '')
    
    
    # dictionary that maps vocab word to the number of clusters it appears in. this will be denominator
    denoms = dict()
    
    list_A = ['she', 'her', 'hers']
    list_B = ['he', 'him', 'his']
        
    with open(COMMENTS + 'RC_' + month + '/part-00000', 'r') as infile:  
        for word in words:
            print(word)
            for line in infile:
                d = json.loads(line)
                text = d['body']
                doc = nlp(text)
                for c in doc._.coref_clusters:
                    str_mentions = []
                    for t in c.mentions:
                        str_mentions.append(t.text)
                    if word in str_mentions:
                        if word in denoms:
                            denoms[word] = denoms[word] + 1
                        else:
                            denoms[word] = 1
    print(denoms)
        
        
def get_nums():
    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                words.append(row['entity'])
    
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    
    f = '2010-04'
    month = f.replace('RC_', '')
    
    # dictionary where key is word and value is list [x, y] with x being num clusters with that word having each pronoun group
    nums = dict()
    
    list_A = ['she', 'her', 'hers']
    list_B = ['he', 'him', 'his']
        
    with open(COMMENTS + 'RC_' + month + '/part-00000', 'r') as infile: 
        for word in words:
            for line in infile:
                d = json.loads(line)
                text = d['body']
                doc = nlp(text)
                for c in doc._.coref_clusters:
                    str_mentions = []
                    for t in c.mentions:
                        str_mentions.append(t.text)


                    if word in str_mentions:
                        print(str_mentions)
                        check_A = any(item.text in list_A for item in c.mentions)
                        check_B = any(item.text in list_B for item in c.mentions)
                        
                        if check_A and not check_B:
                            list_of_values = [1, 0]
                        
                        elif check_B and not check_A:
                            list_of_values = [0, 1]
                         
                        elif check_B and check_A:
                            list_of_values = [1, 1]
                        
                        elif not check_A and not check_B:
                            list_of_values = [0, 0]
              
                        if word not in nums:
                            nums[word] = list_of_values
                        else:
                            nums[word][0] = nums[word][0] + list_of_values[0]
                            nums[word][1] = nums[word][1] + list_of_values[1]
        print(nums)



if __name__ == '__main__':
    get_denoms()
    get_nums()