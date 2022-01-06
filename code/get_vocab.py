'''
Get list of vocab words. 
'''
import csv
import json 
import os
from collections import defaultdict

ROOT = '/mnt/data0/lucy/manosphere/'
DLOGS = '/mnt/data0/dtadimeti/manosphere/logs/'
POSTS = ROOT + 'data/submissions/'
LOGS = ROOT + 'logs/'
COMMENTS = ROOT + 'data/comments/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'
BOTS = LOGS + 'reddit_bots.txt'

def main(): 
    # load vocabulary 
    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                words.append(row['entity'])
	
    if 'she' in words: print("true")
#for word in words: print(word)

if __name__ == '__main__':
    main()
