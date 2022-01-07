'''
Goal: get pronouns that refer to
a word in our vocabulary.
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

# Reading from Lucy's data, writing to Divya's log folder
ROOT = '/mnt/data0/lucy/manosphere/'
DLOGS = '/mnt/data0/dtadimeti/manosphere/logs/'
POSTS = ROOT + 'data/submissions/'
LOGS = ROOT + 'logs/'
COMMENTS = ROOT + 'data/comments/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'
BOTS = LOGS + 'reddit_bots.txt'

def main(): 
    '''
    Output format: subreddit \t cluster1word1$cluster1word2 \t cluster2word1$cluster2word2$cluster2word3$cluster2word4 \n
    '''
    # load vocabulary
    with open(ANN_FILE, 'r') as csvfile:
	reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                words_total.append(row['entity'])
                
    # remove 'she' and 'he' from vocab
    words = words_total[:len(words_total) - 2]
    print(words)

    

    # load coref
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    
    f = '2010-04'
    month = f.replace('RC_', '')
    
    # read in bots from reddit_bots.txt, create list
    bots = []
    with open(BOTS, 'r') as infile:
        for line in infile:
            bots.append(line)
            
      
    outfile = open(DLOGS + 'coref_people/' + month, 'w')
    
    with open(COMMENTS + 'RC_' + month + '/part-00000', 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            text = d['body']
            sr = d['subreddit']
            author = d['author']
            # run the coref on text
            doc = nlp(text)
            outstring = ""
            if text == "" or text == "[deleted]" or text == "[removed]":
                continue
            
        
            span = doc[0 : len(doc)]
            head = span[0].head.text
            root = span.root.text            
                      
            for c in doc._.coref_clusters:                        
           
                # SCENARIO 1:
         
                if span[0].dep_ in {'det','poss'}:
                    if (head == root):
                        doc = doc[1:]
                        cluster_contains_vocab = any(word.text in words for word in c.mentions)
                        if cluster_contains_vocab:
                            for s in c.mentions:
                                outstring += s.text.lower() + "$"
                            
                # SCENARIO 2:                                 
                else:
                    cluster_contains_vocab = any(word.text in words for word in c.mentions)
                    if cluster_contains_vocab:
                        for s in c.mentions:
                            outstring += s.text.lower() + "$"
                        outstring += "\t"

            if author not in bots:
                outfile.write(sr.lower() + "\t" + outstring)
                outfile.write("\n")

        
    if os.path.exists(POSTS + 'RS_' + month + '/part-00000'): 
        post_path = POSTS + 'RS_' + month + '/part-00000'
    else: 
        post_path = POSTS + 'RS_v2_' + month + '/part-00000'
    with open(post_path, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            text = d['selftext']
            sr = d['subreddit']
            author = d['author']
            # run the coref on text
            doc = nlp(text)
            outstring = ""
            
            if text == "" or text == "[deleted]" or text == "[removed]":
                continue
            
            # id_post:  2380578
            
            span = doc[0 : len(doc)]
            head = span[0].head.text
            root = span.root.text
                          
                      
            for c in doc._.coref_clusters:                        
                        
#                 SCENARIO 1:           
                if span[0].dep_ in {'det','poss'}:
                    if (head == root):
                        doc = doc[1:]
                        cluster_contains_vocab = any(word.text in words for word in c.mentions)
                        if cluster_contains_vocab:
                            for s in c.mentions:
                                outstring += s.text.lower() + "$"
                            
                # SCENARIO 2: 
                else:
                    cluster_contains_vocab = any(word.text in words for word in c.mentions)
                    if cluster_contains_vocab:
                        for s in c.mentions:
                            outstring += s.text.lower() + "$"
                        outstring += "\t"

            if author not in bots:
                outfile.write(sr.lower() + "\t" + outstring)
                outfile.write("\n")
               
    outfile.close()
    

       
if __name__ == '__main__':
    main()
