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

# ROOT = '/global/scratch/lucy3_li/manosphere/'

# TODO Read from Lucy's data, write to Divya's own log folder
ROOT = '/mnt/data0/lucy/manosphere/'
DLOGS = '/mnt/data0/dtadimeti/manosphere/logs/'
POSTS = ROOT + 'data/submissions/'
LOGS = ROOT + 'logs/'
COMMENTS = ROOT + 'data/comments/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'
BOTS = LOGS + 'reddit_bots.txt'
# SAMPLE_COMMENTS = '/mnt/data0/dtadimeti/manosphere/data/sample_data2'

def main(): 
    '''
    Output is formatted as 
    subreddit \t cluster1word1$cluster1word2 \t cluster2word1$cluster2word2$cluster2word3$cluster2word4 \n
    '''
    count = 0
    # load vocabulary 
    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                words.append(row['entity'])
    

    # load the coref thingy 
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    
    f = '2010-04'
    month = f.replace('RC_', '')
    
    # read in bots from reddit_bots.txt, create list
    bots = []
    with open(BOTS, 'r') as infile:
        for line in infile:
            bots.append(line)
            
      
    denoms = dict()
    nums = dict()  

    for word in words:
        denoms[word] = 0
        nums[word] = [0, 0, 0]

    
 
    list_A = ['she', 'her', 'hers']
    list_B = ['he', 'him', 'his']
    list_C = ['they', 'them', 'their', 'theirs']
               
    outfile = open(DLOGS + 'coref_people/' + month, 'w')
#     outfile = open(DLOGS + 'coref_people/' + "sample1", 'w')
    
    with open(COMMENTS + 'RC_' + month + '/part-00000', 'r') as infile: 
#     with open(SAMPLE_COMMENTS, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            text = d['body']
            sr = d['subreddit']
            author = d['author']
            # run the coref on text
            doc = nlp(text)
            outstring = ""
            if text == "" or text == "[deleted]" or text == "[removed]":
                count = count + 1
                continue
        
            span = doc[0 : len(doc)]
            head = span[0].head.text
            root = span.root.text

                          
                      
            for c in doc._.coref_clusters:
                str_mentions = []
                for token in c.mentions:
                    str_mentions.append(token.text)
                for word in words:
                    if word in str_mentions:
                        denoms[word] = denoms[word] + 1
                        
                        check_A = any(item in list_A for item in str_mentions)
                        check_B = any(item in list_B for item in str_mentions)
                        check_C = any(item in list_C for item in str_mentions)
                
                        if check_A:
                            nums[word][0] += 1
                        if check_B:
                            nums[word][1] += 1
                        if check_C: 
                            nums[word][2] += 1
                        
           
                # SCENARIO 1:
                # if first index of span is determiner or possessive and head of first index in span is span.root
                # remove the first index (which is the determiner or possessive)    
                # and see if the rest of the span is in the vocab.
                                
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
        count = 0
        for line in infile: 
            d = json.loads(line)
            text = d['selftext']
            sr = d['subreddit']
            author = d['author']
            # run the coref on text
            doc = nlp(text)
            outstring = ""
            
            
            if text == "" or text == "[deleted]" or text == "[removed]":
                count = count + 1
                continue
            
            span = doc[0 : len(doc)]
            head = span[0].head.text
            root = span.root.text
                          
                      
            for c in doc._.coref_clusters:
                str_mentions = []
                                        
                for token in c.mentions:
                    str_mentions.append(token.text)
                for word in words:
                    if word in str_mentions:
                        denoms[word] = denoms[word] + 1 
                        
                        check_A = any(item in list_A for item in str_mentions)
                        check_B = any(item in list_B for item in str_mentions)
                        check_C = any(item in list_C for item in str_mentions)
                
                        if check_A:
                            nums[word][0] += 1
                        if check_B:
                            nums[word][1] += 1
                        if check_C: 
                            nums[word][2] += 1
                        
                        
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
    
    
    data = []
    for word in words:
        if denoms[word] != 0:
            she = (nums[word][0])/(denoms[word])
            he = (nums[word][1])/(denoms[word])
            they = (nums[word][2])/(denoms[word])
            data.append([word, she, he, they])
                                
                                
    col_names = ["She", "He", "They"]
                                
    print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))
        
    with open('table.txt', 'w') as f:
        f.write(tabulate(data, headers=col_names, tablefmt="fancy_grid"))
        




if __name__ == '__main__':
    main()