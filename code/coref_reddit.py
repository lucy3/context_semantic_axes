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

def main(): 
    '''
    Output is formatted as 
    subreddit \t cluster1word1$cluster1word2 \t cluster2word1$cluster2word2$cluster2word3$cluster2word4 \n
    '''
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
    
               
    outfile = open(DLOGS + 'coref_people/' + month + "_nobots", 'w')
#     outfile = open(DLOGS + 'coref_people/' + "sample", 'w')
    
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
        
            span = doc[0 : len(doc)]
            head = span[0].head.text
            root = span.root.text
                          
                      
            for c in doc._.coref_clusters:
                # most representative mention in the cluster
                # cluster_main = c.main.text
               
                
                # SCENARIO 1:
                # if first index of span is determiner or possessive and head of first index in span is span.root
                # remove the first index (which is the determiner or possessive)    
                # and see if the rest of the span is in the vocab.
                
                if span[0].dep_ in {'det','poss'}:
                    if (head == root):
                        doc = doc[1:]
                        cluster_contains_vocab = any(word.text in words for word in c.mentions)
                        # if cluster_main in words:
                        if cluster_contains_vocab:
                            for s in c.mentions:
                                outstring += s.text.lower() + "$"
                            
                # SCENARIO 2: 
                # elif cluster_main in words:
                                
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
        
            span = doc[0 : len(doc)]
            head = span[0].head.text
            root = span.root.text
                          
                      
            for c in doc._.coref_clusters:
                # most representative mention in the cluster
                cluster_main = c.main.text
                
                # SCENARIO 1:
                # if first index of span is determiner or possessive and head of first index in span is span.root
                # remove the first index (which is the determiner or possessive)    
                # and see if the rest of the span is in the vocab.
                
                if span[0].dep_ in {'det','poss'}:
                    # do I need this check?
                    if (head == root):
                        doc = doc[1:]
                        if cluster_main in words:
                            for s in c.mentions:
                                outstring += s.text.lower() + "$"
                            outstring += "\t"
                            
                # SCENARIO 2: 
                elif cluster_main in words:
                    for s in c.mentions:
                        outstring += s.text.lower() + "$"
                    outstring += "\t"
                        
            if author not in bots:
                outfile.write(sr.lower() + "\t" + outstring)
                outfile.write("\n")

        
                    
    outfile.close()

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
    main()
    get_denoms()
    get_nums()