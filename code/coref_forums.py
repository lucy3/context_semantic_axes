'''
Goal: get pronouns that refer to
a word in our vocabulary.
'''
import re
import spacy
import csv
import json
import os
import time
import sys
import neuralcoref
from collections import defaultdict

# ROOT = '/global/scratch/lucy3_li/manosphere/'

ROOT = '/mnt/data0/lucy/manosphere/'
LOGS = ROOT + 'logs/'
DLOGS = '/mnt/data0/dtadimeti/manosphere/logs/'
FORUMS = ROOT + 'data/cleaned_forums/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'

def main(): 
    '''
    Output is formatted as 
    subreddit \t cluster1_word1$cluster1_word2 \t cluster2_word1$cluster2_word2$cluster2_word3$cluster2_word4 \n
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
    
    forum_name = 'avfm'
    
    outfile = open(DLOGS + 'coref_people/' + forum_name, 'w')
    
    with open(FORUMS + forum_name,'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            text = d['text_post']
#             ([0-9]{4}-[0-9]{2})-[0-9]{2}
            date_post = d['date_post']
            date = date_post[0:10]
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
                            
                # SCENARIO 2: 
                elif cluster_main in words:
                    for s in c.mentions:
                        outstring += s.text.lower() + "$"
                   
                outstring += "\t"
                
            outfile.write(date + "\t" + outstring)
            outfile.write("\n")
       
    outfile.close()
    


if __name__ == '__main__':
    main()