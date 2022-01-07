'''
Goal: get gender leanings of pronouns that refer to
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
    
    weird_pronouns = ['women','men','people','man','girls','guy','guys','someone',
                          'friends','person','friend','anyone','children','boys','father',
                          'females','mom','humans','individuals','sir','sisters','soldiers',
                          'infants','prisoners','bartenders','introverts','you guys','most women',
                          'everyone else','one girl','most guys','white women','new people',
                          'cute girls','we men','rich people','one party','one parent',
                          'other companies','we women','abusive parents']
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                words.append(row['entity'])
    
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
            
      
    denoms = dict()
    nums = dict()  

    for word in words:
        denoms[word] = 0
        nums[word] = [0, 0, 0]

    list_A = ['she', 'her', 'hers']
    list_B = ['he', 'him', 'his']
    list_C = ['they', 'them', 'their', 'theirs']
               
#     outfile = open(DLOGS + 'coref_people/' + month, 'w')
    
    with open(COMMENTS + 'RC_' + month + '/part-00000', 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            text = d['body']
            author = d['author']
            # run the coref on text
            doc = nlp(text)
            if text == "" or text == "[deleted]" or text == "[removed]" or author in bots:
                continue
                                    
            for c in doc._.coref_clusters:
                str_mentions = []
                for token in c.mentions:
                    # if none of tokens are pronouns, skip cluster?
                    str_mentions.append(token.text.lower())
#                 for word in words:
                for word in weird_pronouns:
                    if word in str_mentions:
                
#                         denoms[word] = denoms[word] + 1
                        
                        check_A = any(item in list_A for item in str_mentions)
                        check_B = any(item in list_B for item in str_mentions)
                        check_C = any(item in list_C for item in str_mentions)
                
                        if not check_A and not check_B and not check_C:
                            print(word)
                            print(c.mentions)
        
                
#                         if check_A:
#                             nums[word][0] += 1
#                         if check_B:
#                             nums[word][1] += 1
#                         if check_C: 
#                             nums[word][2] += 1
                            
#                         if check_A and check_B:
#                             print(c)
#                             print(text)
#                             print("______________________________________")

        
    if os.path.exists(POSTS + 'RS_' + month + '/part-00000'): 
        post_path = POSTS + 'RS_' + month + '/part-00000'
    else: 
        post_path = POSTS + 'RS_v2_' + month + '/part-00000'
    with open(post_path, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            text = d['selftext']
            author = d['author']
            # run the coref on text
            doc = nlp(text)
            
            
            if text == "" or text == "[deleted]" or text == "[removed]":
                continue
                          
                      
            for c in doc._.coref_clusters:
                str_mentions = []
                                        
                for token in c.mentions:
                    str_mentions.append(token.text.lower())
#                 for word in words:
                for word in weird_pronouns:  
                    if word in str_mentions:
                
                        
#                         denoms[word] = denoms[word] + 1 
                        
                        check_A = any(item in list_A for item in str_mentions)
                        check_B = any(item in list_B for item in str_mentions)
                        check_C = any(item in list_C for item in str_mentions)
                
                        if not check_A and not check_B and not check_C:
                            print(word)
                            print(c.mentions)
                
#                         if check_A:
#                             nums[word][0] += 1
#                         if check_B:
#                             nums[word][1] += 1
#                         if check_C: 
#                             nums[word][2] += 1
                        
                    
               
#     outfile.close()
    
    
#     data = []
#     for word in words:
#         if denoms[word] != 0:
#             she = (nums[word][0])/(denoms[word])
#             he = (nums[word][1])/(denoms[word])
#             they = (nums[word][2])/(denoms[word])
#             clusters = denoms[word]
#             data.append([word, she, he, they, clusters])
#             row_sum = she + he + they 
# #             if row_sum < 1: 
# #                 print(word)

                                
#     col_names = ["She", "He", "They", "Clusters"]
                                
#     print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))
        
#     with open('table.txt', 'w') as f:
#         f.write(tabulate(data, headers=col_names, tablefmt="fancy_grid"))
        
        
        
    
        




if __name__ == '__main__':
    main()