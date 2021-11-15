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
SAMPLE_COMMENTS = '/mnt/data0/dtadimeti/manosphere/data/sample_data'

def main(): 
    '''
    Output is formatted as 
    subreddit \t cluster1word1$cluster1word2 \t cluster2word1$cluster2word2$cluster2word3$cluster2word4 \n
    '''
  
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
    
               
    outfile = open(DLOGS + 'coref_people/' + "sample", 'w')
    
    deps = defaultdict(dict)
    depheads = defaultdict(dict)
    line_num = 0
     
    with open(SAMPLE_COMMENTS, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            text = d['body']
            sr = d['subreddit']
            # run the coref on text
            doc = nlp(text)
#             print(doc[0])
#             head = doc[0].head.text
#             ans = [doc[0], head]
#             print(ans)
 
#             span = doc[0 : len(doc)]
#             head = span[0].head.text
#             root = span.root.text
#             ans = [span[0], head, root]
#             print(ans)
#             for chunk in doc.noun_chunks:
#                 print(chunk.text, chunk.root.text)
                
            
#             for token in doc:
#                 print(token.text, token.dep_, token.head.text, token.head.pos_,[child for child in token.children])
            
#             deps = [token.dep_ for token in doc]
            print(doc[0].dep_)
#             print(deps)

            
            # if span[0].dep_

#             heads = [token.head.text for token in doc]
#             roots = [chunk.root.text for chunk in doc.noun_chunks]
#             print(len(heads))
#             print(len(roots))
#             print([deps[0].head.text, deps[0].root.text])
#             print(deps)
# #             for elem in deps:
# #                 print(elem.root)

            
            
            
            
#             deps = defaultdict(dict) # {linenum : {idx : deprel} } 
#     depheads = defaultdict(dict) # {linenum : {idx : head idx} }
#     line_num = 0
#     with open(COMMENTS + 'RC_' + month + '/part-00000', 'r') as infile: 
#         for line in infile: 
#              d = json.loads(line)
#              text = d['body']
#              sr = d['subreddit']
#              doc, tokens = tagger.tag(text)
#              entities = entityTagger.tag(tokens)
#              outfile.write(sr + '\t')
#              for entity in entities: 
#                  if entity[2] == 'NOM_PER' or entity[2] == 'PROP_PER':
#                      for idx in range(entity[0], entity[1] + 1): 
#                          deps[line_num][idx] = doc[idx].dep_
#                          depheads[line_num][idx] = doc[idx].head.i
#                      head_idx = doc[entity[0]:entity[1] + 1].root.i 
#                      outfile.write(entity[2] + ' ' + str(entity[0]) + ' ' + str(entity[1]) + ' ' + str(head_idx) + ' ' + str(entity[3]) + '\t')
#              outfile.write('\n')
#              line_num += 1
            
            
            # if .head of first index in span is span.root then we know it is possessive/determiner and we can remove the first index
               
                   
                    
    outfile.close()


if __name__ == '__main__':
    main()