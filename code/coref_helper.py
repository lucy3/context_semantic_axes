"""
For calculating gender leaning
"""
ROOT = '/mnt/data0/lucy/manosphere/'
OUT_FOLDER = ROOT + 'logs/coref_results/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'
COREF_LOGS = '/mnt/data0/dtadimeti/manosphere/logs/'
COREF_REDDIT = COREF_LOGS + 'coref_reddit/'
COREF_FORUMS = COREF_LOGS + 'coref_forums/'
COREF_CONTROL = COREF_LOGS + 'coref_control/'

from collections import defaultdict, Counter
from tqdm import tqdm
import csv
import os
import pandas as pd

def load_vocabulary(): 
    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                if row['entity'].lower() == 'she' or row['entity'].lower() == 'he': 
                    continue
                words.append(row['entity'].lower())
    return words

def get_pronoun_map(): 
    pronoun_map = {}
    fem = set(['she', 'her', 'hers', 'herself'])
    for p in fem: 
        pronoun_map[p] = 'fem'
    masc = set(['he', 'him', 'his', 'himself'])
    for p in masc: 
        pronoun_map[p] = 'masc'
    they = set(['they', 'them', 'their', 'theirs', 'themself', 'themselves'])
    for p in they: 
        pronoun_map[p] = 'they'
    it = set(['it', 'its', 'itself'])
    for p in it: 
        pronoun_map[p] = 'it'
    you = set(['you', 'your', 'yours', 'yourself', 'yourselves'])
    for p in you: 
        pronoun_map[p] = 'you'
    return pronoun_map

def main(): 
    # load vocabulary 
    words = load_vocabulary()
    pronoun_map = get_pronoun_map()
    
    d = defaultdict(list) # { (month, community, word) : [fem, masc, masc, fem, etc...] } 
    
    error_file = open(OUT_FOLDER + 'reddit_errors.temp', 'w')
    for year_month in tqdm(os.listdir(COREF_REDDIT)):
        line_num = 0
        with open(COREF_REDDIT + year_month, 'r') as infile: 
            for line in infile: 
                contents = line.strip().split('\t')
                if len(contents) <= 1: continue # no clusters
                community = contents[0]
                for clust in contents[1:]:
                    clust = set(clust.lower().split('$'))
                    clust_vocab_terms = set()
                    pronouns = set()
                    for term in clust: 
                        # bigrams and unigrams with deteminers/posessives
                        w_tokens = term.split(' ')
                        if len(w_tokens) > 3: continue
                        # 'the wife' -> wife, 'the hot wife' -> hot wife
                        w_except_first = ' '.join(w_tokens[1:])
                        # 'hot wife' -> wife
                        last_token = w_tokens[-1]
                        if term in words: 
                            # 'wife'
                            clust_vocab_terms.add(term)
                        if w_except_first in words: 
                            clust_vocab_terms.add(w_except_first)
                        if last_token in words: 
                            clust_vocab_terms.add(last_token)
                        # find pronouns
                        if term in pronoun_map: 
                            pronouns.add(pronoun_map[term])
                            
                    if len(clust_vocab_terms) == 0: 
                        print(clust, pronouns)
                        print(line)
                        error_file.write(str(line_num) + ' ' + year_month + '\n')
                        error_file.write('$'.join(clust) + '\n')
                            
                    for k in clust_vocab_terms: 
                        for pn in pronouns: 
                            d[(year_month, community, k)].append(pn)
                line_num += 1
                            
    print("Creating dataframe...")
    dataframe_d = {'month': [],
                   'community': [], 
                   'word': [], 
                   'fem': [], # count
                   'masc': [], # count
                   'neut': [], # count
                   'it': [],
                   'you': []
                          }
    for tup in tqdm(d): 
        month, community, word = tup
        pronoun_count = Counter(d[tup])
        dataframe_d['month'].append(month)
        dataframe_d['community'].append(community)
        dataframe_d['word'].append(word)
        dataframe_d['fem'].append(pronoun_count['fem'])
        dataframe_d['masc'].append(pronoun_count['masc'])
        dataframe_d['neut'].append(pronoun_count['neut'])
        dataframe_d['it'].append(pronoun_count['it'])
        dataframe_d['you'].append(pronoun_count['you'])
        
    df = pd.DataFrame.from_dict(dataframe_d)
    
    df.to_csv(OUT_FOLDER + 'coref_reddit_df.csv', index=False, header=True)
    
# def old(): 
                
#     d = defaultdict(list) # { (month, community, word) : [fem, masc, masc, fem, etc...] } 
    
#     fem = set(['she', 'her', 'hers', 'herself'])
#     masc = set(['he', 'him', 'his', 'himself'])
#     neut = set(['they', 'them', 'their', 'theirs', 'themself', 'themselves'])
#     it = set(['it', 'its', 'itself'])
#     you = set(['you', 'your', 'yours', 'yourself', 'yourselves'])
    
#     print("Going over coref output...")
#     error_file = open(LOGS + '2017-09_errors.temp', 'w')
#     for month in tqdm(os.listdir(COREF)):
#         if month.endswith('_control') or month == '.DS_Store': continue
#         with open(COREF + month, 'r') as infile:
#             for line_number, line in enumerate(infile): 
#                 contents = line.strip().split('\t')
#                 if len(contents) <= 1: continue
#                 community = contents[0]
#                 for i in range(1, len(contents)): 
#                     cluster = contents[i].lower().split('$')
#                     key = None
#                     val = set()
#                     for w in cluster: 
#                         w_tokens = w.split(' ')
#                         w_except_first = ' '.join(w_tokens[1:])
#                         if w in words: 
#                             key = w
#                         elif w_except_first in words: 
#                             key = w_except_first
#                         elif w in fem: 
#                             val.add('fem')
#                         elif w in masc: 
#                             val.add('masc')
#                         elif w in neut: 
#                             val.add('neut')
#                         elif w in it:
#                             val.add('it')
#                         elif w in you:
#                             val.add('you')
	
#                     if key is None: 
#                         error_file.write("PROBLEM WITH:" + contents[i] + '\n')
#                         error_file.write("LINE NUMBER: " + str(line_number) + '\n')
#                         error_file.write(month + '\t' + line + '\n')
#                         error_file.write("-------\n")
#                     for v in val:
#                         d[(month, community, key)].append(v)
#     error_file.close()                 
    
#     print("Creating dataframe...")
#     dataframe_d = {'month': [],
#                    'community': [], 
#                    'word': [], 
#                    'fem': [], # count
#                    'masc': [], # count
#                    'neut': [], # count
# 		   'it': [],
# 		   'you': []
#                   }
#     for tup in tqdm(d): 
#         month, community, word = tup
#         pronoun_count = Counter(d[tup])
#         dataframe_d['month'].append(month)
#         dataframe_d['community'].append(community)
#         dataframe_d['word'].append(word)
#         dataframe_d['fem'].append(pronoun_count['fem'])
#         dataframe_d['masc'].append(pronoun_count['masc'])
#         dataframe_d['neut'].append(pronoun_count['neut'])
#         dataframe_d['it'].append(pronoun_count['it'])
#         dataframe_d['you'].append(pronoun_count['you'])
        
#     df = pd.DataFrame.from_dict(dataframe_d)
    
#     #df.to_csv(LOGS + 'coref_forums_df.csv', index=False, header=True)
        

if __name__ == "__main__":
    main()
