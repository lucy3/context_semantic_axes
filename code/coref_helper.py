"""
For calculating gender leaning
"""
ROOT = '/mnt/data0/lucy/manosphere/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'
LOGS = ROOT + 'logs/'
COREF = '/mnt/data0/dtadimeti/manosphere/logs/coref_people/'

from collections import defaultdict, Counter
from tqdm import tqdm
import csv
import os
import pandas as pd

def main(): 
    # load vocabulary 
    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                if row['entity'].lower() == 'she' or row['entity'].lower() == 'he': 
                    continue
                words.append(row['entity'].lower())
                
    d = defaultdict(list) # { (month, community, word) : [fem, masc, masc, fem, etc...] } 
    
    fem = set(['she', 'her', 'hers', 'herself'])
    masc = set(['he', 'him', 'his', 'himself'])
    neut = set(['they', 'them', 'their', 'theirs', 'themself', 'themselves'])
    
    print("Going over coref output...")
    error_file = open(LOGS + 'coref_errors.temp', 'w')
    for month in tqdm(os.listdir(COREF)):
        if month.endswith('_control') or month == '.DS_Store': continue
        with open(COREF + month, 'r') as infile:
            for line in infile: 
                contents = line.strip().split('\t')
                if len(contents) <= 1: continue
                community = contents[0]
                for i in range(1, len(contents)): 
                    cluster = contents[i].lower().split('$')
                    key = None
                    val = set()
                    for w in cluster: 
                        w_tokens = w.split(' ')
                        w_except_first = ' '.join(w_tokens[1:])
                        if w in words: 
                            key = w
                        elif w_except_first in words: 
                            key = w_except_first
                        elif w in fem: 
                            val.add('fem')
                        elif w in masc: 
                            val.add('masc')
                        elif w in neut: 
                            val.add('neut')
                    if key is None: 
                        error_file.write("PROBLEM WITH:" + contents[i] + '\n')
                        error_file.write(month + '\t' + line + '\n')
                        error_file.write("-------\n")
                    for v in val:
                        d[(month, community, key)].append(v)
    error_file.close()                 
    
    print("Creating dataframe...")
    dataframe_d = {'month': [],
                   'community': [], 
                   'word': [], 
                   'fem': [], # count
                   'masc': [], # count
                   'neut': [], # count
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
        
    df = pd.DataFrame.from_dict(dataframe_d)
    
    df.to_csv(LOGS + 'pronoun_df.csv', index=False, header=True)
        

if __name__ == "__main__":
    main()