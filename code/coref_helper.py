"""
For calculating gender leaning
"""
ROOT = '/mnt/data0/lucy/manosphere/'
COREF_FOLDER = ROOT + 'logs/coref_results/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'
COREF_LOGS = '/mnt/data0/dtadimeti/manosphere/logs/'
SUB_META = ROOT + 'data/subreddits.txt'

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

def get_subreddit_categories(): 
    categories = defaultdict(str)
    categories_rev = defaultdict(list)
    with open(SUB_META, 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in reader: 
            name = row['Subreddit'].strip().lower()
            if name.startswith('/r/'): name = name[3:]
            if name.startswith('r/'): name = name[2:]
            if name.endswith('/'): name = name[:-1]
            categories[name] = row['Category after majority agreement']
            categories_rev[row['Category after majority agreement']].append(name)
    return categories, categories_rev

def create_coref_df(dataset): 
    # load vocabulary 
    words = load_vocabulary()
    pronoun_map = get_pronoun_map()
    categories, categories_rev = get_subreddit_categories()
    
    d = defaultdict(list) # { (month, community, word) : [fem, masc, masc, fem, etc...] } 
    
    error_file = open(COREF_FOLDER + dataset + '_errors.temp', 'w')
    for filename in tqdm(os.listdir(COREF_FOLDER)):
        if not filename.startswith(dataset): continue
        year_month = filename.replace(dataset + '_', '')
        year = year_month.split('-')[0]
        line_num = 0
        with open(COREF_FOLDER + filename, 'r') as infile: 
            reader = csv.reader(infile, delimiter='\t')
            for contents in reader: 
                if len(contents) <= 1: 
                    line_num += 1
                    continue # no clusters
                community = contents[0]
                if dataset == 'reddit': 
                    cat = categories[community]
                elif dataset == 'CONTROL': 
                    cat = 'CONTROL'
                if cat == 'Health' or cat == 'Criticism': 
                    line_num += 1
                    continue
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
                        error_file.write(str(line_num) + ' ' + year_month + '\n')
                        error_file.write('\t'.join(contents) + '\n')
                        error_file.write('$'.join(clust) + '\n')

                    for k in clust_vocab_terms: 
                        for pn in pronouns: 
                            d[(year, cat, k)].append(pn)
                line_num += 1
                            
    print("Creating dataframe...")
    dataframe_d = {'year': [],
                   'community': [], 
                   'word': [], 
                   'fem': [], # count
                   'masc': [], # count
                   'they': [], # count
                   'it': [],
                   'you': []
                          }
    for tup in tqdm(d): 
        year, community, word = tup
        pronoun_count = Counter(d[tup])
        dataframe_d['year'].append(year)
        dataframe_d['community'].append(community)
        dataframe_d['word'].append(word)
        dataframe_d['fem'].append(pronoun_count['fem'])
        dataframe_d['masc'].append(pronoun_count['masc'])
        dataframe_d['they'].append(pronoun_count['they'])
        dataframe_d['it'].append(pronoun_count['it'])
        dataframe_d['you'].append(pronoun_count['you'])
        
    df = pd.DataFrame.from_dict(dataframe_d)
    
    df.to_csv(COREF_FOLDER + 'coref_' + dataset + '_df.csv', index=False, header=True)
    
def main(): 
    create_coref_df('CONTROL')
    create_coref_df('reddit')

if __name__ == "__main__":
    main()
