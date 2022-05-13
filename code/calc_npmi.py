"""
Building off of wikipedia_embeddings.py, 
this script is used to get embeddings based on
contexts with high npmi, accounting for how
we need to run leave-one-out validation later on
"""

import requests
import json
import copy
from tqdm import tqdm
from transformers import BasicTokenizer, BertTokenizerFast, BertModel, BertTokenizer
from functools import partial
from collections import Counter, defaultdict
import random
import torch
import numpy as np
import math
import sys

ROOT = '/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_axes_contexts(axis, adj_lines): 
    axis_lines = set()
    for adj in axis: 
        for line in adj_lines[adj]: 
            axis_lines.add((adj, line))
    return axis_lines

def get_coocur_counts(line, tokenizer=None, lines_adj=None):
    contents = line.split('\t')
    line_num = contents[0]
    words_in_line = lines_adj[line_num]
    text = '\t'.join(contents[1:]).lower()
    # account for dashed words
    text = text.replace('-', 'xqxq')
    tokens = tokenizer.tokenize(text)
    counts = Counter(tokens)
    ret = []
    for adj in words_in_line: 
        counts_copy = copy.deepcopy(counts)
        # don't count adj co-occurring with self
        counts_copy[adj] -= 1
        ret.append((adj, counts_copy))
    return ret

def get_adj_word_counts(): 
    '''
    Input: wikipedia data
    Output: {adj : { w : total count in that adj's lines} }
    '''
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import Row, SQLContext
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    
    with open(LOGS + 'wikipedia/adj_lines.json', 'r') as infile: 
        adj_lines = json.load(infile) # {adj : [line IDs]}
    lines_adj = defaultdict(list) # {line ID: [adj]}
    for adj in adj_lines: 
        for line in adj_lines[adj]: 
            lines_adj[str(line)].append(adj.replace('-', 'xqxq'))
        
    btokenizer = BasicTokenizer(do_lower_case=True)
    wikipedia_file = LOGS + 'wikipedia/adj_data/part-00000'
    data = sc.textFile(wikipedia_file)
    data = data.flatMap(partial(get_coocur_counts, tokenizer=btokenizer, lines_adj=lines_adj))
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    ret = data.collectAsMap()
    with open(LOGS + 'wikipedia/adj_coocur_counts.json', 'w') as outfile: 
        json.dump(ret, outfile)
        
    sc.stop()
        
def npmi_helper(totals_synset, l_counts, r_counts): 
    if (l_counts + r_counts) < 5: 
        # disregard rare words 
        return (0, 0)
    # p(w) = count of w in both axes / total count in both axes
    p_w = (l_counts + r_counts) / totals_synset['both']
    # left npmi
    if l_counts == 0: 
        npmi_left = -1
    else: 
        p_w_left = l_counts / totals_synset['1']
        pmi_left = math.log(p_w_left / p_w)
        h_left = -math.log(l_counts / totals_synset['both'])
        npmi_left = pmi_left / h_left
    # right npmi
    if r_counts == 0: 
        npmi_right = -1
    else: 
        p_w_right = r_counts / totals_synset['2']
        pmi_right = math.log(p_w_right / p_w)
        h_right = -math.log(r_counts / totals_synset['both'])
        npmi_right = pmi_right / h_right
    return npmi_left, npmi_right

def get_axes_adj(): 
    # full axes' adj { axes : (list1, list2) } 
    axes_adj = defaultdict(tuple) 
    with open(LOGS + 'semantics_val/wordnet_axes.txt', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t') 
            if len(contents) < 3: continue # no antonyms
            synset = contents[0]
            axis1 = contents[1].split(',')
            axis2 = contents[2].split(',')
            axes_adj[synset] = (axis1, axis2)
    return axes_adj

def get_npmi_axes_contexts(): 
    '''
    For each axes, gets word co-occurrence counts with each adj 
    in the axes. Use these counts to calculate npmi of each
    co-occuring word with each side of the axes. 
    ''' 
    with open(LOGS + 'wikipedia/adj_coocur_counts.json', 'r') as infile: 
        adj_coocur_counts = defaultdict(Counter, json.load(infile)) # {adj : { w : total count }}
        
    axes_adj = get_axes_adj() # full axes' adj { axes : (list1, list2) } 
    
    for synset in tqdm(axes_adj): 
        if synset != 'educated.a.01': continue # TODO: delete
        left, right = axes_adj[synset]
        left_counts = Counter()
        for adj in left: 
            left_counts += adj_coocur_counts[adj.replace('-', 'xqxq')]
        right_counts = Counter()
        for adj in right: 
            right_counts += adj_coocur_counts[adj.replace('-', 'xqxq')]
        totals = Counter() # {'1': total, '2': total, 'both': total}
        totals['1'] = sum(left_counts.values())
        totals['2'] = sum(right_counts.values())
        totals['both'] = totals['1'] + totals['2']
        
        # full axes npmi
        vocab = set(left_counts.keys()) | set(right_counts.keys())
        npmi_scores = defaultdict(dict) # keys are what adj is left out, 'N/A' for full set
        for w in vocab: 
            npmi_left, npmi_right = npmi_helper(totals, left_counts[w], right_counts[w])
            npmi_scores['N/A'][w] = (npmi_left, npmi_right)
            
        # leave one out npmi 
        for adj in left: 
            loov_counts = copy.deepcopy(left_counts)
            loov_counts -= adj_coocur_counts[adj]
            loov_totals = Counter()
            loov_totals['1'] = sum(loov_counts.values())
            loov_totals['2'] = totals['2']
            loov_totals['both'] = loov_totals['1'] + loov_totals['2']
            for w in vocab: 
                npmi_left, npmi_right = npmi_helper(loov_totals, loov_counts[w], right_counts[w])
                npmi_scores[adj][w] = (npmi_left, npmi_right)
            
        for adj in right: 
            loov_counts = copy.deepcopy(right_counts)
            loov_counts -= adj_coocur_counts[adj]
            loov_totals = Counter()
            loov_totals['1'] = totals['1']
            loov_totals['2'] = sum(loov_counts.values())
            loov_totals['both'] = loov_totals['1'] + loov_totals['2']
            for w in vocab: 
                npmi_left, npmi_right = npmi_helper(loov_totals, left_counts[w], loov_counts[w])
                npmi_scores[adj][w] = (npmi_left, npmi_right)
                
        with open(LOGS + 'wikipedia/npmi_scores/' + synset + '.json', 'w') as outfile: 
            json.dump(npmi_scores, outfile)
        
def get_high_npmi_lines(): 
    '''
    This function was ABANDONED because the npmi
    scores were not helpful. 
    '''
    file_name = 'violent.a.01.json'
    npmi_scores_file = LOGS + 'wikipedia/npmi_scores/' + file_name
    btokenizer = BasicTokenizer(do_lower_case=True)
    
    with open(npmi_scores_file, 'r') as infile: 
        npmi_scores = json.load(infile)
        
    with open(LOGS + 'wikipedia/adj_lines.json', 'r') as infile: 
        adj_lines = json.load(infile) # {adj : [line IDs]}
        
    this_synset = file_name.replace('.json', '')
    with open(LOGS + 'semantics_val/wordnet_axes.txt', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t') 
            if len(contents) < 3: continue # no antonyms
            synset = contents[0]
            if synset == this_synset: 
                axis_left = contents[1].split(',')
                axis_right = contents[2].split(',')
    
    # get lines to create two corpora
    corpora_left = set()
    for adj in axis_left: 
        adj = adj.replace('-', 'xqxq')
        corpora_left.update(adj_lines[adj])
    corpora_right = set()
    for adj in axis_right: 
        adj = adj.replace('-', 'xqxq')
        corpora_right.update(adj_lines[adj])
    
    # calculate npmi for full set of sentences 
    full_scores = npmi_scores['N/A'] 
    left_line_scores = Counter()
    right_line_scores = Counter()
    lines2tokens = {}
    wikipedia_file = LOGS + 'wikipedia/adj_data/part-00000' # TODO: change to adj_data
    with open(wikipedia_file, 'r') as infile: 
        for line in infile: 
            contents = line.split('\t') 
            line_ID = int(contents[0])
            if line_ID not in corpora_left and line_ID not in corpora_right: 
                continue
            text = '\t'.join(contents[1:]).replace('-', 'xqxq').lower()
            tokens = btokenizer.tokenize(text)
            lines2tokens[line_ID] = tokens
            if line_ID in corpora_left: 
                # TODO: get scores for loov, excluding lines associated with that adj
                scores = [full_scores[t][0] for t in tokens if (t in full_scores and full_scores[t] != [0, 0])] # TODO: remove if statement
                left_line_scores[line_ID] = np.mean(scores)
            if line_ID in corpora_right: 
                scores = [full_scores[t][1] for t in tokens if (t in full_scores and full_scores[t] != [0, 0])] # TODO: remove if statement
                right_line_scores[line_ID] = np.mean(scores)
            
    top_left = left_line_scores.most_common(100)
    top_right = right_line_scores.most_common(100)
    for i, tup in enumerate(top_left): 
        print("LEFT", tup[1], tup[0], lines2tokens[tup[0]])
        if i > 10: break
            
    for i, tup in enumerate(top_right): 
        print("RIGHT", tup[1], tup[0], lines2tokens[tup[0]])
        if i > 10: break
        
    # for line on each axes side, taking account adj left out: calculate average npmi. {(line ID, adj) : average npmi}
    # keep only top 100 line ID for each side, save as {line ID: [(adj, axes)]}
    # for left out vector use the representation based on random contexts from earlier 
    
def inspect_npmi_scores(): 
    '''
    Examine npmi scores for different antonym pairs
    '''
    filenames = ['friendly.a.01.json', 'educated.a.01.json', 
                 'violent.a.01.json', 'favorable.a.01.json', 'colorful.a.01.json', 
                 'desirable.a.01.json']
    
    for filename in filenames: 
        npmi_scores_file = LOGS + 'wikipedia/npmi_scores/' + filename
        print(filename)
    
        with open(npmi_scores_file, 'r') as infile: 
            npmi_scores = json.load(infile)
            
        full_scores = npmi_scores['N/A'] 
        left_scores = Counter()
        for w in full_scores: 
            left_scores[w] = full_scores[w][0]
        print(left_scores.most_common(20))
        print()
        right_scores = Counter()
        for w in full_scores: 
            right_scores[w] = full_scores[w][1]
        print(right_scores.most_common(20))
        print()
    
def main(): 
    #get_adj_word_counts()
    #get_npmi_axes_contexts()
    get_high_npmi_lines()
    #inspect_npmi_scores()
    
if __name__ == '__main__':
    main()
