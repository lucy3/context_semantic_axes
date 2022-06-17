"""
Applies axes to word embeddings
"""
from validate_semantics import load_wordnet_axes, get_poles_bert
from tqdm import tqdm
from scipy.spatial.distance import cosine
import json
import numpy as np
from collections import Counter, defaultdict
from fastdist import fastdist
from helpers import get_vocab
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from transformers import BasicTokenizer, BertTokenizerFast, BertModel, BertTokenizer
import os
import csv
import torch
import copy
import inflect
from nltk import tokenize
import sys

csv.field_size_limit(sys.maxsize)

ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
EMBED_PATH = LOGS + 'semantics_mano/embed/'
AGG_EMBED_PATH = LOGS + 'semantics_mano/agg_embed/'
VARIANT_OUT = LOGS + 'semantics_mano/variant_scores/'
WOMEN_OUT = LOGS + 'semantics_mano/women_scores/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_manosphere_vecs(inpath): 
    '''
    Load z-scored embeddings for each vocabulary term
    '''
    bert_mean = np.load(LOGS + 'wikipedia/mean_BERT.npy')
    bert_std = np.load(LOGS + 'wikipedia/std_BERT.npy')
    
    vocab_order = []
    full_reps = []
    with open(inpath, 'r') as infile: 
        d = json.load(infile) # {term : vector}
    for term in sorted(d.keys()): 
        standard_vec = (np.array(d[term]) - bert_mean) / bert_std
        vocab_order.append(term)
        full_reps.append(standard_vec)
    full_reps = np.array(full_reps)
    print("Number of reps", full_reps.shape)
    return full_reps, vocab_order

def get_good_axes(zscore=True): 
    '''
    This is copied from axes_occupation_viz.ipynb. 
    '''
    if zscore: 
        quality_file_path = LOGS + 'semantics_val/axes_quality_bert-base-prob-zscore.txt'
    else: 
        quality_file_path = LOGS + 'semantics_val/axes_quality_bert-base-prob.txt'
    scores = defaultdict(dict) # {synset: {word : (predicted, true)}}
    with open(quality_file_path, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            scores[contents[0]][contents[1]] = (float(contents[2]), contents[3])
    avg_scores = Counter()
    good_synsets = set()
    for synset in scores: 
        left_scores = []
        right_scores = []
        for w in scores[synset]: 
            if scores[synset][w][1] == 'left': 
                left_scores.append(-1*scores[synset][w][0])
            else: 
                right_scores.append(scores[synset][w][0])
        if left_scores != []: 
            # some are empty since they only had one word with reps
            avg_scores[synset + '_left'] = np.mean(left_scores) 
        if right_scores != []: 
            avg_scores[synset + '_right'] = np.mean(right_scores) 
        if avg_scores[synset + '_left'] >= 0 and avg_scores[synset + '_right'] >= 0: 
            good_synsets.add(synset)
    return good_synsets

def project_onto_axes(): 
    '''
    The output is a dictionary of axis: list of scores, in order of full_reps
    '''
    print("getting axes...")
    axes, axes_vocab = load_wordnet_axes()
    # synset : (right_vec, left_vec)
    adj_poles = get_poles_bert(axes, 'bert-base-prob-zscore')
    good_axes = get_good_axes()
    
    print("getting word vectors...")
    full_reps, vocab_order = load_manosphere_vecs(AGG_EMBED_PATH + 'mano_overall.json')
    
    print("calculating bias of every word to every axis...")
    scores = defaultdict(list) 
    for pole in tqdm(adj_poles): 
        if pole not in good_axes: continue
        left_vecs, right_vecs = adj_poles[pole]
        left_pole = left_vecs.mean(axis=0)
        right_pole = right_vecs.mean(axis=0)
        microframe = right_pole - left_pole
        # note that this is cosine distance, not cosine similarity
        c_w_f = fastdist.vector_to_matrix_distance(microframe, full_reps, fastdist.cosine, "cosine")
        scores[pole] = list(c_w_f)
        
    with open(LOGS + 'semantics_mano/results/scores.json', 'w') as outfile: 
        json.dump(scores, outfile)
        
    with open(LOGS + 'semantics_mano/results/vocab_order.txt', 'w') as outfile: 
        outfile.write('\n'.join(vocab_order))
            
def get_overall_embeddings(): 
    '''
    Reaggregates based on per-year, per-community/platform
    embeddings and their counts
    This way each vocab word has one embedding. 
    '''
    total_count = Counter() # {term : count}
    overall_vec = {}
    
    # go through reddit
    years = range(2008, 2020)
    for y in tqdm(years):
        with open(EMBED_PATH + 'reddit_' + str(y) + '.json', 'r') as infile: 
            d = json.load(infile) # { term_category_year : vector }
        with open(EMBED_PATH + 'reddit_' + str(y) + '_wordcounts.json', 'r') as infile: 
            word_counts = json.load(infile)
        for key in sorted(d.keys()): 
            count = word_counts[key]
            parts = key.split('_')
            term = parts[0]
            vec = np.array(d[key])*count
            total_count[term] += count
            if term not in overall_vec: 
                overall_vec[term] = np.zeros(3072)
            overall_vec[term] += vec
    forums = ['avfm', 'mgtow', 'incels', 'pua_forum', 'red_pill_talk', 'rooshv', 'the_attraction']
    # go through forum 
    for f in tqdm(forums): 
        with open(EMBED_PATH + 'forum_' + f + '.json', 'r') as infile: 
            d = json.load(infile) # { term_category_year : vector }
        with open(EMBED_PATH + 'forum_' + f + '_wordcounts.json', 'r') as infile: 
            word_counts = json.load(infile)
        for key in sorted(d.keys()): 
            count = word_counts[key]
            parts = key.split('_')
            term = parts[0]
            vec = np.array(d[key])*count
            total_count[term] += count
            if term not in overall_vec: 
                overall_vec[term] = np.zeros(3072)
            overall_vec[term] += vec
    
    for term in overall_vec: 
        overall_vec[term] = list(overall_vec[term] / total_count[term]) 
    with open(AGG_EMBED_PATH + 'mano_overall.json', 'w') as outfile: 
        json.dump(overall_vec, outfile)
        
def get_yearly_embeddings(): 
    '''
    Reaggregates based on per-year, per-community/platform
    embeddings and their counts
    This way each vocab word has one embedding. 
    '''
    total_count = Counter() # {term + year : count}
    overall_vec = {}
    
    # go through reddit
    years = range(2008, 2020)
    for y in tqdm(years):
        with open(EMBED_PATH + 'reddit_' + str(y) + '.json', 'r') as infile: 
            d = json.load(infile) # { term_category_year : vector }
        with open(EMBED_PATH + 'reddit_' + str(y) + '_wordcounts.json', 'r') as infile: 
            word_counts = json.load(infile)
        for key in sorted(d.keys()): 
            count = word_counts[key]
            parts = key.split('_')
            term = parts[0]
            vec = np.array(d[key])*count
            total_count[term + '_' + str(y)] += count
            if term not in overall_vec: 
                overall_vec[term + '_' + str(y)] = np.zeros(3072)
            overall_vec[term + '_' + str(y)] += vec
    forums = ['avfm', 'mgtow', 'incels', 'pua_forum', 'red_pill_talk', 'rooshv', 'the_attraction']
    # go through forum 
    for f in tqdm(forums): 
        with open(EMBED_PATH + 'forum_' + f + '.json', 'r') as infile: 
            d = json.load(infile) # { term_category_year : vector }
        with open(EMBED_PATH + 'forum_' + f + '_wordcounts.json', 'r') as infile: 
            word_counts = json.load(infile)
        for key in sorted(d.keys()): 
            count = word_counts[key]
            parts = key.split('_')
            term = parts[0]
            y = parts[-1]
            if y == 'None': continue
            vec = np.array(d[key])*count
            total_count[term + '_' + str(y)] += count
            if term not in overall_vec: 
                overall_vec[term + '_' + str(y)] = np.zeros(3072)
            overall_vec[term + '_' + str(y)] += vec
    
    for term in overall_vec: 
        overall_vec[term] = list(overall_vec[term] / total_count[term]) 
    with open(AGG_EMBED_PATH + 'mano_yearly.json', 'w') as outfile: 
        json.dump(overall_vec, outfile)
        
def batch_data(): 
    vocab = ['moids', 'femoids', 'foids', 'women', 'men']
    tokenizer = BasicTokenizer(do_lower_case=True)

    batch_size = 8
    batch_sentences = [] # each item is a list
    batch_words = [] # each item is a list
    batch_meta = []
    
    curr_batch = []
    curr_words = []
    curr_meta = []
    
    VAR_DIR = LOGS + 'variants/'
    for json_file in os.listdir(VAR_DIR):
        if not json_file.endswith('_id2sent.json'):  continue
        print(json_file)
        id2sent_file = VAR_DIR + json_file
        word2id_file = VAR_DIR + json_file.replace('_id2sent.json', '_word2id.json')
        with open(id2sent_file, 'r') as infile: 
            id2sent = json.load(infile)
        with open(word2id_file, 'r') as infile: 
            word2id = json.load(infile)

        sentID_word = defaultdict(list) # {sentID : [terms in line]}
        sentID_meta = defaultdict(str) # {sentID : year_category}
        for key in word2id: 
            contents = key.split('_')
            meta = '_'.join(contents[1:])
            word = contents[0]
            for sentID in word2id[key]: 
                sentID_word[sentID].append(word)
                sentID_meta[sentID] = meta
    
        for sentID in tqdm(id2sent): 
            sent = id2sent[sentID]
            tokens = tokenizer.tokenize(sent)
            meta = sentID_meta[sentID]
            words = sentID_word[sentID]
            for w in words: 
                idx = tokens.index(w)
                replaced_tokens = copy.deepcopy(tokens)
                replaced_tokens[idx] = 'people'
                curr_batch.append(replaced_tokens)
                curr_words.append((w, idx))
                curr_meta.append(meta)
                if len(curr_batch) == batch_size: 
                    batch_sentences.append(curr_batch)
                    batch_words.append(curr_words)
                    batch_meta.append(curr_meta)
                    curr_batch = []
                    curr_words = []   
                    curr_meta = []
        if len(curr_batch) != 0: # fence post
            batch_sentences.append(curr_batch)
            batch_words.append(curr_words)
            batch_meta.append(curr_meta)
    return batch_sentences, batch_words, batch_meta

def get_microframe_matrix(zscore=True): 
    axes, axes_vocab = load_wordnet_axes()
    # synset : (right_vec, left_vec)
    if zscore: 
        adj_poles = get_poles_bert(axes, 'bert-base-prob-zscore')
    else: 
        adj_poles = get_poles_bert(axes, 'bert-base-prob')
    good_axes = get_good_axes(zscore=zscore)
    m = []
    pole_order = []
    for pole in sorted(adj_poles.keys()): 
        if pole not in good_axes: continue
        pole_order.append(pole)
        left_vecs, right_vecs = adj_poles[pole]
        left_pole = left_vecs.mean(axis=0)
        right_pole = right_vecs.mean(axis=0)
        microframe = right_pole - left_pole
        m.append(microframe)
    if zscore: 
        variant_outpath = VARIANT_OUT + 'pole_order.txt'
    else: 
        variant_outpath = VARIANT_OUT + 'pole_order_noz.txt'
    with open(variant_outpath, 'w') as outfile: 
        for pole in pole_order:
            outfile.write(pole + '\n')
    return np.array(m)

def get_bert_embeddings(batch_sentences, batch_words, batch_meta, bert_mean, bert_std, m, zscore=True): 
    word_reps = defaultdict(list) # {word : [[axis scores for each occurrence]]} 
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    layers = [-4, -3, -2, -1] # last four layers
    model.to(device)
    model.eval()
    
    for i, batch in enumerate(tqdm(batch_sentences)): # for every batch
        encoded_inputs = tokenizer(batch, is_split_into_words=True, padding=True, truncation=True, 
             return_tensors="pt")
        encoded_inputs.to(device)
        outputs = model(**encoded_inputs, output_hidden_states=True)
        states = outputs.hidden_states # tuple
        # batch_size x seq_len x 3072
        vector = torch.cat([states[i] for i in layers], 2) # concatenate last four
        for j in range(len(batch)): # for every example
            word_ids = encoded_inputs.word_ids(j)
            word_tokenids = defaultdict(list) # {word : [token ids]}
            for k, word_id in enumerate(word_ids): # for every token
                target_word_id = batch_words[i][j][1]
                target_word = batch_words[i][j][0]
                if word_id is not None and word_id == target_word_id: 
                    word_tokenids[target_word].append(k)
            for word in word_tokenids: 
                token_ids_word = np.array(word_tokenids[word]) 
                word_embed = vector[j][token_ids_word]
                word_embed = word_embed.mean(dim=0).detach().cpu().numpy() # average word pieces
                if np.isnan(word_embed).any(): 
                    print("PROBLEM!!!", word, batch[j])
                    return 
                word_cat = word + '_' + batch_meta[i][j]
                if zscore: 
                    word_embed = (word_embed - bert_mean) / bert_std # z-score
                else: 
                    word_embed = np.float64(word_embed)
                    m = np.float64(m)
                word_scores = fastdist.vector_to_matrix_distance(word_embed, m, fastdist.cosine, "cosine")
                word_reps[word_cat].append(list(word_scores))
                
        torch.cuda.empty_cache()
        
    return word_reps
        
def get_axes_scores_variants(): 
    '''
    for each occurrence of men, women, foids, femoids, moids, 
    replace them with "people" and calculate the axis scores of
    that embedding and save the scores
    
    'Foids' and 'foid' are wordpieces.
    '''
    print("batching...")
    batch_sentences, batch_words, batch_meta = batch_data()
    print("NUMBER OF BATCHES:", len(batch_sentences))
    
    bert_mean = np.load(LOGS + 'wikipedia/mean_BERT.npy')
    bert_std = np.load(LOGS + 'wikipedia/std_BERT.npy')
    
    print("getting microframe matrix...")
    m = get_microframe_matrix()
    
    word_reps = get_bert_embeddings(batch_sentences, batch_words, batch_meta, bert_mean, bert_std, m)
        
    with open(VARIANT_OUT + 'scores.json', 'w') as outfile: 
        json.dump(word_reps, outfile)
        
def batch_data_domains(replace=False): 
    tokenizer = BasicTokenizer(do_lower_case=True)
    vocab = set(['feminists', 'women', 'girls', 'females'])
    batch_size = 8
    batch_sentences = [] # each item is a list
    batch_words = [] # each item is a list
    batch_meta = []
    curr_batch = []
    curr_words = []
    curr_meta = []
    
    with open(LOGS + 'wikipedia/women_data/part-00000', 'r') as infile: 
        print("going through wikipedia...")
        for line in infile: 
            contents = line.split('\t')
            text = '\t'.join(contents[1:])
            tokens = tokenizer.tokenize(text)
            overlap = set(tokens) & vocab
            for w in overlap: 
                # "women" tends to appear in additional sentences
                if len(overlap) > 1 and w == 'women': continue 
                idx = tokens.index(w)
                if replace: 
                    replaced_tokens = copy.deepcopy(tokens)
                    replaced_tokens[idx] = 'people'
                    curr_batch.append(replaced_tokens)
                else: 
                    curr_batch.append(tokens)
                curr_words.append((w, idx))
                curr_meta.append('wikipedia')
                if len(curr_batch) == batch_size: 
                    batch_sentences.append(curr_batch)
                    batch_words.append(curr_words)
                    batch_meta.append(curr_meta)
                    curr_batch = []
                    curr_words = []   
                    curr_meta = []
    with open(LOGS + 'women_control_sample.csv', 'r') as infile: 
        print("going through control...")
        reader = csv.reader(infile, delimiter='\t')
        for row in reader: 
            w = row[0]
            sents = tokenize.sent_tokenize(row[4])
            for sent in sents: 
                tokens = tokenizer.tokenize(sent)
                if w in tokens: 
                    idx = tokens.index(w)
                    if replace: 
                        replaced_tokens = copy.deepcopy(tokens)
                        replaced_tokens[idx] = 'people'
                        curr_batch.append(replaced_tokens)
                    else: 
                        curr_batch.append(tokens)
                    curr_words.append((w, idx))
                    curr_meta.append('control')
                    if len(curr_batch) == batch_size: 
                        batch_sentences.append(curr_batch)
                        batch_words.append(curr_words)
                        batch_meta.append(curr_meta)
                        curr_batch = []
                        curr_words = []   
                        curr_meta = []
                    break
    with open(LOGS + 'women_extreme_sample.csv', 'r') as infile: 
        print("going through extreme...")
        reader = csv.reader(infile, delimiter='\t')
        for row in reader: 
            w = row[0]
            sents = tokenize.sent_tokenize(row[4])
            for sent in sents: 
                tokens = tokenizer.tokenize(sent)
                if w in tokens: 
                    idx = tokens.index(w)
                    if replace: 
                        replaced_tokens = copy.deepcopy(tokens)
                        replaced_tokens[idx] = 'people'
                        curr_batch.append(replaced_tokens)
                    else: 
                        curr_batch.append(tokens)
                    curr_words.append((w, idx))
                    curr_meta.append('extreme')
                    if len(curr_batch) == batch_size: 
                        batch_sentences.append(curr_batch)
                        batch_words.append(curr_words)
                        batch_meta.append(curr_meta)
                        curr_batch = []
                        curr_words = []   
                        curr_meta = []
                    break            
    if len(curr_batch) != 0: # fence post
        batch_sentences.append(curr_batch)
        batch_words.append(curr_words)
        batch_meta.append(curr_meta)
    return batch_sentences, batch_words, batch_meta
        
def get_axes_scores_domains(replace=False, zscore=True): 
    '''
    Apply axes on occurrences of women, feminists, girls, and females
    across domains. 
    '''
    print("batching...")
    batch_sentences, batch_words, batch_meta = batch_data_domains(replace=replace)
    
    bert_mean = np.load(LOGS + 'wikipedia/mean_BERT.npy')
    bert_std = np.load(LOGS + 'wikipedia/std_BERT.npy')
    
    print("getting microframe matrix...")
    m = get_microframe_matrix(zscore=zscore)
    
    word_reps = get_bert_embeddings(batch_sentences, batch_words, batch_meta, bert_mean, bert_std, m, zscore=zscore)
        
    if zscore: 
        with open(WOMEN_OUT + str(replace) + '_scores.json', 'w') as outfile: 
            json.dump(word_reps, outfile)
    else: 
        with open(WOMEN_OUT + str(replace) + '_scores_noz.json', 'w') as outfile: 
            json.dump(word_reps, outfile)
        
def batch_data_time(replace=True): 
    p_cache = {} # words that have already been through inflect 
    
    p = inflect.engine()
    tokenizer = BasicTokenizer(do_lower_case=True)
    batch_size = 8
    batch_sentences = [] # each item is a list
    batch_words = [] # each item is a list
    batch_meta = []
    curr_batch = []
    curr_words = []
    curr_meta = []
    with open(LOGS + 'women_control_sample_time.csv', 'r') as infile: 
        print("going through control...")
        reader = csv.reader(infile, delimiter='\t')
        for row in reader: 
            month = row[0]
            line_num = row[1]
            w = row[2]
            if w not in p_cache: 
                if p.singular_noun(w): # returns True, so plural
                    p_cache[w] = 'people'
                else: 
                    p_cache[w] = 'person' 
            replacement = p_cache[w]
            sents = tokenize.sent_tokenize(row[4])
            for sent in sents: 
                tokens = tokenizer.tokenize(sent)
                if w in tokens: 
                    idx = tokens.index(w)
                    if replace: 
                        replaced_tokens = copy.deepcopy(tokens)
                        replaced_tokens[idx] = replacement
                        curr_batch.append(replaced_tokens)
                    else: 
                        curr_batch.append(tokens)
                    curr_words.append((w, idx))
                    curr_meta.append('control_' + month + '_' + replacement + '_' + line_num)
                    if len(curr_batch) == batch_size: 
                        batch_sentences.append(curr_batch)
                        batch_words.append(curr_words)
                        batch_meta.append(curr_meta)
                        curr_batch = []
                        curr_words = []   
                        curr_meta = []
                    break
    with open(LOGS + 'women_extreme_sample_time.csv', 'r') as infile: 
        print("going through extreme...")
        reader = csv.reader(infile, delimiter='\t')
        for row in reader: 
            month = row[0]
            line_num = row[1]
            w = row[2]
            if w not in p_cache: 
                if p.singular_noun(w): # returns True, so plural
                    p_cache[w] = 'people'
                else: 
                    p_cache[w] = 'person' 
            replacement = p_cache[w]
            sents = tokenize.sent_tokenize(row[4])
            for sent in sents: 
                tokens = tokenizer.tokenize(sent)
                if w in tokens: 
                    idx = tokens.index(w)
                    if replace: 
                        replaced_tokens = copy.deepcopy(tokens)
                        replaced_tokens[idx] = replacement
                        curr_batch.append(replaced_tokens)
                    else: 
                        curr_batch.append(tokens)
                    curr_words.append((w, idx))
                    curr_meta.append('extreme_' + month + '_' + replacement + '_' + line_num)
                    if len(curr_batch) == batch_size: 
                        batch_sentences.append(curr_batch)
                        batch_words.append(curr_words)
                        batch_meta.append(curr_meta)
                        curr_batch = []
                        curr_words = []   
                        curr_meta = []
                    break            
    if len(curr_batch) != 0: # fence post
        batch_sentences.append(curr_batch)
        batch_words.append(curr_words)
        batch_meta.append(curr_meta)
    return batch_sentences, batch_words, batch_meta
        
def get_axes_scores_over_time(replace=True): 
    print("batching...")
    batch_sentences, batch_words, batch_meta = batch_data_time(replace=replace)
    
    bert_mean = np.load(LOGS + 'wikipedia/mean_BERT.npy')
    bert_std = np.load(LOGS + 'wikipedia/std_BERT.npy')
    
    print("getting microframe matrix...")
    m = get_microframe_matrix()
    
    word_reps = get_bert_embeddings(batch_sentences, batch_words, batch_meta, bert_mean, bert_std, m)
    
    with open(WOMEN_OUT + str(replace) + '_time_scores.json', 'w') as outfile: 
        json.dump(word_reps, outfile)

def main(): 
    #get_overall_embeddings()
    #project_onto_axes() 
    #get_axes_scores_variants()
    #get_axes_scores_domains(replace=False)
    #get_axes_scores_domains(replace=True)
    #get_axes_scores_domains(replace=False, zscore=False)
    #get_axes_scores_domains(replace=True, zscore=False)
    get_axes_scores_over_time(replace=True)

if __name__ == '__main__':
    main()
