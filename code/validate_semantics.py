'''
This script does the following: 
- validate the separability of each axis based on LOOV
- apply axes to occupations and "person" 

Separate functions handle doing these things
for GloVe and BERT. 
'''
from collections import defaultdict
import json
from nltk.corpus import wordnet as wn
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import spatial, stats
import math
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
import re
from nltk import tokenize
import random
import itertools

#ROOT = '/global/scratch/users/lucy3_li/manosphere/'
ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
GLOVE = DATA + 'glove/'
LOGS = ROOT + 'logs/'
                
def load_wordnet_axes(): 
    '''
    Function used by many tasks for loading in the axes. 
    @output: 
    - axes: {synset : ([left pole adj], [right pole adj])}
    - axes_vocab: set of words in all poles
    '''
    axes = {}
    axes_vocab = set()
    with open(LOGS + 'semantics_val/wordnet_axes.txt', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            if len(contents) < 3: continue # no antonyms
            synset = contents[0]
            axis1 = contents[1].split(',')
            axis2 = contents[2].split(',')
            axes_vocab.update(axis1)
            axes_vocab.update(axis2)
            axes[synset] = (axis1, axis2)
    return axes, axes_vocab
    
def get_poles_glove(vec_dict, axes): 
    '''
    @input: 
    - vec_dict: dictionary of GloVe vectors 
    - axes: output of load_wordnet_axes, or {synset : ([left pole adj], [right pole adj])}
    @output: 
    - adj_poles: {synset : (matrix containing embeddings for left pole, matrix)}
    '''
    adj_poles = {} # synset : (right_vec, left_vec)
    for pole in sorted(axes.keys()): 
        left = axes[pole][0] # list of adj str
        left_vec = []
        for w in left: 
            if w in vec_dict: 
                left_vec.append(vec_dict[w])
        if len(left_vec) == 0: continue
        left_vec = np.array(left_vec)
        
        right = axes[pole][1] # list of adj str
        right_vec = []
        for w in right: 
            if w in vec_dict: 
                right_vec.append(vec_dict[w])
        if len(right_vec) == 0: continue
        right_vec = np.array(right_vec)
        
        adj_poles[pole] = (left_vec, right_vec)
    return adj_poles

def get_mean_std_glove(): 
    '''
    This is no longer used, because
    it did not make GloVe perform better. 
    '''
    all_vecs = []
    with open(GLOVE + 'glove.6B.300d.txt', 'r') as infile:
        for line in infile: 
            contents = line.split()
            word = contents[0]
            vec = np.array([float(i) for i in contents[1:]])
            all_vecs.append(vec)
    glove_mean = np.mean(all_vecs, axis=0)
    np.save(GLOVE + 'mean.npy', glove_mean)
    glove_std = np.std(all_vecs, axis=0)
    np.save(GLOVE + 'std.npy', glove_std)

def get_glove_vecs(vocab, axes_vocab, exp_name): 
    '''
    Get glove representations of a vocab, e.g. occupations
    @inputs: 
    - vocab: set of words, e.g. occupations
    - axes_vocab: set of adj
    - exp_name: str representing experiment name
    @output: 
    - dictionary from word or bigram to GloVe embedding
    '''
    if 'zscore' in exp_name: 
        glove_mean = np.load(GLOVE + 'mean.npy')
        glove_std = np.load(GLOVE + 'std.npy')
    bigram_tokens = set()
    for w in vocab: 
        tokens = w.split()
        if len(tokens) == 2: 
            bigram_tokens.update(tokens)
    glove_vecs = {}
    with open(GLOVE + 'glove.6B.300d.txt', 'r') as infile:
        for line in infile: 
            contents = line.split()
            word = contents[0]
            if word in vocab or word in axes_vocab or word in bigram_tokens: 
                vec = np.array([float(i) for i in contents[1:]])
                if 'zscore' in exp_name: 
                    vec = (vec - glove_mean) / glove_std
                glove_vecs[word] = vec
    # average representations for bigrams
    for w in vocab: 
        tokens = w.split()
        if len(tokens) == 2: 
            rep = []
            for tok in tokens: 
                if tok in glove_vecs: 
                    vec = glove_vecs[tok]
                    if 'zscore' in exp_name: 
                        vec = (vec - glove_mean) / glove_std
                    rep.append(vec)
            assert len(rep) == 2
            rep = np.mean(np.array(rep), axis=0)
            glove_vecs[w] = rep
    return glove_vecs

def save_frameaxis_inputs(file_path, sent_path, lexicon_name, exp_name): 
    '''
    For a lexicon that has scores for different words,
    saves vectors associated with that lexicon.
    @input: 
        - lexicon_dict: output of occupations() in setup_semantics.py
        - sent_dict: dictonary of sentences containing occupations
    @output: 
    - word matrix: npy file of GloVe vectors 
    - score_matrix: npy file of scores, 0 for low, 1 for high
    - vocab.txt: file of word strings in same order as npy files
    '''
    with open(file_path, 'r') as infile:
        lexicon_dict = json.load(infile)
    with open(sent_path, 'r') as infile:
        sent_dict = json.load(infile)
    vocab = set(sent_dict.keys())
    
    axes, axes_vocab = load_wordnet_axes()
    glove_vecs = get_glove_vecs(vocab, axes_vocab, exp_name)
    
    for c in lexicon_dict.keys(): 
        word_matrix = []
        score_matrix = []
        word_order = []
        for score in lexicon_dict[c]: 
            for word in lexicon_dict[c][score]: 
                if word not in vocab: continue
                word_matrix.append(glove_vecs[word])
                word_order.append(word)
                if score == 'high': 
                    score_matrix.append(1)
                elif score == 'low': 
                    score_matrix.append(0)
        score_matrix = np.array(score_matrix)
        word_matrix = np.array(word_matrix)
        np.save(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '_scores.npy', score_matrix)
        np.save(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '_words.npy', word_matrix)
        
        with open(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '_vocab.txt', 'w') as outfile: 
            for word in word_order: 
                outfile.write(word + '\n')

def frameaxis_helper(score_matrices, word_matrices, adj_poles, calc_pval=False): 
    '''
    called by frameaxis_bert() and frameaxis_glove()
    Calculates bias and also performs bootstrapping. 
    '''
    np.random.seed(0)
    biases = defaultdict(dict) # {c : { pole : (bias_sep, effect, bias1, bias2) } }
    for c in score_matrices: 
        score_matrix = score_matrices[c]
        word_matrix = word_matrices[c]
        
        for pole in tqdm(adj_poles): 
            left_vecs, right_vecs = adj_poles[pole]
            this_word_matrix = word_matrix
                
            microframe = left_vecs.mean(axis=0) - right_vecs.mean(axis=0)
            c_w_f = cosine_similarity(this_word_matrix, microframe.reshape(1, -1))
            c_w_f1 = c_w_f[score_matrix == 0] # all other occupations
            c_w_f2 = c_w_f[score_matrix == 1] # this occupation category
            b_t_f1 = np.mean(c_w_f1) # bias 
            b_t_f2 = np.mean(c_w_f2) # bias
            
            # calculate diff between these occupations' mean and population cosine sim
            # using bootstrap sample of all occupations of size c_w_f2.shape[0]
            if calc_pval: 
                # calculate statistical significance 
                random_samples = []
                random_std = []
                N = 1000
                for i in range(N): 
                    # bootstrap samples from everywhere
                    idx = np.random.choice(c_w_f.shape[0], size=c_w_f2.shape[0], replace=True)
                    sample = c_w_f[idx, :]
                    # calculate bias on sample
                    b_t_sample = np.mean(sample)
                    std_sample = np.std(sample)
                    random_samples.append(b_t_sample)
                    random_std.append(std_sample)
                t_stat, p_val = stats.ttest_1samp(random_samples, b_t_f2) # one sample t test 
                effect = b_t_f2 - np.mean(random_samples)
            else: 
                p_val = 0
                effect = 0
            biases[c][pole] = (p_val, effect, b_t_f1, b_t_f2)
    return biases

def load_inputs(file_path, lexicon_name): 
    '''
    Used by frameaxis_glove() to load occupation-related inputs. 
    '''
    with open(file_path, 'r') as infile:
        lexicon_dict = json.load(infile)
    score_matrices = {}
    word_matrices = {} 
    for c in lexicon_dict: 
        score_matrices[c] = np.load(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '_scores.npy')
        word_matrices[c] = np.load(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '_words.npy')

    return score_matrices, word_matrices
        
def frameaxis_glove(file_path, sent_path, lexicon_name, calc_pval=False, exp_name=''): 
    '''
    Need to call save_frameaxis_inputs() before running this function. 
    '''
    with open(file_path, 'r') as infile:
        lexicon_dict = json.load(infile)
    with open(sent_path, 'r') as infile:
        sent_dict = json.load(infile)
    vocab = set(sent_dict.keys())
    axes, axes_vocab = load_wordnet_axes()
    glove_vecs = get_glove_vecs(vocab, axes_vocab, exp_name)
    adj_poles = get_poles_glove(glove_vecs, axes)
    score_matrices, word_matrices = load_inputs(file_path, lexicon_name)
    
    biases = frameaxis_helper(score_matrices, word_matrices, adj_poles, calc_pval=calc_pval)
                
    with open(LOGS + 'semantics_val/' + lexicon_name + '/frameaxis_' + exp_name + '.json', 'w') as outfile:
        json.dump(biases, outfile)
        
def get_poles_bert(axes, exp_name): 
    assert 'bert' in exp_name
    
    adj_poles = {} # synset : (right_vec, left_vec)
    if exp_name in ['bert-default', 'bert-zscore']: 
        in_folder = LOGS + 'wikipedia/substitutes/bert-default/'
    elif 'sub' in exp_name: 
        if exp_name.startswith('bert-base-sub') and 'mask' not in exp_name: 
            in_folder = LOGS + 'wikipedia/substitutes/bert-base-sub/'
        elif exp_name.startswith('bert-base-sub') and 'mask' in exp_name: 
            in_folder = LOGS + 'wikipedia/substitutes/bert-base-sub-mask/'
        else: 
            in_folder = LOGS + 'wikipedia/substitutes/' + exp_name + '/'
    elif 'prob' in exp_name: 
        if exp_name.startswith('bert-base-prob'): 
            short_name = exp_name.replace('-zscore', '')
            in_folder = LOGS + 'wikipedia/substitutes/' + short_name + '/'
    with open(in_folder + 'word_rep_key.json', 'r') as infile: 
        word_rep_keys = json.load(infile)
    for pole in sorted(axes.keys()): 
        left = axes[pole][0] # list of words
        left_pole = pole + '_left'
        left_vec, _ = get_vecs_and_map(in_folder, left, left_pole, word_rep_keys, exp_name)
        right = axes[pole][1] # list of words
        right_pole = pole + '_right'
        right_vec, _ = get_vecs_and_map(in_folder, right, right_pole, word_rep_keys, exp_name)
        left_vec = np.array(left_vec)
        right_vec = np.array(right_vec)
        adj_poles[pole] = (left_vec, right_vec)
    return adj_poles
        
def frameaxis_bert(file_path, lexicon_name, exp_name='', calc_pval=False, normalize_person=True,
                   random_person=False): 
    '''
    Analous to frameaxis_glove(). 
    @inputs: 
    - calc_pval: whether to do bootstrapping for significance
    - random_person: whether to use randomly sampled person vectors or other occupations,
        intended for exp_name='person'
    '''
    print("running", exp_name)
    with open(file_path, 'r') as infile:
        lexicon_dict = json.load(infile)
    with open(LOGS + 'semantics_val/' + lexicon_name + '_BERT.json', 'r') as infile: 
        bert_vecs = json.load(infile)
    if 'zscore' in exp_name: 
        bert_mean = np.load(LOGS + 'wikipedia/mean_BERT.npy')
        bert_std = np.load(LOGS + 'wikipedia/std_BERT.npy')
        for vec in bert_vecs: 
            bert_vecs[vec] = (np.array(bert_vecs[vec]) - bert_mean) / bert_std
    else: 
        for vec in bert_vecs: 
            bert_vecs[vec] = np.array(bert_vecs[vec])
        
    axes, axes_vocab = load_wordnet_axes()
    print("getting poles...")
    adj_poles = get_poles_bert(axes, exp_name)
        
    print("getting matrices...")
    score_matrices = {}
    word_matrices = {} 
    for c in lexicon_dict.keys(): 
        word_matrix = []
        score_matrix = []
        word_order = []
        for score in lexicon_dict[c]: 
            for word in lexicon_dict[c][score]: 
                if word not in bert_vecs: continue
                if not random_person: 
                    word_matrix.append(bert_vecs[word])
                    word_order.append(word)
                    if score == 'high': 
                        score_matrix.append(1)
                    elif score == 'low': 
                        score_matrix.append(0)
                else: 
                    if score == 'high': 
                        word_matrix.append(bert_vecs[word])
                        word_order.append(word)
                        score_matrix.append(1)
        if random_person: 
            person_vecs = np.load(LOGS + 'semantics_val/person.npy')
            for i in range(person_vecs.shape[0]): 
                if 'zscore' in exp_name: 
                    word_matrix.append((person_vecs[i] - bert_mean) / bert_std)
                else: 
                    word_matrix.append(person_vecs[i])
                word_order.append('person')
                score_matrix.append(0)
        score_matrices[c] = np.array(score_matrix)
        word_matrices[c] = np.array(word_matrix)
        
    print("running frameaxis...")
    biases = frameaxis_helper(score_matrices, word_matrices, adj_poles, calc_pval=calc_pval)
    
    if random_person: 
        exp_name += '_randomp'
    with open(LOGS + 'semantics_val/' + lexicon_name + '/frameaxis_' + exp_name + '.json', 'w') as outfile:
        json.dump(biases, outfile)
        
def loo_val_helper(arr, left_vec, right_vec, exp_name):
    '''
    called by loo_val_glove() and loo_val_bert()
    Takes mean of each pole, creates microframe, and calculates
    cosine similarity of array. 
    '''
    left_pole = left_vec.mean(axis=0)
    right_pole = right_vec.mean(axis=0)
    microframe = right_pole - left_pole
    sim = 1 - spatial.distance.cosine(arr, microframe)
    if math.isnan(sim): print(microframe, arr, sim)
    return sim
        
def loo_val_glove(vec_dict, axes, exp_name): 
    '''
    leave-one-out validation where we calculate the simlarity of 
    one adjective to microframes 
    '''
    with open(LOGS + 'semantics_val/axes_quality_' + exp_name + '.txt', 'w') as outfile: 
        for pole in sorted(axes.keys()): 
            left = axes[pole][0] # list of words
            left_vec = [] # list of vectors 
            left_vocab = []
            for w in left: 
                if w in vec_dict: 
                    left_vec.append(vec_dict[w])
                    left_vocab.append(w)
            right = axes[pole][1]
            right_vec = [] # list of vectors 
            right_vocab = []
            for w in right: 
                if w in vec_dict: 
                    right_vec.append(vec_dict[w])
                    right_vocab.append(w)

            left_vec = np.ma.array(left_vec, mask=False)
            right_vec = np.ma.array(right_vec, mask=False)
            for i in range(left_vec.shape[0]): 
                mask = np.ones(left_vec.shape[0], dtype=bool)
                mask[i] = False
                new_left = left_vec[mask,:]
                arr = left_vec[i]
                sim = loo_val_helper(arr, new_left, right_vec, exp_name=exp_name)
                outfile.write(pole + '\t' + left_vocab[i] + '\t' + str(sim) + '\tleft\n')

            for i in range(right_vec.shape[0]): 
                mask = np.ones(right_vec.shape[0], dtype=bool)
                mask[i] = False
                new_right = right_vec[mask,:]
                arr = right_vec[i]
                sim = loo_val_helper(arr, left_vec, new_right, exp_name=exp_name)
                outfile.write(pole + '\t' + right_vocab[i]
                              + '\t' + str(sim) + '\tright\n')
                
def get_vecs_and_map(in_folder, side, side_pole, word_rep_keys, exp_name): 
    '''
    Gets vectors for one side of a synset. 
    @outputs
    side_vec: np.ma.array where each row is a vector
    rep_keys_map: { word : [indices that make up the word's aggregate vector] } 
    '''
    pole = side_pole.split('_')[0]
    if side_pole not in word_rep_keys: 
        # fall back on bert random
        in_folder = LOGS + 'wikipedia/substitutes/bert-default/'
        with open(in_folder + 'word_rep_key.json', 'r') as infile: 
            word_rep_keys = json.load(infile)
 
    rep_keys = word_rep_keys[side_pole] # [[line_num, word]]
    rep_keys_map = defaultdict(list) 
    for i, rk in enumerate(rep_keys): 
        line_num = rk[0]
        w = rk[1]
        rep_keys_map[w].append(i)
    side_vec = np.load(in_folder + side_pole + '.npy')
    if 'zscore' in exp_name: 
        bert_mean = np.load(LOGS + 'wikipedia/mean_BERT.npy')
        bert_std = np.load(LOGS + 'wikipedia/std_BERT.npy')
        side_vec = (side_vec - bert_mean) / bert_std
    side_vec = np.ma.array(side_vec, mask=False)
    return side_vec, rep_keys_map

def loo_val_bert(in_folder, axes, exp_name): 
    with open(in_folder + 'word_rep_key.json', 'r') as infile: 
        word_rep_keys = json.load(infile)
    with open(LOGS + 'semantics_val/axes_quality_' + exp_name + '.txt', 'w') as outfile: 
        for pole in tqdm(sorted(axes.keys())): 
            left = axes[pole][0] # list of words
            left_pole = pole + '_left'
            left_vec, lrep_keys_map = get_vecs_and_map(in_folder, left, left_pole, \
                                                               word_rep_keys, exp_name)
            
            right = axes[pole][1]
            right_pole = pole + '_right'
            right_vec, rrep_keys_map = get_vecs_and_map(in_folder, right, right_pole, \
                                                               word_rep_keys, exp_name)
            
            for w in lrep_keys_map: 
                idx = lrep_keys_map[w]
                mask = np.ones(left_vec.shape[0], dtype=bool)
                mask[idx] = False # mask out vectors corresponding to word
                new_left = left_vec[mask,:]
                if new_left.shape[0] == 0: continue
                arr = left_vec[idx,:].mean(axis=0) 
                sim = loo_val_helper(arr, new_left, right_vec, exp_name=exp_name)
                outfile.write(pole + '\t' + w + '\t' + str(sim) + '\tleft\n')
                
            for w in rrep_keys_map: 
                idx = rrep_keys_map[w]
                mask = np.ones(right_vec.shape[0], dtype=bool)
                mask[idx] = False
                new_right = right_vec[mask,:]
                if new_right.shape[0] == 0: continue
                arr = right_vec[idx,:].mean(axis=0)
                sim = loo_val_helper(arr, left_vec, new_right, exp_name=exp_name)
                outfile.write(pole + '\t' + w + '\t' + str(sim) + '\tright\n')

def check_separability(exp_name): 
    axes, axes_vocab = load_wordnet_axes()
    vocab = set()
    if exp_name in ['default', 'glove-zscore']: 
        vec_dict = get_glove_vecs(vocab, axes_vocab, exp_name)
        loo_val_glove(vec_dict, axes, exp_name)
        return
    if exp_name in ['bert-default', 'bert-zscore']: 
        in_folder = LOGS + 'wikipedia/substitutes/bert-default/'
        loo_val_bert(in_folder, axes, exp_name)
    elif 'sub' in exp_name and 'bert' in exp_name: 
        if exp_name.startswith('bert-base-sub') and 'mask' not in exp_name: 
            in_folder = LOGS + 'wikipedia/substitutes/bert-base-sub/'
        elif exp_name.startswith('bert-base-sub') and 'mask' in exp_name: 
            in_folder = LOGS + 'wikipedia/substitutes/bert-base-sub-mask/'
        else: 
            in_folder = LOGS + 'wikipedia/substitutes/' + exp_name + '/'
        loo_val_bert(in_folder, axes, exp_name)
    elif 'prob' in exp_name and 'bert' in exp_name: 
        if exp_name.startswith('bert-base-prob'): 
            short_name = exp_name.replace('-zscore', '')
            in_folder = LOGS + 'wikipedia/substitutes/' + short_name + '/'
        loo_val_bert(in_folder, axes, exp_name)
    
def main(): 
#     # ------ SEPARABILITY ------
#     check_separability('default')
#     check_separability('glove-zscore')
#     check_separability('bert-default')
#     check_separability('bert-zscore')
#     check_separability('bert-base-prob')
#     check_separability('bert-base-prob-zscore')
#     # ------ BERT OCCUPATIONS ------
#     frameaxis_bert(DATA + 'semantics/cleaned/occupations.json', 'occupations', exp_name='bert-default', calc_pval=True)
#     frameaxis_bert(DATA + 'semantics/cleaned/occupations.json', 'occupations', exp_name='bert-zscore', calc_pval=True)
#     frameaxis_bert(DATA + 'semantics/cleaned/occupations.json', 'occupations', exp_name='bert-base-prob', calc_pval=True)
#     frameaxis_bert(DATA + 'semantics/cleaned/occupations.json', 'occupations', 
#                    exp_name='bert-base-prob-zscore', calc_pval=True)
#   #  ------ BERT PERSON ------
#     frameaxis_bert(DATA + 'semantics/cleaned/occupations.json', 'person',
#                    exp_name='bert-default', calc_pval=True)
#     frameaxis_bert(DATA + 'semantics/cleaned/occupations.json', 'person',
#                    exp_name='bert-zscore', calc_pval=True)
#     frameaxis_bert(DATA + 'semantics/cleaned/occupations.json', 'person',
#                        exp_name='bert-base-prob', calc_pval=True)
    frameaxis_bert(DATA + 'semantics/cleaned/occupations.json', 'person',
                       exp_name='bert-base-prob-zscore', calc_pval=True) 
    # ------ GLOVE -------
#     save_frameaxis_inputs(DATA + 'semantics/cleaned/occupations.json', DATA + 'semantics/occupation_sents.json', 'occupations', exp_name='default')
#     frameaxis_glove(DATA + 'semantics/cleaned/occupations.json', DATA + 'semantics/occupation_sents.json', 
#                     'occupations', exp_name='default', calc_pval=True)

if __name__ == '__main__':
    main()
