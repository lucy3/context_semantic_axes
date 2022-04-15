"""
For getting BERT embeddings of key words (occupations
and adjectives) from Wikipedia. 

Adjectives: 
- sample_wikipedia() gets a sample of sentences 
  containing adjectives 
- get_axes_contexts() to get contexts for BERT-default
- get_adj_embeddings() based on different subsets of contexts

Occupations: 
- get_occupation_embeddings()

BERT mean and std
- get_bert_mean_std()
"""
import requests
import json
from tqdm import tqdm
from transformers import BasicTokenizer, BertTokenizerFast, BertModel, BertTokenizer
#from pyspark import SparkConf, SparkContext
#from pyspark.sql import Row, SQLContext
from functools import partial
from collections import Counter, defaultdict
import random
import torch
from nltk import tokenize
import numpy as np
import os
import copy

ROOT = '/global/scratch/users/lucy3_li/manosphere/'
#ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------
# Adjective functions
# --------------

def get_adj(): 
    '''
    Read in adjectives in WordNet axes 
    @input: 
    - wordnet_axes.txt, or output of setup_semantics.py
    '''
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
    return axes_vocab

def get_content_lines(line): 
    '''
    There are a lot of if statements here
    because I want to return False as fast as possible. 
    '''
    # only get wikitext content
    line = line.strip()
    if len(line) < 10:
        return False
    line_content = line.startswith('<doc') or line.startswith('</doc')
    if line_content: 
        return False
    return True
    
def get_sentences(line): 
    # used by sample_wikipedia()
    sents = tokenize.sent_tokenize(line.strip())
    return sents

def contains_vocab(tup, tokenizer=None, vocab=set()): 
    '''
    Input: [(line, line_id)]
    Output: [(vocab_token1, line_id), (vocab_token2, line_id)]
    
    Since BERT was originally made for sentence level tasks, we split Wikipedia into sentences,
    and for efficiency, keep those that are > 10 words and < 150 words long.
    '''
    line, line_id = tup
    line = line.replace('-', 'xqxq')
    tokens = tokenizer.tokenize(line)
    if len(tokens) < 10 or len(tokens) > 150: 
        return []
    overlap = set(tokens) & vocab
    ret = []
    for w in overlap: 
        ret.append((w, line_id))
    return ret

def exact_sample(tup): 
    # used by sample_wikipedia()
    w = tup[0]
    occur = tup[1]
    if len(occur) < 1000: 
        return tup
    else: 
        return (tup[0], random.sample(occur, 1000))
            
def sample_wikipedia(): 
    '''
    Finds occurrences of vocab words in wikipedia. 
    
    Note: the adjectives in "adj_lines.json" have "xqxq"
    instead of "-" to account for dashed words for tokenization. 
    '''
    vocab = get_adj()
    vocab_name = 'adj'
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    
    new_vocab = set()
    for w in vocab: # helps handle dashed words
        if '-' in w: 
            new_vocab.add(w.replace('-', 'xqxq'))
        else: 
            new_vocab.add(w)

    wikipedia_file = '/mnt/data0/corpora/wikipedia/text/all_files.txt'
    #wikipedia_file = '/mnt/data0/corpora/wikipedia/small_wiki'
    tokenizer = BasicTokenizer(do_lower_case=True)
    data = sc.textFile(wikipedia_file).filter(get_content_lines)
    data = data.flatMap(get_sentences).filter(lambda sent: len(sent.split()) > 10)
    data = data.zipWithUniqueId() 
    token_data = data.flatMap(partial(contains_vocab, tokenizer=tokenizer, vocab=new_vocab))
    token_counts = token_data.map(lambda tup: (tup[0], 1)).reduceByKey(lambda n1, n2: n1 + n2)
    fractions = token_counts.map(lambda tup: (tup[0], min(1.0, 5000.0/tup[1]))).collectAsMap() 
    token_data = token_data.sampleByKey(False, fractions) # approx sample before exact sample
    token_data = token_data.groupByKey().mapValues(list).map(exact_sample).collectAsMap()
    with open(LOGS + 'wikipedia/' + vocab_name + '_lines.json', 'w') as outfile: 
        json.dump(token_data, outfile)
    line_ids_to_keep = set()
    for token in token_data: 
        line_ids_to_keep.update(token_data[token])
    data = data.filter(lambda tup: tup[1] in line_ids_to_keep).map(lambda tup: str(tup[1]) + '\t' + tup[0]) 
    data.coalesce(1).saveAsTextFile(LOGS + 'wikipedia/' + vocab_name + '_data')
    sc.stop()
    
def count_axes(): 
    '''
    This counts the number of sentences
    we got from Wikipedia that could possibly be used
    to represent each pole. 
    '''
    with open(LOGS + 'wikipedia/adj_lines.json', 'r') as infile: 
        adj_lines = json.load(infile)
    total_count = Counter()
    for adj in adj_lines: 
        total_count[adj] = len(adj_lines[adj])
        
    synset_counts = Counter()
    min_count = float("inf")
    max_count = 0
    with open(LOGS + 'semantics_val/wordnet_axes.txt', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            synset = contents[0]
            axis1 = contents[1].split(',')
            axis1_count = sum([total_count[w] for w in axis1])
            axis2 = contents[2].split(',')
            axis2_count = sum([total_count[w] for w in axis2])
            synset_counts[synset] = [axis1_count, axis2_count]
            min_count = min([axis1_count, axis2_count, min_count])
            max_count = max([axis1_count, axis2_count, max_count])
            
    print(min_count, max_count)
            
    with open(LOGS + 'wikipedia/axes_counts.json', 'w') as outfile: 
        json.dump(synset_counts, outfile)
    
def sample_random_contexts(ss, axis, adj_lines): 
    '''
    Adj that have '-' in adj_lines are replaced
    with xqxq so in order to match them to the adj
    in wordnet_axes, we need to replace and then compare.
    '''
    axis_lines = set()
    for adj in axis:
        a = adj.replace('-', 'xqxq') 
        for line in adj_lines[a]: 
            axis_lines.add((ss, adj, line))
    axis_lines = random.sample(axis_lines, 100)
    return axis_lines

def get_axes_contexts(): 
    '''
    This function gets random contexts of each adjective. 
    Inputs: 
        - adj_lines.json: adjectives to lines in wikipedia
        - wordnet_axes.txt: axes to adjectives
    Output: 
        - a dictionary from line number to adj in line
    '''
    with open(LOGS + 'wikipedia/adj_lines.json', 'r') as infile: 
        adj_lines = json.load(infile)
    
    ret = defaultdict(list) # {line_num: [(adj, synset)]}
    with open(LOGS + 'semantics_val/wordnet_axes.txt', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t') 
            synset = contents[0]
            axis1 = contents[1].split(',')
            axis1_lines = sample_random_contexts(synset + '_left', axis1, adj_lines)
            for tup in axis1_lines: 
                ret[tup[2]].append([tup[1], tup[0]])
            axis2 = contents[2].split(',')
            axis2_lines = sample_random_contexts(synset + '_right', axis2, adj_lines)
            for tup in axis2_lines: 
                ret[tup[2]].append([tup[1], tup[0]])
    with open(LOGS + 'wikipedia/adj_lines_random.json', 'w') as outfile: 
        json.dump(ret, outfile) 
        
def batch_adj_data(input_json, exp_name): 
    '''
    Batches data for get_adj_embeddings()
    '''
    with open(input_json, 'r') as infile:
        lines_tokens = json.load(infile) # {line_num: [(adj, synset)]}
    batch_size = 8
    batch_sentences = [] # each item is a list
    batch_words = [] # each item is a list
    batch_tups = []
    curr_batch = []
    curr_words = []
    curr_tups = []
    btokenizer = BasicTokenizer(do_lower_case=True)
    vocab = set()
    with open(LOGS + 'wikipedia/adj_data/part-00000', 'r') as infile: 
        for line in infile:
            contents = line.split('\t')
            line_num = contents[0]
            if line_num not in lines_tokens: continue
            text = '\t'.join(contents[1:]).lower()
            words_in_line = [t[0] for t in lines_tokens[line_num]]
            tups_in_line = [(line_num, t[0], t[1]) for t in lines_tokens[line_num]]
            vocab.update(tups_in_line)
            dashed_words = set()
            for w in words_in_line: 
                if '-' in w: 
                    new_w = w.replace('-', 'xqxq')
                    text = text.replace(w, new_w)
                    dashed_words.add(new_w)
            tokens = btokenizer.tokenize(text)
            if 'mask' in exp_name: 
                # replace target word with [MASK]
                for i in range(len(tokens)): 
                    if tokens[i] in dashed_words or tokens[i] in words_in_line: 
                        new_tokens = copy.deepcopy(tokens)
                        new_tokens[i] = '[MASK]'
                        curr_batch.append(new_tokens)
                        curr_words.append(words_in_line)
                        curr_tups.append(tups_in_line)
                        if len(curr_batch) == batch_size: 
                            batch_sentences.append(curr_batch)
                            batch_words.append(curr_words)
                            batch_tups.append(curr_tups)
                            curr_batch = []
                            curr_words = []
                            curr_tups = []                
            else: 
                for i in range(len(tokens)): 
                    if tokens[i] in dashed_words:
                        tokens[i] = tokens[i].replace('xqxq', '-')
                curr_batch.append(tokens)
                curr_words.append(words_in_line)
                curr_tups.append(tups_in_line)
                if len(curr_batch) == batch_size: 
                    batch_sentences.append(curr_batch)
                    batch_words.append(curr_words)
                    batch_tups.append(curr_tups)
                    curr_batch = []
                    curr_words = []
                    curr_tups = []
        if len(curr_batch) != 0: # fence post
            batch_sentences.append(curr_batch)
            batch_words.append(curr_words)
            batch_tups.append(curr_tups)
    return batch_sentences, batch_words, batch_tups, vocab

def get_adj_embeddings(exp_name, save_agg=True): 
    '''
    This function gets embeddings for adjectives in Wikipedia
    This is slightly messy because adjectives in Wordnet 
    can be multiple words. 
    The boolean save_agg saves aggregate word embeddings (averaged
    for each word in each synset), or it stacks each individual 
    vector into an array (more memory intensive). 
    '''
    if exp_name.startswith('bert-base-sub'): 
        input_json = LOGS + 'wikipedia/adj_lines_base-substitutes.json'
    elif exp_name.startswith('bert-base-prob'): 
        input_json = LOGS + 'wikipedia/adj_lines_base-probs.json'
    elif exp_name.startswith('bert-large-sub'): 
        input_json = LOGS + 'wikipedia/adj_lines_large-substitutes.json'
    elif exp_name == 'bert-default': 
        input_json = LOGS + 'wikipedia/adj_lines_random.json'
    print("Batching contexts...")
    batch_sentences, batch_words, batch_tups, vocab = batch_adj_data(input_json, exp_name)
    print("Getting model...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    layers = [-4, -3, -2, -1] # last four layers
    model.to(device)
    model.eval()
    # initialize word representations
    if save_agg: 
        word_reps = {}
        for tup in vocab: 
            word_reps[tup] = np.zeros(3072)
        word_counts = Counter()
    else: 
        word_reps = defaultdict(list) # {ss : [numpy arrays]}
        # {ss : [(line_num, adj)]} where index corresponds to vector in word_reps 
        word_rep_keys = defaultdict(list) 
    for i, batch in enumerate(tqdm(batch_sentences)): # for every batch
        word_tokenids = {} # { j : { word : [token ids] } }
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
                if word_id is not None: 
                    curr_word = batch[j][word_id]
                    if 'mask' in exp_name: 
                        if curr_word == '[MASK]': 
                            word_tokenids[curr_word].append(k)
                    else: 
                        if curr_word in batch_words[i][j]: 
                            word_tokenids[curr_word].append(k)
            for tup in batch_tups[i][j]: 
                line_num, word, ss = tup
                if 'mask' in exp_name: 
                    token_ids_word = np.array(word_tokenids['[MASK]']) 
                else: 
                    token_ids_word = np.array(word_tokenids[word]) 
                word_embed = vector[j][token_ids_word]
                word_embed = word_embed.mean(dim=0).detach().cpu().numpy() # average word pieces
                if np.isnan(word_embed).any(): 
                    print("PROBLEM!!!", word, batch[j])
                    return 
                if save_agg: 
                    word_ss = word + '@' + ss
                    word_reps[word_ss] += word_embed
                    word_counts[word_ss] += 1
                else: 
                    word_reps[ss].append(word_embed)
                    word_rep_keys[ss].append([line_num, word])
        torch.cuda.empty_cache()
    if save_agg: 
        res = {}
        for tup in word_counts: 
            res[tup] = list(word_reps[tup] / word_counts[tup])
        with open(LOGS + 'semantics_val/adj_BERT.json', 'w') as outfile: 
            json.dump(res, outfile)
    else: 
        out_folder = LOGS + 'wikipedia/substitutes/' + exp_name + '/'
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        for ss in word_reps: 
            out_array = np.array(word_reps[ss])
            np.save(out_folder + ss + '.npy', out_array)
        with open(out_folder + 'word_rep_key.json', 'w') as outfile: 
            json.dump(word_rep_keys, outfile)
        
# --------------
# Occupation functions
# --------------
    
def get_occupation_embeddings(): 
    '''
    For each stretch of wikitext, get BERT embeddings
    of occupation words
    '''
    with open(DATA + 'semantics/occupation_sents.json', 'r') as infile: 
        occ_sents = json.load(infile) 
        
    print("Batching data...")
    batch_size = 8
    batch_sentences = [] # each item is a list
    batch_idx = [] # each item is a list
    batch_words = []
    curr_batch = []
    curr_words = []
    curr_idx = []
    btokenizer = BasicTokenizer(do_lower_case=True)
    for occ in occ_sents: 
        for text in occ_sents[occ]: 
            tokens = btokenizer.tokenize(text)
            curr_batch.append(tokens)
            # take care of bigrams 
            curr_word_tokens = btokenizer.tokenize(occ)
            word_ids = []
            temp_str = ' '.join(curr_word_tokens)
            # sliding window of size curr_word_tokens over tokens
            for i in range(len(tokens) - len(curr_word_tokens)): 
                window = tokens[i:i+len(curr_word_tokens)]
                if ' '.join(window) == temp_str:
                    word_ids.extend(range(i, i+len(curr_word_tokens)))
            curr_idx.append(word_ids)
            curr_words.append(occ)
            if len(curr_batch) == batch_size: 
                batch_sentences.append(curr_batch)
                batch_words.append(curr_words)
                batch_idx.append(curr_idx)
                curr_batch = []
                curr_words = []
                curr_idx = []
    if len(curr_batch) != 0: # fence post
        batch_sentences.append(curr_batch)
        batch_words.append(curr_words)
        batch_idx.append(curr_idx)

    print("Getting model...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    layers = [-4, -3, -2, -1] # last four layers
    model.to(device)
    model.eval()
    
    word_reps = {}
    for occ in occ_sents: 
        word_reps[occ] = np.zeros(3072)
    word_counts = Counter()
    
    for i, batch in enumerate(tqdm(batch_sentences)): # for every batch
        word_tokenids = {} # { j : { word : [token ids] } }
        encoded_inputs = tokenizer(batch, is_split_into_words=True, padding=True, truncation=True, 
             return_tensors="pt")
        encoded_inputs.to(device)
        outputs = model(**encoded_inputs, output_hidden_states=True)
        states = outputs.hidden_states # tuple
        # batch_size x seq_len x 3072
        vector = torch.cat([states[i] for i in layers], 2) # concatenate last four
        for j in range(len(batch)): # for every example
            word_ids = encoded_inputs.word_ids(j)
            word_tokenids = []
            for k, word_id in enumerate(word_ids): # for every token
                if word_id is not None and word_id in batch_idx[i][j]:
                    word_tokenids.append(k)
            token_ids_word = np.array(word_tokenids) 
            word_embed = vector[j][token_ids_word]
            word_embed = word_embed.mean(dim=0).cpu().detach().numpy() # average word pieces
            if np.isnan(word_embed).any(): 
                print("PROBLEM!!!", word, batch[j])
                return 
            occ = batch_words[i][j]
            word_reps[occ] += word_embed
            word_counts[occ] += 1
    
    res = {}
    for w in word_counts: 
        res[w] = list(word_reps[w] / word_counts[w])
    with open(LOGS + 'semantics_val/occupations_BERT.json', 'w') as outfile: 
        json.dump(res, outfile)
        
def get_bert_mean_std(): 
    '''
    Get a mean and standard deviation vector
    from a sample of BERT embeddings, one random
    word per context drawn from approx 10% 
    of the adjective dataset 
    '''
    random.seed(0)
    batch_size = 6
    batch_sentences = [] # each item is a list
    curr_batch = []
    print("Batching data...")
    prob = 5
    with open(LOGS + 'wikipedia/adj_data/part-00000', 'r') as infile: 
        for line in infile:
            if random.randrange(100) > prob: continue
            contents = line.split('\t')
            text = '\t'.join(contents[1:]).lower()
            curr_batch.append(text)
            if len(curr_batch) == batch_size: 
                batch_sentences.append(curr_batch)
                curr_batch = []
        if len(curr_batch) != 0: # fence post
            batch_sentences.append(curr_batch)
    print("Num batches:", len(batch_sentences))
    print("Getting model...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    layers = [-4, -3, -2, -1] # last four layers
    model.to(device)
    model.eval()

    print("Calculate mean...")
    word_rep = np.zeros(3072)
    word_count = 0
    for i, batch in enumerate(tqdm(batch_sentences)): # for every batch
        encoded_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        encoded_inputs.to(device)
        outputs = model(**encoded_inputs, output_hidden_states=True)
        states = outputs.hidden_states # tuple
        # batch_size x seq_len x 3072
        vector = torch.cat([states[i] for i in layers], 2) # concatenate last four
        indices = np.random.randint(vector.size()[1], size=vector.size()[0])
        # batch_size x 3072
        vector = vector[np.arange(vector.size()[0]),indices,:]
        word_count += vector.size()[0]
        word_rep += vector.sum(dim=0).cpu().detach().numpy()
    mean_word_rep = word_rep / word_count
    
    print("Calculate std...")
    word_rep = np.zeros(3072)
    word_count = 0
    for i, batch in enumerate(tqdm(batch_sentences)): # for every batch
        encoded_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        encoded_inputs.to(device)
        outputs = model(**encoded_inputs, output_hidden_states=True)
        states = outputs.hidden_states # tuple
        # batch_size x seq_len x 3072
        vector = torch.cat([states[i] for i in layers], 2) # concatenate last four
        indices = np.random.randint(vector.size()[1], size=vector.size()[0])
        # batch_size x 3072
        vector = vector[np.arange(vector.size()[0]),indices,:].cpu().detach().numpy()
        word_count += vector.shape[0]
        vector = np.square(vector - mean_word_rep)
        word_rep += np.sum(vector, axis=0)
    std_word_rep = np.sqrt(word_rep / word_count)
    np.save(LOGS + 'wikipedia/mean_BERT.npy', mean_word_rep)
    np.save(LOGS + 'wikipedia/std_BERT.npy', std_word_rep)

def main(): 
    #sample_wikipedia()
    #get_axes_contexts()
    #print("----------------------")
    #get_adj_embeddings('bert-default', save_agg=False)
    #get_adj_embeddings('bert-base-sub-mask', save_agg=False)
    #get_adj_embeddings('bert-base-prob', save_agg=False)
    #print("**********************")
    #get_bert_mean_std()
    get_occupation_embeddings()

if __name__ == '__main__':
    main()
