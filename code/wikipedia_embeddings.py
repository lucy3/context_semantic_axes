"""
For getting BERT embeddings of key words from wikipedia
"""
import requests
import json
from tqdm import tqdm
import wikitextparser as wtp
from transformers import BasicTokenizer, BertTokenizerFast, BertModel, BertTokenizer
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SQLContext
from functools import partial
from collections import Counter, defaultdict
import random
import torch
from nltk import tokenize
import numpy as np

ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_adj(): 
    '''
    Read in adjectives
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
    
def get_occupations(): 
    '''
    Read in occupation words
    '''
    pass

def contains_vocab(tup, tokenizer=None, vocab=set()): 
    '''
    Input: [(line, line_id)]
    Output: [(vocab_token1, line_id), (vocab_token2, line_id)]
    
    Since BERT was originally made for sentence level tasks, we split Wikipedia into sentences,
    and for efficiency, keep those that are > 10 words and < 150 words long.
    '''
    line, line_id = tup
    line = wtp.remove_markup(line)
    line = line.replace('-', 'xqxq')
    tokens = tokenizer.tokenize(line)
    if len(tokens) < 10 or len(tokens) > 150: 
        return []
    overlap = set(tokens) & vocab
    ret = []
    for w in overlap: 
        ret.append((w, line_id))
    return ret

def get_content_lines(line): 
    '''
    If you need to rerun wikipedia sampling multiple times, a speed up could be
    to tokenize only once with wordpiece tokenizer, join using spaces, re replace
    ' ##' (next to non-white space boundary) with ''. Then can run whitespace tokenizer. 
    '''
    # only get wikitext content
    line = line.strip()
    if len(line) < 10:
        return False
    line_content = not line.startswith('{{') and not line.startswith('<') and \
        not line.startswith('=') and not line.startswith('*') and not line.startswith('#') and \
        not line.startswith(';') and not line.startswith(':')
    if not line_content: 
        return False
    try: 
        line = wtp.remove_markup(line)
    except AttributeError: 
        # one line with a url breaks wtp
        print("####ERROR", line)
        return False
    return True

def exact_sample(tup): 
    w = tup[0]
    occur = tup[1]
    if len(occur) < 1000: 
        return tup
    else: 
        return (tup[0], random.sample(occur, 1000))
    
def get_sentences(line): 
    line = wtp.remove_markup(line)
    sents = tokenize.sent_tokenize(line)
    return sents
            
def sample_wikipedia(vocab, vocab_name): 
    '''
    Finds occurrences of vocab words in wikipedia. 
    
    Note: the adjectives in "adj_lines.json" have "xqxq"
    instead of "-" to account for dashed words for tokenization. 
    '''
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    
    new_vocab = set()
    for w in vocab: # helps handle dashed words
        if '-' in w: 
            new_vocab.add(w.replace('-', 'xqxq'))
        else: 
            new_vocab.add(w)

    wikipedia_file = '/mnt/data0/corpora/wikipedia/enwiki-20211101-pages-meta-current.xml'
    #wikipedia_file = '/mnt/data0/corpora/wikipedia/small_wiki'
    tokenizer = BasicTokenizer(do_lower_case=True)
    data = sc.textFile(wikipedia_file).filter(get_content_lines)
    data = data.flatMap(get_sentences)
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
    
def count_vocab_words(line, tokenizer=None, vocab=set()): 
    '''
    This is deprecated. 
    '''
    # more conservative cutoff in search to account for wordpieces
    try: 
        line = wtp.remove_markup(line)
    except AttributeError: 
        # one line with a url breaks wtp
        print("####ERROR", line)
        return []
    tokens = tokenizer.tokenize(line)
    counts = Counter(tokens)
    wspace_tokens = line.lower().split()
    wspace_counts = Counter(wspace_tokens)
    ret = []
    for w in wspace_counts: 
        # because bert tokenizer splits words with '-'
        if '-' in w and w in vocab: 
            ret.append((w, wspace_counts[w]))    
    for w in counts: 
        if w in vocab: 
            ret.append((w, counts[w]))
    return ret
    
def batch_adj_data(): 
    vocab = get_adj()
    vocab_name = 'adj'
    with open(LOGS + 'wikipedia/' + vocab_name + '_lines_random.json', 'r') as infile:
        lines_tokens = json.load(infile) # {line_num: [vocab words in line]}
    batch_size = 8
    batch_sentences = [] # each item is a list
    batch_words = [] # each item is a list
    curr_batch = []
    curr_words = []
    btokenizer = BasicTokenizer(do_lower_case=True)
    vocab = set()
    with open(LOGS + 'wikipedia/' + vocab_name + '_data/part-00000', 'r') as infile: 
        for line in infile:
            contents = line.split('\t')
            line_num = contents[0]
            if line_num not in lines_tokens: continue
            text = '\t'.join(contents[1:])
            text = wtp.remove_markup(text).lower()
            words_in_line = lines_tokens[line_num]
            vocab.update(words_in_line)
            dash_words = set()
            for w in words_in_line: 
                if '-' in w:  
                    sub_w = w.replace('-', 'xqxq')
                    dash_words.add(sub_w)
                    text = text.replace(w, sub_w)
            tokens = btokenizer.tokenize(text)
            if dash_words: 
                for i in range(len(tokens)): 
                    if tokens[i] in dash_words: 
                        tokens[i] = tokens[i].replace('xqxq', '-')
            curr_batch.append(tokens)
            curr_words.append(words_in_line)
            if len(curr_batch) == batch_size: 
                batch_sentences.append(curr_batch)
                batch_words.append(curr_words)
                curr_batch = []
                curr_words = []
        if len(curr_batch) != 0: # fence post
            batch_sentences.append(curr_batch)
            batch_words.append(curr_words)
    return batch_sentences, batch_words, vocab

def get_adj_embeddings(): 
    '''
    This function gets embeddings for adjectives in Wikipedia
    This is slightly messy because adjectives in Wordnet 
    can be multiple words. 
    '''
    print("Batching contexts...")
    batch_sentences, batch_words, vocab = batch_adj_data()
    print("Getting model...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    layers = [-4, -3, -2, -1] # last four layers
    model.to(device)
    model.eval()
    # initialize word representations
    word_reps = {}
    for w in vocab: 
        word_reps[w] = np.zeros(3072)
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
            word_tokenids = defaultdict(list) # {word : [token ids]}
            for k, word_id in enumerate(word_ids): # for every token
                if word_id is not None: 
                    curr_word = batch[j][word_id]
                    if curr_word in batch_words[i][j]: 
                        word_tokenids[curr_word].append(k)
            for word in batch_words[i][j]: 
                token_ids_word = np.array(word_tokenids[word]) 
                word_embed = vector[j][token_ids_word]
                word_embed = word_embed.mean(dim=0).cpu().detach().numpy() # average word pieces
                if np.isnan(word_embed).any(): 
                    print(word, batch[j])
                    return
                word_reps[word] += word_embed
                word_counts[word] += 1
    res = {}
    for word in word_counts: 
        res[word] = list(word_reps[word] / word_counts[word])
    with open(LOGS + 'semantics_val/adj_BERT.json', 'w') as outfile: 
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
            text = '\t'.join(contents[1:])
            text = wtp.remove_markup(text).lower()
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

def count_axes(): 
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
            if len(contents) < 3: continue # no antonyms
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

def sample_random_contexts(axis, adj_lines): 
    axis_lines = set()
    for adj in axis: 
        for line in adj_lines[adj]: 
            a = adj.replace('xqxq', '-')
            axis_lines.add((a, line))
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
    
    ret = defaultdict(list) # { line_num: [vocab words in line]}
    with open(LOGS + 'semantics_val/wordnet_axes.txt', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t') 
            if len(contents) < 3: continue # no antonyms
            synset = contents[0]
            axis1 = contents[1].split(',')
            axis1_lines = sample_random_contexts(axis1, adj_lines)
            for tup in axis1_lines: 
                ret[tup[1]].append(tup[0])
            axis2 = contents[2].split(',')
            axis2_lines = sample_random_contexts(axis2, adj_lines)
            for tup in axis2_lines: 
                ret[tup[1]].append(tup[0])
    with open(LOGS + 'wikipedia/adj_lines_random.json', 'w') as outfile: 
        json.dump(ret, outfile)

def main(): 
    vocab = get_adj()
    sample_wikipedia(vocab, 'adj')
    #get_axes_contexts()
    #get_adj_embeddings()
    #get_bert_mean_std()

if __name__ == '__main__':
    main()
