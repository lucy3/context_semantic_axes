"""
For getting BERT embeddings of key words from wikipedia
"""
import requests
import json
from tqdm import tqdm
import wikitextparser as wtp
from transformers import BasicTokenizer, BertTokenizerFast, BertModel
#from pyspark import SparkConf, SparkContext
#from pyspark.sql import Row, SQLContext
from functools import partial
from collections import Counter, defaultdict
import random
import torch

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
    '''
    # more conservative cutoff in search to account for wordpieces
    line, line_id = tup
    try: 
        line = wtp.remove_markup(line)
    except AttributeError: 
        # some short lines with url breaks wtp
        print("####ERROR", line)
        return []
    tokens = tokenizer.tokenize(line)[:450]
    overlap = set(tokens) & vocab
    wspace_tokens = line.lower().split()[:450]
    overlap = (set(wspace_tokens) & vocab) | overlap 
    ret = []
    for w in overlap: 
        ret.append((w, line_id))
    return ret

def get_content_lines(line): 
    # only get wikitext content
    line = line.strip()
    return not line.startswith('{{') and not line.startswith('<') and \
        not line.startswith('==')

def exact_sample(tup): 
    w = tup[0]
    occur = tup[1]
    if len(occur) < 1000: 
        return tup
    else: 
        return (tup[0], random.sample(occur, 1000))
            
def sample_wikipedia(vocab, vocab_name): 
    '''
    Finds occurrences of vocab words in wikipedia
    '''
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    wikipedia_file = '/mnt/data0/corpora/wikipedia/enwiki-20211101-pages-meta-current.xml'
    #wikipedia_file = '/mnt/data0/corpora/wikipedia/small_wiki'
    tokenizer = BasicTokenizer(do_lower_case=True)
    data = sc.textFile(wikipedia_file).filter(get_content_lines)
    data = data.zipWithUniqueId() 
    token_data = data.flatMap(partial(contains_vocab, tokenizer=tokenizer, vocab=vocab))
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
    # more conservative cutoff in search to account for wordpieces
    try: 
        line = wtp.remove_markup(line)
    except AttributeError: 
        # one line with a url breaks wtp
        print("####ERROR", line)
        return []
    tokens = tokenizer.tokenize(line)[:450]
    counts = Counter(tokens)
    wspace_tokens = line.lower().split()[:450]
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
    with open(LOGS + 'wikipedia/temp_' + vocab_name + '_lines.json', 'r') as infile: 
        token_lines = json.load(infile) # {vocab word: [lines it appears in]}
    lines_tokens = defaultdict(list) # {line_num: [vocab words in line]}
    for token in token_lines: 
        for line_num in token_lines[token]: 
            lines_tokens[line_num].append(token)
    # TODO: get a mapping from word to axes it belongs to
    batch_size = 128
    batch_sentences = [] # each item is a list
    batch_words = [] # each item is a list
    batch_spans = []
    curr_batch = []
    curr_words = []
    curr_spans = []
    with open(LOGS + 'wikipedia/temp_' + vocab_name + '_data/part-00000', 'r') as infile: 
        for line in infile:
            contents = line.split('\t')
            line_num = contents[0]
            text = '\t'.join(contents[1:])
            text = wtp.remove_markup(text)
            words_in_line = lines_tokens[line_num]
            lower_text = text.lower()
            spans = [-1]*len(text) # each index is either -1 or the index of word in words_in_line
            for i, word in enumerate(words_in_line): 
                res = re.search(r'\b' + word + r'\b', lower_text)
                if res is None: print("PROBLEM WITH", word, lower_text)
                for j in range(res.start(), res.end()): 
                    spans[j] = i 
            curr_batch.append(text)
            curr_words.append(words_in_line)
            curr_spans.append(spans) 
            if len(curr_batch) == batch_size: 
                batch_sentences.append(curr_batch)
                batch_words.append(curr_words)
                batch_spans.append(curr_spans)
                curr_batch = []
                curr_words = []
                curr_spans = []
                #break # TODO remove this
        if len(curr_batch) != 0: # fence post
            batch_sentences.append(curr_batch)
            batch_words.append(curr_words)
            batch_spans.append(curr_spans)
    return batch_sentences, batch_words, batch_spans

def get_adj_embeddings(): 
    '''
    This function gets embeddings for adjectives in Wikipedia
    This is slightly messy because adjectives in Wordnet 
    can be multiple words. 
    '''
    batch_sentences, batch_words, batch_spans = batch_adj_data()
    return 
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    layers = [-4, -3, -2, -1] # last four layers
    model.eval()
    model.to(device)
    for i, batch in enumerate(batch_sentences): # for every batch
        word_tokenids = {} # { j : { word : [token ids] } }
        encoded_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        # possibly lame way to get tokens for multi-"word" adjectives (e.g. "fifty-nine")
        for j in range(len(batch)): # for every example
            this_w_tid = defaultdict(list) # { word : [token ids] }  
            # TODO: get mapping from word to character span 
            word_ids = encoded_inputs.word_ids(j)
            for k, word_id in enumerate(word_ids): # for every token
                if word_id is not None: 
                    span = encoded_inputs.token_to_chars(j, word_id)
                    # if span.end maps to a word, then add k to dict of word to token ids
                    if span.end in word_charidx: 
                        this_word = word_charidx[span.end]
                        this_w_tid[this_word].append(k)
            word_tokenids[j] = this_w_tid
        break
        outputs = model(**encoded_inputs)
        states = output.hidden_states
        vector = torch.stack([states[i] for i in layers]) # concatenate last four
        # TODO: check size of vector 
        for j in range(len(batch)): # for every example
            for word in batch_words[i][j]: 
                token_ids_word = np.array(word_tokenids[j][word]) 
                word_tokens_output = vector[j, token_ids_word]
                word_tokens_output.mean(dim=0) # average word pieces
        # TODO: sum onto zero-initialized vectors for each axes, count occurrences
    # TODO: divide sum by total to get an average representation of axes
    # TODO: save axes into matrix, save axes in txt in order of matrix rows

def count_axes(): 
    with open(LOGS + 'wikipedia/adj_counts.json', 'r') as infile: 
        total_count = Counter(json.load(infile))
        
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

def main(): 
    #vocab = get_adj()
    #sample_wikipedia(vocab, 'adj')
    get_adj_embeddings()
    #count_axes()

if __name__ == '__main__':
    main()
