"""
For getting BERT embeddings of key words from wikipedia
"""
import requests
import json
from tqdm import tqdm
import wikitextparser as wtp
from transformers import BasicTokenizer
#from pyspark import SparkConf, SparkContext
#from pyspark.sql import Row, SQLContext
from functools import partial
from collections import Counter

ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'

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

def contains_vocab(line, tokenizer=None, vocab=set()): 
    # more conservative cutoff in search to account for wordpieces
    tokens = tokenizer.tokenize(line)[:450]
    return len(set(tokens) & vocab) != 0

def get_content_lines(line): 
    # only get wikitext content
    line = line.strip()
    return not line.startswith('{{') and not line.startswith('<')
            
def sample_wikipedia(vocab, vocab_name): 
    '''
    Finds occurrences of vocab words in a sample of wikipedia
    '''
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    wikipedia_file = '/mnt/data0/corpora/wikipedia/enwiki-20211101-pages-meta-current.xml'
    tokenizer = BasicTokenizer(do_lower_case=True)
    data = sc.textFile(wikipedia_file).filter(get_content_lines)
    data = data.sample(False,0.15,0)
    data = data.filter(partial(contains_vocab, tokenizer=tokenizer, vocab=vocab))
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
    
def count_instances(vocab, vocab_name): 
    '''
    Count the total number of instances of each
    vocab word in the sample of wikipedia. 
    '''
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    
    tokenizer = BasicTokenizer(do_lower_case=True)
    data = sc.textFile(LOGS + 'wikipedia/' + vocab_name + '_data')
    data = data.flatMap(partial(count_vocab_words, tokenizer=tokenizer, vocab=vocab))
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    total_count = Counter(data.collectAsMap())
    with open(LOGS + 'wikipedia/' + vocab_name + '_counts.json', 'w') as outfile: 
        json.dump(total_count, outfile)
        
    sc.stop()
    
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
    vocab = get_adj()
    #sample_wikipedia(vocab, 'adj')
    #get_embeddings(vocab, 'adj')
    count_axes()

if __name__ == '__main__':
    main()
