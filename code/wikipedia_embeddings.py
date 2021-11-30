"""
For getting BERT embeddings of key words from wikipedia
"""
import requests
import json
from tqdm import tqdm
import wikitextparser as wtp
from transformers import BasicTokenizer
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SQLContext
from functools import partial

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
    tokens = tokenizer.tokenize(line)[:512]
    return len(set(tokens) & vocab) != 0
            
def sample_wikipedia(vocab, vocab_name): 
    '''
    Finds occurrences of vocab words in a sample of wikipedia
    '''
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    wikipedia_file = '/mnt/data0/corpora/wikipedia/enwiki-20211101-pages-meta-current.xml'
    tokenizer = BasicTokenizer(do_lower_case=True)
    data = sc.textFile(wikipedia_file).sample(False,0.1,0)
    data = data.filter(partial(contains_vocab, tokenizer=tokenizer, vocab=vocab))
    data.coalesce(1).saveAsTextFile(LOGS + 'wikipedia/' + vocab_name + '_data')
    
    sc.stop()

def main(): 
    vocab = get_adj()
    sample_wikipedia(vocab, 'adj')

if __name__ == '__main__':
    main()
