"""
Various helper functions
"""
import csv
from collections import Counter, defaultdict
from transformers import BasicTokenizer
import os
import json
import re
#from pyspark import SparkConf, SparkContext
#from pyspark.sql import SQLContext
#from functools import partial
import sys
import time
import math
from tqdm import tqdm
from helpers import get_sr_cats, get_manual_people
from nltk.stem.porter import PorterStemmer

ROOT = '/mnt/data0/lucy/manosphere/'
#ROOT = '/global/scratch/lucy3_li/manosphere/'
COMMENTS = ROOT + 'data/comments/'
POSTS = ROOT + 'data/submissions/'
FORUMS = ROOT + 'data/cleaned_forums/'
PEOPLE_FILE = ROOT + 'data/people.csv'
NONPEOPLE_FILE = ROOT + 'data/non-people.csv'
LOGS = ROOT + 'logs/'
UD = LOGS + 'urban_dict.csv'
WORD_COUNT_DIR = LOGS + 'gram_counts/'

def get_manual_nonpeople(): 
    '''
    This is the list of words in community glossaries 
    that are not labeled as people. 
    
    This function does not seem to be used anywhere. 
    '''
    words = set()
    with open(NONPEOPLE_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['community'].strip() != 'generic': 
                word = row['word']
                if word != '':
                    if word.lower() in words: print('REPEAT', word)
                    words.add(word.lower())
    return words

def calculate_ud_coverage(): 
    """
    See how many common nouns for people turn up in urban dictionary.
    """
    people, sing2plural = get_manual_people()
    num_definitions = Counter()
    with open(UD, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('|')
            word = contents[0].lower()
            if word in people: 
                num_definitions[word] += 1
    missing_count = 0
    for w in sing2plural: 
        if num_definitions[w] == 0: 
            if num_definitions[sing2plural[w]] == 0: 
                missing_count += 1
    print("Total number of people:", len(sing2plural))
    print("Missing:", missing_count)
    print("Number of definitions:", num_definitions.values())

def count_glossword_time_place(): 
    all_terms, _ = get_manual_people()
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    
    categories = get_sr_cats()
    df = load_gram_counts(categories, sqlContext)
    df = df.filter(df.word.isin(all_terms))
    pd_df = df.toPandas()
    sc.stop()
    
    missing = all_terms - set(pd_df['word'].to_list()) 
    print("Number of words missing:", len(missing))
    for w in missing: 
        print(w)
    
    pd_df.to_csv(LOGS + 'glossword_time_place.csv')
        
def get_ngrams_glosswords(): 
    '''
    Get number of tokens in glossary words, 
    to see what we are missing if we only use bigrams and unigrams. 
    '''
    all_terms, _ = get_manual_people()
    num_tokens = defaultdict(list)
    for w in all_terms: 
        num_tokens[len(w.split())].append(w)
    print(num_tokens.keys())
    print(num_tokens[5], num_tokens[4], num_tokens[3])

def load_gram_counts(categories, sqlContext): 
    reddit_df = sqlContext.read.parquet(WORD_COUNT_DIR + 'subreddit_counts')
    leave_out = []
    for sr in categories: 
        if categories[sr] == 'Health' or categories[sr] == 'Criticism': 
            leave_out.append(sr)
    reddit_df = reddit_df.filter(~reddit_df.community.isin(leave_out))
    
    forum_df = sqlContext.read.parquet(WORD_COUNT_DIR + 'forum_counts')
    df = forum_df.union(reddit_df)
    return df

def update_tagged_counts(line, i, categories, tokenizer, deps, depheads, tagged_counts, 
                       prefix_counts, reddit=True): 
    content = line.split('\t')
    entities = content[1:]
    if reddit: 
        sr = content[0]
        cat = categories[sr.lower()]
        if cat == 'Health' or cat == 'Criticism': 
            return tagged_counts, prefix_counts
    for entity in entities: 
        if entity.strip() == '': continue
        tup = entity.lower().split(' ')
        label = tup[0]
        start = int(tup[1])
        end = int(tup[2])
        head = int(tup[3])
        phrase = ' '.join(tup[4:])
        phrase = tokenizer.tokenize(phrase)
 
        phrase_start = 0 
        deprel = deps[str(i)][tup[1]]
        dephead = depheads[str(i)][tup[1]]
        det_poss = ''
        if (deprel == 'poss' or deprel == 'det') and dephead == head: 
            det_poss = phrase[0]
            phrase_start = 1
        # get bigrams and unigrams from start to end
        other_phrase = phrase[phrase_start:]
        if len(other_phrase) < 3: 
            this_entity = ' '.join(other_phrase)
            tagged_counts[this_entity][label] += 1
            prefix_counts[this_entity][det_poss] += 1
    return tagged_counts, prefix_counts
    
def count_tagged_entities(): 
    '''
    Gather tagged entity unigrams and bigrams
    that we would want to include in our analysis 
    '''
    # look at tagged entities 
    tagged_counts = defaultdict(Counter) # { entity : {proper noun : count, common noun: count} }
    prefix_counts = defaultdict(Counter) # { entity : {'my' : count, 'the': count} }
    
    tokenizer = BasicTokenizer(do_lower_case=True)
    categories = get_sr_cats()
    
    months = os.listdir(COMMENTS)
    for folder in months: 
        m = folder.replace('RC_', '')
        if m == 'bad_jsons': continue
        with open(LOGS + 'deprel_reddit/' + m + '_deps.json', 'r') as infile: 
            deps = json.load(infile)
        with open(LOGS + 'deprel_reddit/' + m + '_depheads.json', 'r') as infile: 
            depheads = json.load(infile)
        with open(LOGS + 'tagged_people/' + m, 'r') as infile: 
            for i, line in enumerate(infile): 
                tagged_counts, prefix_counts = update_tagged_counts(line, i, categories, tokenizer, \
                                                 deps, depheads, tagged_counts, \
                                                 prefix_counts, reddit=True) 

    for f in os.listdir(FORUMS): 
        with open(LOGS + 'deprel_forums/' + f + '_deps.json', 'r') as infile: 
            deps = json.load(infile)
        with open(LOGS + 'deprel_forums/' + f + '_depheads.json', 'r') as infile: 
            depheads = json.load(infile)
        with open(LOGS + 'tagged_people/' + f, 'r') as infile: 
            for i, line in enumerate(infile): 
                tagged_counts, prefix_counts = update_tagged_counts(line, i, categories, tokenizer, \
                                                 deps, depheads, tagged_counts, \
                                                 prefix_counts, reddit=False) 

    # save tagged counts
    outpath = WORD_COUNT_DIR + 'tagged_counts_full.json'
    with open(outpath, 'w') as outfile: 
        json.dump(tagged_counts, outfile)
    outpath = WORD_COUNT_DIR + 'prefix_counts_full.json'
    with open(outpath, 'w') as outfile: 
        json.dump(prefix_counts, outfile)
    
def get_significant_entities(): 
    # load tagged counts    
    inpath = WORD_COUNT_DIR + 'tagged_counts_full.json'
    with open(inpath, 'r') as infile: 
        tagged_counts = json.load(infile)
    inpath = WORD_COUNT_DIR + 'prefix_counts_full.json'
    with open(inpath, 'r') as infile: 
        prefix_counts = json.load(infile)
    
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    
    categories = get_sr_cats()
    df = load_gram_counts(categories, sqlContext)

    all_counts = df.rdd.map(lambda x: (x[0], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    all_counts = Counter(all_counts)
    
    sc.stop()
    
    # save significant entities that occur at least X times 
    unigrams = Counter()
    bigrams = Counter()
    for gram in tagged_counts: 
        # the word needs to be popular
        if all_counts[gram] < 500: continue
        tagged_total = sum(tagged_counts[gram].values())
        # at least half of its instances should be people to avoid ambiguity
        if tagged_total < (all_counts[gram] / 5): continue
        gram_len = len(gram.split(' '))
        assert gram_len < 3 and gram_len > 0
        if gram_len == 1: 
            unigrams[gram] = all_counts[gram]
        if gram_len == 2: 
            bigrams[gram] = all_counts[gram]
    with open(LOGS + 'significant_entities.csv', 'w') as outfile: 
        fieldnames = ['ngram', 'entity', 'total_count', 'tagged_count', 'ner_labels', 'det_poss_count']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        for tup in unigrams.most_common(): 
            d = {}
            d['ngram'] = 'unigram'
            d['entity'] = tup[0]
            d['total_count'] = tup[1]
            d['tagged_count'] = sum(tagged_counts[tup[0]].values())
            d['ner_labels'] = json.dumps(tagged_counts[tup[0]])
            d['det_poss_count'] = json.dumps(prefix_counts[tup[0]])
            writer.writerow(d)
        for tup in bigrams.most_common(): 
            d = {}
            d['ngram'] = 'bigram'
            d['entity'] = tup[0]
            d['total_count'] = tup[1]
            d['tagged_count'] = sum(tagged_counts[tup[0]].values())
            d['ner_labels'] = json.dumps(tagged_counts[tup[0]])
            d['det_poss_count'] = json.dumps(prefix_counts[tup[0]])
            writer.writerow(d)
            
def test_lemmatizer(): 
    import spacy
    
    spacy_nlp = spacy.load("en_core_web_sm")
    words, sing2plural = get_manual_people()
    for sing_w in sing2plural: 
        plural_w = sing2plural[sing_w]
        doc = spacy_nlp('They\'re the ' + plural_w)
        plural_lemmas = []
        for token in doc: 
            plural_lemmas.append(token.lemma_)
        doc = spacy_nlp('They\'re the ' + sing_w)
        sing_lemmas = []
        for token in doc: 
            sing_lemmas.append(token.lemma_)
        if plural_lemmas != sing_lemmas: 
            print(plural_lemmas, sing_lemmas)
    print('-----------')
    stemmer = PorterStemmer()
    for sing_w in sing2plural: 
        plural_w = sing2plural[sing_w]
        if plural_w == '': continue
        stem1 = stemmer.stem(plural_w.split()[-1])
        stem2 = stemmer.stem(sing_w.split()[-1])
        if stem1 != stem2: 
            print(sing_w, plural_w, stem1, stem2) 

def find_examples(w, outfile): 
    '''
    Print some examples of a word
    '''
    months = ['2009-01', '2012-01', '2015-01', '2018-01']
    for month in months: 
        with open(COMMENTS + 'RC_' + month + '/part-00000', 'r') as infile:
            i = 0
            for line in infile: 
                d = json.loads(line)
                if w in d['body'].lower(): 
                    outfile.write(d['body'] + '\n')
                    outfile.write('----------------\n')
                    i =+ 1
                    if i > 3: break

def write_out_examples(): 
    '''
    This function prints out examples of some
    unknown terms being used on Reddit 
    '''
    questionable = set()
    with open(ROOT + 'data/ann_sig_entities.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Q': 
                questionable.add(row['entity'])
    with open(LOGS + 'q_vocab_examples.txt', 'w') as outfile: 
        for w in tqdm(questionable): 
            outfile.write('####### WORD ' + w  + '\n')
            outfile.write('----------------\n')
            find_examples(' ' + w + ' ', outfile)

def main(): 
    #count_glosswords_in_tags()
    #get_ngrams_glosswords()
    #test_lemmatizer()
    #get_significant_entities()
    #count_tagged_entities()
    #count_glossword_time_place()
    write_out_examples()

if __name__ == '__main__':
    main()
