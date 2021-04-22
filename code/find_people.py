"""
Various helper functions
"""
import csv
from collections import Counter, defaultdict
from transformers import BasicTokenizer
import os
import json
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from functools import partial
import sys
import time
import math

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

def get_manual_people(): 
    """
    get list of words, add plural forms
    """
    words = set()
    sing2plural = {}
    with open(PEOPLE_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            word_sing = row['word (singular)'].strip()
            plural = row['word (plural)'].strip()
            if word_sing != '':
                if word_sing.lower() in words: print('REPEAT', word_sing)
                words.add(word_sing.lower())
                sing2plural[word_sing.lower()] = plural.lower()
            if plural != '': 
                if plural.lower() in words: print('REPEAT', plural)
                assert word_sing != ''
                words.add(plural.lower())
    return words, sing2plural

def get_manual_nonpeople(): 
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
    see how many common nouns for people turn up in urban dictionary
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
    
def get_sr_cats(): 
    categories = defaultdict(str)
    with open(ROOT + 'data/subreddits.txt', 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in reader: 
            name = row['Subreddit'].strip().lower()
            if name.startswith('/r/'): name = name[3:]
            if name.startswith('r/'): name = name[2:]
            if name.endswith('/'): name = name[:-1]
            categories[name] = row['Category after majority agreement']
    return categories

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
                       reddit=True, truncate=True): 
    content = line.split('\t')
    entities = content[1:]
    if reddit: 
        sr = content[0]
        cat = categories[sr.lower()]
        if cat == 'Health' or cat == 'Criticism': 
            return tagged_counts
    for entity in entities: 
        if entity.strip() == '': continue
        tup = entity.lower().split(' ')
        label = tup[0]
        start = int(tup[1])
        end = int(tup[2])
        head = int(tup[3])
        phrase = ' '.join(tup[4:])
        phrase = tokenizer.tokenize(phrase)
        if truncate: 
            phrase_start = 0
            # calculate phrase start, excluding determiner or possessive dep on root
            deprel = deps[str(i)][tup[1]]
            dephead = depheads[str(i)][tup[1]]
            if (deprel == 'poss' or deprel == 'det') and dephead == head: 
                det_poss = phrase[0]
                phrase_start = 1
            # get bigrams and unigrams from start to root
            if (1 + head - start) - phrase_start in set([1, 2]): 
                other_phrase = phrase[phrase_start:1 + head - start]
                this_entity = ' '.join(other_phrase)
                tagged_counts[this_entity] += 1
        else: 
            phrase_start = 0 
            deprel = deps[str(i)][tup[1]]
            dephead = depheads[str(i)][tup[1]]
            if (deprel == 'poss' or deprel == 'det') and dephead == head: 
                det_poss = phrase[0]
                phrase_start = 1
            # get bigrams and unigrams from start to end
            other_phrase = phrase[phrase_start:]
            if len(other_phrase) < 3: 
                this_entity = ' '.join(other_phrase)
                tagged_counts[this_entity] += 1
    return tagged_counts
    
def count_tagged_entities(truncate=False): 
    '''
    Gather tagged entity unigrams and bigrams
    that we would want to include in our analysis 
    '''
    # look at tagged entities 
    tagged_counts = Counter()
    tokenizer = BasicTokenizer(do_lower_case=True)
    categories = get_sr_cats()
    
    months = os.listdir(COMMENTS)
    for folder in months: 
        m = folder.replace('RC_', '')
        if not os.path.exists(LOGS + 'deprel_reddit/' + m + '_deps.json'): 
            print("DOES NOT EXIST!!!!!", m)
            continue
        with open(LOGS + 'deprel_reddit/' + m + '_deps.json', 'r') as infile: 
            deps = json.load(infile)
        with open(LOGS + 'deprel_reddit/' + m + '_depheads.json', 'r') as infile: 
            depheads = json.load(infile)
        with open(LOGS + 'tagged_people/' + m, 'r') as infile: 
            for i, line in enumerate(infile): 
                tagged_counts = update_tagged_counts(line, i, categories, tokenizer, \
                                                 deps, depheads, tagged_counts, \
                                                 reddit=False, truncate=truncate) 

    for f in os.listdir(FORUMS): 
        print('*************', f)
        with open(LOGS + 'deprel_forums/' + f + '_deps.json', 'r') as infile: 
            deps = json.load(infile)
        with open(LOGS + 'deprel_forums/' + f + '_depheads.json', 'r') as infile: 
            depheads = json.load(infile)
        with open(LOGS + 'tagged_people/' + f, 'r') as infile: 
            for i, line in enumerate(infile): 
                tagged_counts = update_tagged_counts(line, i, categories, tokenizer, \
                                                 deps, depheads, tagged_counts, \
                                                 reddit=False, truncate=truncate)  

    # save tagged counts
    if truncate: 
        outpath = WORD_COUNT_DIR + 'tagged_counts.json'
    else: 
        outpath = WORD_COUNT_DIR + 'tagged_counts_full.json'
    with open(outpath, 'w') as outfile: 
        json.dump(tagged_counts, outfile)
    
def get_signficant_entities(truncate=False): 
    # load tagged counts    
    if truncate: 
        inpath = WORD_COUNT_DIR + 'tagged_counts.json'
    else: 
        inpath = WORD_COUNT_DIR + 'tagged_counts_full.json'
    with open(inpath, 'r') as infile: 
        tagged_counts = json.load(infile)
    
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    
    categories = get_sr_cats()
    df = load_gram_counts(categories, sqlContext)

    all_counts = df.rdd.map(lambda x: (x[0], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    all_counts = Counter(all_counts)
    u_total = df.rdd.filter(lambda x: len(x[0].split(' ')) == 1).map(lambda x: (1, x[1])).reduceByKey(lambda x,y: x + y).collect()[0][1]
    b_total = df.rdd.filter(lambda x: len(x[0].split(' ')) == 2).map(lambda x: (1, x[1])).reduceByKey(lambda x,y: x + y).collect()[0][1]
    
    sc.stop()
    # save significant entities that occur at least X times 
    unigrams = Counter()
    bigrams = Counter()
    for gram in tagged_counts: 
        if tagged_counts[gram] < 10: continue
        if all_counts[gram] < 500: continue
        gram_len = len(gram.split(' '))
        assert gram_len < 3 and gram_len > 0
        if gram_len == 1: 
            unigrams[gram[0]] = all_counts[gram]
        if gram_len == 2: 
            bigrams[gram] = all_counts[gram]
    for tup in unigrams.most_common(): 
        print("********UNIGRAM", tup[0], tup[1], all_counts[tup[0]]) # TODO: also write out the count of ner labels they use and their determiner or possessive counts  
    for tup in bigrams.most_common(): 
        print("--------BIIGRAM", tup[0], tup[1], all_counts[tup[0]])

def main(): 
    #count_glosswords_in_tags()
    #get_ngrams_glosswords()
    get_significant_entities(truncate=False)
    #count_tagged_entities(truncate=False)
    #count_glossword_time_place()

if __name__ == '__main__':
    main()
