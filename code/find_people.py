"""
Various helper functions
"""
import csv
from collections import Counter, defaultdict
import os
import json
import re
#from pyspark import SparkConf, SparkContext
from functools import partial
import sys
import time

#ROOT = '/mnt/data0/lucy/manosphere/'
ROOT = '/global/scratch/lucy3_li/manosphere/'
COMMENTS = ROOT + 'data/comments/'
POSTS = ROOT + 'data/submissions/'
PEOPLE_FILE = ROOT + 'data/people.csv'
NONPEOPLE_FILE = ROOT + 'data/non-people.csv'
UD = ROOT + 'logs/urban_dict.csv'
LOGS = ROOT + 'logs/'

#conf = SparkConf()
#sc = SparkContext(conf=conf)

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
    
def get_term_count_comment(line, all_terms=None): 
    '''
    Returns a list where keys are subreddits and terms and values
    are counts
    '''
    d = json.loads(line)
    if 'body' not in d: 
        return []
    text = d['body'].lower()
    sr = d['subreddit'].lower()
    term_counts = Counter()
    for term in all_terms: 
        res = re.findall(r'\b' + re.escape(term) + r'\b', text)
        term_counts[term] += len(res)
    ret = []
    for term in term_counts: 
        ret.append(((sr, term), term_counts[term]))
    return ret 

def get_term_count_post(line, all_terms=None): 
    d = json.loads(line)
    if 'selftext' not in d: 
        return []
    text = d['selftext'].lower()
    sr = d['subreddit'].lower()
    term_counts = Counter()
    for term in all_terms:
        res = re.findall(r'\b' + re.escape(term) + r'\b', text)
        term_counts[term] += len(res)
    ret = []
    for term in term_counts: 
        ret.append(((sr, term), term_counts[term]))
    return ret 
    
def count_words_reddit():
    """
    This function is deprecated.
    It may still be able to run, but it is very slow (1 week)
    and counts all occurrences of both people and nonpeople glossary words
    """ 
    people, _ = get_manual_people()
    nonpeople = get_manual_nonpeople()
    all_terms = people | nonpeople
    word_time_place = defaultdict(Counter) # month_community : {word : count}
    for f in os.listdir(COMMENTS): 
        if f == 'bad_jsons': continue
        month = f.replace('RC_', '')
        print(month)
        data = sc.textFile(COMMENTS + f + '/part-00000')
        data = data.flatMap(partial(get_term_count_comment, all_terms=all_terms))
        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        data = data.collectAsMap()
        for key in data: 
            word_time_place[month + '_' + key[0]][key[1]] += data[key]
        
        if os.path.exists(POSTS + 'RS_' + month + '/part-00000'): 
            post_path = POSTS + 'RS_' + month + '/part-00000'
        else: 
            post_path = POSTS + 'RS_v2_' + month + '/part-00000'
        data = sc.textFile(post_path)
        data = data.flatMap(partial(get_term_count_post, all_terms=all_terms))
        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        data = data.collectAsMap()
        for key in data: 
            word_time_place[month + '_' + key[0]][key[1]] += data[key]

    with open(LOGS + 'glossword_time_place.json', 'w') as outfile: 
        json.dump(word_time_place, outfile)

def count_words_reddit_parallel(): 
    all_terms, _ = get_manual_people()
    f = sys.argv[1]
    month = f.replace('RC_', '')
    term_counts = defaultdict(Counter)
    with open(COMMENTS + f + '/part-00000', 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            if 'body' not in d: continue
            text = d['body'].lower()
            sr = d['subreddit'].lower()
            for term in all_terms: 
                res = re.findall(r'\b' + re.escape(term) + r'\b', text)
                term_counts[sr][term] += len(res)

    if os.path.exists(POSTS + 'RS_' + month + '/part-00000'): 
        post_path = POSTS + 'RS_' + month + '/part-00000'
    else: 
        post_path = POSTS + 'RS_v2_' + month + '/part-00000'
    with open(post_path, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            if 'selftext' not in d: continue
            text = d['selftext'].lower()
            sr = d['subreddit'].lower()
            for term in all_terms:
                res = re.findall(r'\b' + re.escape(term) + r'\b', text)
                term_counts[sr][term] += len(res)
    with open(LOGS + 'glossword_time_place/' + month + '.json', 'w') as outfile: 
        json.dump(term_counts, outfile)

def save_occurring_glosswords(): 
    '''
    Save only glossary words that actually appear in the text, 
    print out ones that do not appear. 
    '''
    words = Counter()
    all_words = set()
    for f in os.listdir(LOGS + 'glossword_time_place/'): 
        with open(LOGS + 'glossword_time_place/' + f, 'r') as infile: 
            counts = json.load(infile)
            for sr in counts: 
                for w in counts[sr]: 
                    all_words.add(w)
                    if counts[sr][w] != 0: 
                        words[w] += counts[sr][w]
    print("Missing glossary words:")
    num_missing = 0
    for w in sorted(all_words): 
        if w not in words: 
            print(w)
            num_missing += 1
    print("Number of missing", num_missing, "out of", len(all_words), "words")
    with open(LOGS + 'glossword_counts.json', 'w') as outfile: 
        json.dump(words, outfile)
        
def get_term_count_tagged(line, all_terms=None):
    '''
    Right now this only works on Reddit tags
    '''
    entities = line.strip().split('\t')
    sr = entities[0]
    text = '\t'.join(entities[1:]) 
    term_counts = Counter()
    for term in all_terms: 
        res = re.findall(r'\b' + re.escape(term) + r'\b', text)
        term_counts[term] += len(res)
    ret = []
    for term in term_counts: 
        ret.append((sr + '$' + term, term_counts[term]))
    return ret
        
def count_glosswords_in_tags(): 
    '''
    For every glossary word, count how much it occurs in tagged spans
    '''
    all_terms, _ = get_manual_people()
    data = sc.textFile(LOGS + 'all_tagged_people')
    data = data.flatMap(partial(get_term_count_tagged, all_terms=all_terms))
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    data = data.collectAsMap()
    with open(LOGS + 'tagged_glossword_counts.json', 'w') as outfile: 
        json.dump(data, outfile)
        
def get_ngrams_glosswords(): 
    '''
    Get number of tokens in glossary words
    '''
    all_terms, _ = get_manual_people()
    num_tokens = defaultdict(list)
    for w in all_terms: 
        num_tokens[len(w.split())].append(w)
    print(num_tokens.keys())
    print(num_tokens[5], num_tokens[4], num_tokens[3])

def main(): 
    #count_words_reddit()
    #count_words_reddit_parallel()
    #save_occurring_glosswords()
    #count_glosswords_in_tags()
    get_ngrams_glosswords()
    #sc.stop()

if __name__ == '__main__':
    main()
