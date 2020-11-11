"""
Various helper functions
"""
import csv
from collections import Counter, defaultdict
import os
import json
import re

ROOT = '/mnt/data0/lucy/manosphere/'
COMMENTS = ROOT + 'data/comments/'
POSTS = ROOT + 'data/submissions/'
PEOPLE_FILE = ROOT + 'data/people.csv'
NONPEOPLE_FILE = ROOT + 'data/non-people.csv'
UD = ROOT + 'logs/urban_dict.csv'
LOGS = ROOT + 'logs/'

def get_manual_people(): 
    """
    get list of words, add plural forms
    """
    words = set()
    sing2plural = {}
    with open(PEOPLE_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['community'].strip() != 'generic': 
                word_sing = row['word (singular)'].strip()
                plural = row['word (plural)'].strip()
                if word_sing != '':
                    if word_sing.lower() in words: print('REPEAT', word_sing)
                    words.add(word_sing.lower())
                    sing2plural[word_sing] = plural
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
    
def count_words_reddit(): 
    people, _ = get_manual_people()
    nonpeople = get_manual_nonpeople()
    all_terms = people | nonpeople
    word_time_place = defaultdict(Counter) # month_community : {word : count}
    for f in os.listdir(COMMENTS): 
        if f == 'bad_jsons': continue
        month = f.replace('RC_', '')
        with open(COMMENTS + f + '/part-00000', 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                if 'body' not in d: continue
                text = d['body'].lower()
                sr = d['subreddit'].lower()
                for term in all_terms: 
                    res = re.findall(r'\b' + re.escape(term) + r'\b', text)
                    if len(res) > 0: 
                        word_time_place[month + '_' + sr][term] += len(res)
        with open(SUBMISSIONS + 'RS_' + month + '/part-00000', 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                if 'selftext' not in d: continue
                text = d['selftext'].lower()
                sr = d['subreddit'].lower()
                for term in all_terms:
                    res = re.findall(r'\b' + re.escape(term) + r'\b', text)
                    if len(res) > 0: 
                        word_time_place[month + '_' + sr][term] += len(res)
        break # TODO: remove
    with open(LOGS + 'glossword_time_place.json', 'w') as outfile: 
        json.dump(word_time_place, outfile)

def main(): 
    count_words_reddit()

if __name__ == '__main__':
    main()