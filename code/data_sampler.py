"""
Reservoir sampling 

Single pass over each subreddit and forum
to sample 30 posts from each. 
"""
import json
import os
import sys
from collections import defaultdict, Counter
import random
import csv
import re

#ROOT = '/global/scratch/lucy3_li/manosphere/'
ROOT = '/mnt/data0/lucy/manosphere/'
POSTS = ROOT + 'data/submissions/'
LOGS = ROOT + 'logs/'
COMMENTS = ROOT + 'data/comments/'
FORUMS = ROOT + 'data/cleaned_forums/'
REDDIT_OUT = LOGS + 'reddit_sample'
FORUM_OUT = LOGS + 'forum_sample'
GLOSSWORD_OUT = LOGS + 'glossword_sample'
PEOPLE_FILE = ROOT + 'data/people.csv'

def get_manual_people(): 
    """
    get list of words, add plural forms
    copied from find_people.py
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

def valid_line(text): 
    return text.strip() != '[removed]' and text.strip() != '[deleted]' and text.strip() != ''

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

def sample_reddit(): 
    categories = get_sr_cats()
    k = 25
    samples = defaultdict(list) # {cat: [30 comments]}
    subreddit_count = Counter() # {cat: number of times seen}
    for f in os.listdir(COMMENTS):
        if f == 'bad_jsons': continue 
        month = f.replace('RC_', '')
        print(month)
        line_number = 0
        with open(COMMENTS + f + '/part-00000', 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                text = d['body']
                sr = d['subreddit'].lower()
                cat = categories[sr]
                subreddit_count[cat] += 1
                if len(samples[cat]) < k and valid_line(text): 
                    samples[cat].append((line_number, month, sr, text))
                elif valid_line(text): 
                    idx = int(random.random() * subreddit_count[cat])
                    if idx < k: 
                        samples[cat][idx] = (line_number, month, sr, text)
                line_number += 1
        if os.path.exists(POSTS + 'RS_' + month + '/part-00000'): 
            post_path = POSTS + 'RS_' + month + '/part-00000'
        else: 
            post_path = POSTS + 'RS_v2_' + month + '/part-00000'
        with open(post_path, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                text = d['selftext']
                sr = d['subreddit'].lower()
                cat = categories[sr]
                subreddit_count[cat] += 1
                if len(samples[cat]) < k and valid_line(text): 
                    samples[cat].append((line_number, month, sr, text))
                elif valid_line(text): 
                    idx = int(random.random() * subreddit_count[cat])
                    if idx < k: 
                        samples[cat][idx] = (line_number, month, sr, text)
                line_number += 1
    with open(REDDIT_OUT, 'w') as outfile: 
        writer = csv.writer(outfile, delimiter='\t')
        for cat in samples: 
            for tup in samples[cat]: 
                writer.writerow([cat, str(tup[0]), tup[1], tup[2], tup[3]])

def sample_forums(): 
    k = 25
    samples = defaultdict(list)
    forum_count = Counter()
    for f in os.listdir(FORUMS):
        print(f) 
        line_number = 0
        with open(FORUMS + f, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                text = d['text_post']
                forum_count[f] += 1
                if len(samples[f]) < k:
                    samples[f].append((line_number, f, text))
                else: 
                    idx = int(random.random() * forum_count[f])
                    if idx < k: 
                        samples[f][idx] = (line_number, f, text)
                line_number += 1
    with open(FORUM_OUT, 'w') as outfile: 
        writer = csv.writer(outfile, delimiter='\t')
        for f in samples: 
            for tup in samples[f]: 
                writer.writerow([tup[1], tup[0], tup[2]])
                
def sample_by_glossword(): 
    '''
    Get 5 sentences per glossary word
    For words with both singular and plural forms
    this would then be 10 sentences per word 
    '''
    all_words, _ = get_manual_people()
    k = 5
    samples = defaultdict(list) # {word: [5 comments]}
    glossword_count = Counter() # {word: number of times seen}
    
    # through reddit comments and posts
    for f in os.listdir(COMMENTS):
        if f == 'bad_jsons': continue 
        month = f.replace('RC_', '')
        print(month)
        line_number = 0
        with open(COMMENTS + f + '/part-00000', 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                text = d['body']
                sr = d['subreddit'].lower()
                for word in all_words: 
                    if word in text: # fast check
                        if re.search(r'\b' + re.escape(word) + r'\b', text) is not None: 
                            glossword_count[word] += 1
                            if len(samples[word]) < k and valid_line(text): 
                                samples[word].append((line_number, month, sr, text))
                            elif valid_line(text): 
                                idx = int(random.random() * glossword_count[word])
                                if idx < k: 
                                    samples[word][idx] = (line_number, month, sr, text)
                line_number += 1
                
        if os.path.exists(POSTS + 'RS_' + month + '/part-00000'): 
            post_path = POSTS + 'RS_' + month + '/part-00000'
        else: 
            post_path = POSTS + 'RS_v2_' + month + '/part-00000'
        with open(post_path, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                text = d['selftext']
                sr = d['subreddit'].lower()
                for word in all_words: 
                    if word in text: # fast check
                        if re.search(r'\b' + re.escape(word) + r'\b', text) is not None: 
                            glossword_count[word] += 1
                            if len(samples[word]) < k and valid_line(text): 
                                samples[word].append((line_number, month, sr, text))
                            elif valid_line(text): 
                                idx = int(random.random() * glossword_count[word])
                                if idx < k: 
                                    samples[word][idx] = (line_number, month, sr, text)
                line_number += 1
                
    # through forums
    for f in os.listdir(FORUMS):
        print(f) 
        line_number = 0
        with open(FORUMS + f, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                text = d['text_post']
                for word in all_words: 
                    if word in text: # fast check
                        if re.search(r'\b' + re.escape(word) + r'\b', text) is not None: 
                            glossword_count[word] += 1
                            if len(samples[word]) < k:
                                samples[word].append((line_number, 'no-month', f, text))
                            else: 
                                idx = int(random.random() * glossword_count[word])
                                if idx < k: 
                                    samples[word][idx] = (line_number, 'no-month', f, text)
                            line_number += 1
    
    with open(GLOSSWORD_OUT, 'w') as outfile: 
        writer = csv.writer(outfile, delimiter='\t')
        for word in samples: 
            for tup in samples[word]: 
                writer.writerow([word, str(tup[0]), tup[1], tup[2], tup[3]])

def main(): 
    #sample_reddit()
    #sample_forums()
    sample_by_glossword()

if __name__ == '__main__':
    main()
