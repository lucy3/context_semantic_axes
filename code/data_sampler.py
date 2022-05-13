"""
Reservoir sampling 

Single pass over each subreddit and forum
to sample a specific number of posts from each. 

This is used for evaluating different NER models. 
"""
import json
import os
import sys
from collections import defaultdict, Counter
import random
import csv
import re
from helpers import get_sr_cats, valid_line, get_manual_people

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

def sample_reddit(): 
    '''
    For NER evaluation, k = 25
    '''
    categories = get_sr_cats()
    k = 25
    samples = defaultdict(list) # {cat: [25 comments]}
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
                if cat == 'Health' or cat == 'Criticism': continue
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
                if cat == 'Health' or cat == 'Criticism': continue
                subreddit_count[cat] += 1
                if len(samples[cat]) < k and valid_line(text): 
                    samples[cat].append((line_number, month, sr, text))
                elif valid_line(text): 
                    idx = int(random.random() * subreddit_count[cat])
                    if idx < k: 
                        samples[cat][idx] = (line_number, month, sr, text)
                line_number += 1
        
    with open(REDDIT_OUT + '_' + str(k), 'w') as outfile: 
        writer = csv.writer(outfile, delimiter='\t')
        for cat in samples: 
            for tup in samples[cat]: 
                writer.writerow([cat, str(tup[0]), tup[1], tup[2], tup[3]])

def sample_forums(): 
    '''
    For NER evaluation, k = 25
    '''
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
    with open(FORUM_OUT + '_' + str(k), 'w') as outfile: 
        writer = csv.writer(outfile, delimiter='\t')
        for f in samples: 
            for tup in samples[f]: 
                writer.writerow([tup[1], tup[0], tup[2]])
                
def sample_by_glossword(): 
    '''
    Get 2 sentences per glossary word
    For words with both singular and plural forms
    
    We do not use this because it produces too many examples
    for human to go through. 
    '''
    all_words, _ = get_manual_people()
    categories = get_sr_cats()
    k = 2
    samples = defaultdict(list) # {word: [comments]}
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
                cat = categories[sr]
                if cat == 'Health' or cat == 'Criticism': continue

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
                cat = categories[sr]
                if cat == 'Health' or cat == 'Criticism': continue

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
    sample_reddit()
    sample_forums()

if __name__ == '__main__':
    main()
