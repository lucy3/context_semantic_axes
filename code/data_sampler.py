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

#ROOT = '/global/scratch/lucy3_li/manosphere/'
ROOT = '/mnt/data0/lucy/manosphere/'
POSTS = ROOT + 'data/submissions/'
LOGS = ROOT + 'logs/'
COMMENTS = ROOT + 'data/comments/'
FORUMS = ROOT + 'data/cleaned_forums/'
REDDIT_OUT = LOGS + 'reddit_sample'
FORUM_OUT = LOGS + 'forum_sample'

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

def main(): 
    #sample_reddit()
    sample_forums()

if __name__ == '__main__':
    main()
