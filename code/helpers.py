import csv
from collections import defaultdict
import json

ROOT = '/mnt/data0/lucy/manosphere/'
PEOPLE_FILE = ROOT + 'data/people.csv'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'

def get_vocab(): 
    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                words.append(row['entity'].lower())
    return words

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

def check_valid_comment(line): 
    '''
    For Reddit comments
    '''
    comment = json.loads(line)
    return 'body' in comment and comment['body'].strip() != '[deleted]' \
            and comment['body'].strip() != '[removed]'

def check_valid_post(line): 
    '''
    For Reddit posts
    '''
    d = json.loads(line)
    return 'selftext' in d

def get_bot_set(): 
    bots = set()
    with open(ROOT + 'logs/reddit_bots.txt', 'r') as infile: 
        for line in infile: 
            bots.add(line.strip())
    return bots
    
def remove_bots(line, bot_set=set()): 
    d = json.loads(line)
    return 'author' in d and d['author'] not in bot_set