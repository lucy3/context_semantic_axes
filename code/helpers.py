import csv
from collections import defaultdict

ROOT = '/mnt/data0/lucy/manosphere/'
PEOPLE_FILE = ROOT + 'data/people.csv'

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