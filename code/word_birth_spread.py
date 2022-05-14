"""
Gets earliest date that
vocab terms appear across reddit_rel
and forum_rel
"""

#from pyspark import SparkConf, SparkContext
from functools import partial
import subprocess
from helpers import get_vocab
import string
from nltk import ngrams
import json
import os
import time

IN_S = '/mnt/data0/corpora/reddit/submissions/'
IN_C = '/mnt/data0/corpora/reddit/comments/'
ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
SUBS = ROOT + 'data/submissions/'
COMS = ROOT + 'data/comments/'
FORUMS = ROOT + 'data/cleaned_forums/'
CONTROL = ROOT + 'data/reddit_control/'

def get_dumb_lines(line): 
    try: 
        json.loads(line)
    except json.decoder.JSONDecodeError:
        return True
    return False

def month_year_iter(start, end):
    '''
    https://stackoverflow.com/questions/5734438/how-to-create-a-month-iterator
    '''
    start_contents = start.split('-')
    start_month = int(start_contents[1])
    start_year = int(start_contents[0])
    end_contents = end.split('-')
    end_month = int(end_contents[1])
    end_year = int(end_contents[0])
    ym_start= 12*start_year + start_month - 1
    ym_end= 12*end_year + end_month - 1
    for ym in range( ym_start, ym_end ):
        y, m = divmod( ym, 12 )
        month = str(m + 1)
        if len(month) == 1: 
            month = '0' + month
        yield str(y) + '-' + month
        
def get_n_gramlist(toks, n=2):   
    # stack overflow said this was fastest
    nngramlist = set()
    for s in ngrams(toks,n=n):        
        nngramlist.add(' '.join(s))              
    return nngramlist
        
def get_subreddit_vocab(line, vocab=set()): 
    '''
    @inputs: 
    - vocab: remaining vocab to find
    
    This function uses a fast/basic tokenizer, since
    we are looking for words over the entirety of Reddit
    '''
    d = json.loads(line)
    if 'selftext' in d: 
        text = d['selftext'].lower()
    elif 'body' in d: 
        text = d['body'].lower()
    else: 
        text = ''
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    toks = text.split()
    unigrams = set(toks)
    bigrams = get_n_gramlist(toks, 2)
    overlap = (unigrams & vocab) | (bigrams & vocab)
    ret = []
    for w in overlap: 
        ret.append((w, [d['subreddit'].lower()]))
    return ret

def unpack_file(d, f):
    start = time.time()
    print("Unpacking", d, f)
    if f.endswith('.xz'): 
        p = subprocess.Popen(['xz', '--keep', '--decompress', f], cwd=d)
        p.wait()
    elif f.endswith('.zst'): 
        p = subprocess.Popen(['unzstd', f], cwd=d)
        p.wait()
    elif f.endswith('.bz2'): 
        p = subprocess.Popen(['bzip2', '-dk', f], cwd=d) 
        p.wait()
    else: 
        print("NOT IMPLEMENTED")
    print("TIME:", time.time()-start)
    
def pack_file(d, f): 
    filename = f.split('.')[0]
    print("Deleting", d, filename)
    p = subprocess.Popen(['rm', filename], cwd=d)
    p.wait()

def find_word_birthdates_reddit(sc): 
    '''
    Searches through Reddit comments and submissions
    for the first post and community in which a vocab term is used 
    
    '''
    remaining_vocab = set(get_vocab())

    seed = 0
    min_month = '2005-11'
    max_month = '2019-12'
    for month in month_year_iter(min_month, max_month):
        start = time.time() 
        # check if path exists
        sub_input = ''
        for suffix in ['.xz', '.zst', '.bz2']:
            if os.path.exists(IN_S + 'RS_' + month + suffix): 
                sub_input = 'RS_' + month + suffix
            elif os.path.exists(IN_S + 'RS_v2_' + month + suffix): 
                sub_input = 'RS_v2_' + month + suffix
            if sub_input != '': break
        com_input = ''
        for suffix in ['.xz', '.zst', '.bz2']:
            if os.path.exists(IN_C + 'RC_' + month + suffix): 
                com_input = 'RC_' + month + suffix
            if com_input != '': break
         
        if sub_input != '' and com_input != '': 
            unpack_file(IN_S, sub_input)
            unpack_file(IN_C, com_input) 
            filename = sub_input.split('.')[0]
            data = sc.textFile(IN_S + filename) 
            data = data.filter(lambda line: not get_dumb_lines(line))
            sub_data = data.filter(lambda line: 'subreddit' in json.loads(line))
            sub_data = sub_data.flatMap(partial(get_subreddit_vocab, vocab=remaining_vocab))
        
            filename = com_input.split('.')[0]
            data = sc.textFile(IN_C + filename) 
            data = data.filter(lambda line: not get_dumb_lines(line))
            com_data = data.filter(lambda line: 'subreddit' in json.loads(line))
            com_data = com_data.flatMap(partial(get_subreddit_vocab, vocab=remaining_vocab))
            both_data = sub_data.union(com_data)
            both_data = both_data.reduceByKey(lambda n1, n2: n1 + n2) # {vocab term : [list of subreddits]}
            both_data = both_data.collectAsMap()
            
            found_vocab = set(list(both_data.keys()))
            remaining_vocab = remaining_vocab - found_vocab
            
            with open(LOGS + 'word_births/' + month + '.json', 'w') as outfile: 
                json.dump(both_data, outfile)
    
            # pack posts and comments
            pack_file(IN_S, sub_input) 
            pack_file(IN_C, com_input) 
            
            if len(remaining_vocab) == 0: 
                print("ALL VOCAB FOUND BY", month)
                break
        print("TIME:", time.time() - start)
        
def get_forum_vocab(line, vocab=set()): 
    '''
    @output: 
    list of (word, [date])
    '''
    d = json.loads(line)
    idx = str(d['id_post'])
    if d['date_post'] is None: 
        return []
    date_time_str = d["date_post"].split('-')
    year = date_time_str[0]
    month = date_time_str[1]
    date_month = year + '-' + month
    text = d['text_post'].lower()
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    toks = text.split()
    unigrams = set(toks)
    bigrams = get_n_gramlist(toks, 2)
    overlap = (unigrams & vocab) | (bigrams & vocab)
    ret = []
    for w in overlap: 
        ret.append((w, date_month))
    return ret

def get_min_month(n1, n2): 
    '''
    returns the lesser of the two months
    '''
    if n1 == n2: 
        return n1
    n1_split = n1.split('-')
    year1 = int(n1_split[0])
    month1 = int(n1_split[1])
    n2_split = n2.split('-')
    year2 = int(n2_split[0])
    month2 = int(n2_split[1])
    if year1 < year2: 
        return n1
    elif year2 < year1: 
        return n2
    else: 
        # same year
        if month1 < month2: 
            return n1
        else: 
            return n2
        
def find_word_birthdates_forum(sc): 
    vocab = set(get_vocab())
    
    for filename in os.listdir(FORUMS):
        data = sc.textFile(FORUMS + filename)
        data = data.flatMap(partial(get_forum_vocab, vocab=vocab)).reduceByKey(get_min_month)
        data = data.collectAsMap()
        
        with open(LOGS + 'word_births/FORUM_' + filename + '.json', 'w') as outfile: 
            json.dump(data, outfile)
            
def get_overall_birthdateplace(): 
    '''
    This function loads up all of the jsons in word_births
    and then gets the earliest birth month and places. 
    '''
    birth_date = {} #  {vocab : year-month }
    birth_places = {} # {vocab : [list of subreddits or 'FORUM_' + forum]}
    for filename in os.listdir(LOGS + 'word_births'): 
        if not filename.endswith('json'): continue
        if filename.startswith('FORUM'): 
            # each file is a forum
            forum = filename.replace('.json', '') # FORUM_forum
            with open(LOGS + 'word_births/' + filename, 'r') as infile: 
                d = json.load(infile)
            for w in d: 
                curr_date = d[w]
                if w not in birth_date: 
                    birth_date[w] = curr_date
                    birth_places[w] = [forum]
                else: 
                    other_date = birth_date[w]
                    min_month = get_min_month(curr_date, other_date)
                    if min_month == curr_date and min_month == other_date: 
                        # add forum to birth_places
                        birth_places[w].append(forum)
                    elif min_month == curr_date: 
                        # earliest date seen so far 
                        birth_date[w] = curr_date
                        birth_places[w] = [forum]
                    # else, min_month is other_date and do nothing
        else: 
            # each file is a month of reddit
            with open(LOGS + 'word_births/' + filename, 'r') as infile: 
                d = json.load(infile)
            curr_date = filename.replace('.json', '')
            for w in d: 
                subreddits = list(set(d[w]))
                if w not in birth_date: 
                    birth_date[w] = curr_date
                    birth_places[w] = subreddits
                else: 
                    other_date = birth_date[w]
                    min_month = get_min_month(curr_date, other_date)
                    if min_month == curr_date and min_month == other_date: 
                        birth_places[w].extend(subreddits)
                    elif min_month == curr_date: 
                        birth_date[w] = curr_date
                        birth_places[w] = subreddits
                        
    res = {}
    for w in birth_date: 
        res[w] = [birth_date[w], birth_places[w]]
    with open(LOGS + 'overall_word_births.json', 'w') as outfile: 
        json.dump(res, outfile)

def main():
    #conf = SparkConf()
    #sc = SparkContext(conf=conf)
    #find_word_birthdates_reddit(sc)
    #find_word_birthdates_forum(sc)
    get_overall_birthdateplace()
    #sc.stop()
    

if __name__ == '__main__':
    main()
