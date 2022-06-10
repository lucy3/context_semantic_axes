"""
File for using Spark to filter out manosphere communities from
entire Reddit dataset, sample control dataset, and detect bots. 

Possible file extensions include
- .bz2
- .zst
- .xz
"""
from pyspark import SparkConf, SparkContext
import subprocess
import time
import json
import os
import csv 
from collections import Counter
from helpers import get_sr_cats
from nltk import ngrams
from functools import partial
import string

conf = SparkConf()
sc = SparkContext(conf=conf)

IN_S = '/mnt/data0/corpora/reddit/submissions/'
IN_C = '/mnt/data0/corpora/reddit/comments/'
UD = '/mnt/data0/corpora/urban_dictionary/UD2019/Oct19/all_definitions.dat'
ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
SUBS = ROOT + 'data/submissions/'
COMS = ROOT + 'data/comments/'
CONTROL = ROOT + 'data/reddit_control/'

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

def check_duplicate_months(d, months): 
    """
    There is a month that occurs twice in the pushshift data. 
    Does it contain the same comments in both files? 
   """
    for dups in months: 
        dup1 = dups[0]
        dup2 = dups[1]
        unpack_file(d, dup1)
        filename = dup1.split('.')[0]
        # map to IDs, collect as set
        data = sc.textFile(d + filename)
        data = data.map(lambda line: json.loads(line)['id'])
        ids1 = set(data.collect())
        pack_file(d, dup1)
        
        unpack_file(d, dup2)
        filename = dup2.split('.')[0]
        # map to IDs, collect as set
        data = sc.textFile(d + filename)
        data = data.map(lambda line: json.loads(line)['id'])
        ids2 = set(data.collect())
        pack_file(d, dup2)
        
        # check that the IDs are the same for both files
        if ids1 != ids2: 
            print("DIFFERENCE", len(ids1 - ids2), len(ids2 - ids1))
        else: 
            print("IT IS FINE!!!!!!!!!!")

def get_dumb_lines(line): 
    try: 
        json.loads(line)
    except json.decoder.JSONDecodeError:
        return True
    return False

def extract_relevant_subreddits(in_d, out_d): 
    """
    Creates new files containing 
    jsons of only relevant subreddits
    @inputs: 
    - in_d: folder with inputs
    - out_d: folder with outputs
    """
    relevant_subs = set()
    with open(DATA + 'subreddit_names.txt', 'r') as infile: 
        for line in infile: 
            name = line.strip().lower()
            if name.startswith('/r/'): name = name[3:]
            if name.startswith('r/'): name = name[2:]
            if name.endswith('/'): name = name[:-1]
            relevant_subs.add(name)
    for f in os.listdir(in_d):
        filename = f.split('.')[0]
        if os.path.isdir(out_d + filename): continue # skip ones we already have
        unpack_file(in_d, f)
        data = sc.textFile(in_d + filename)
        not_wanted = data.filter(get_dumb_lines).collect()
        data = data.filter(lambda line: not get_dumb_lines(line))
        rel_data = data.filter(lambda line: 'subreddit' in json.loads(line) and \
                    json.loads(line)['subreddit'].lower() in relevant_subs)
        rel_data.coalesce(1).saveAsTextFile(out_d + filename)
        if len(not_wanted) > 0: 
            # write bad lines to bad_jsons
            with open(out_d + 'bad_jsons/' + filename + '.txt', 'w') as outfile: 
                for line in not_wanted:
                    outfile.write(line + '\n') 
        pack_file(in_d, f)
        
def get_month_totals(): 
    '''
    Remove love_shy, Criticism and Health subreddits when taking in 
    consideration per-month total post + comment counts 
    '''
    categories = get_sr_cats() 
    month_totals = Counter()
    with open(LOGS + 'submission_counts.json', 'r') as infile:
        sr_month = json.load(infile)
    for month in sr_month: 
        for sr in sr_month[month]: 
            if categories[sr] != 'Health' and categories[sr] != 'Criticism':
                month_totals[month] += sr_month[month][sr]
    # remove Criticism and Health subreddits 
    with open(LOGS + 'comment_counts.json', 'r') as infile:
        sr_coms = json.load(infile)
    for month in sr_coms: 
        for sr in sr_coms[month]: 
            if categories[sr] != 'Health' and categories[sr] != 'Criticism':
                month_totals[month] += sr_coms[month][sr]
    with open(LOGS + 'forum_count.json', 'r') as infile:
        forum_month = json.load(infile)
    for month in forum_month: 
        for forum in forum_month[month]: 
            if forum == 'love_shy': continue
            if month == 'None-None' or month == '1970-01': continue
            month_totals[month] += forum_month[month][forum]
    return month_totals
        
def sample_reddit_control(): 
    '''
    Sample a set of Reddit that is in equal size to manosphere dataset
    
    By the time I ran this function, subreddit_names had Health and
    Criticism subreddits removed from the list. 
    '''
    # get total number of posts + comments per month across communities 
    month_totals = get_month_totals()
    # get subreddits that are in our dataset 
    relevant_subs = set()
    with open(DATA + 'subreddit_names.txt', 'r') as infile: 
        for line in infile: 
            name = line.strip().lower()
            if name.startswith('/r/'): name = name[3:]
            if name.startswith('r/'): name = name[2:]
            if name.endswith('/'): name = name[:-1]
            relevant_subs.add(name)

    seed = 0
    for month in sorted(month_totals.keys()):
        if os.path.exists(DATA + 'reddit_control/' + month): 
            continue
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
        
        # if inputs exist, filter only subreddits not in our dataset 
        if sub_input != '' and com_input != '': 
            unpack_file(IN_S, sub_input)
            unpack_file(IN_C, com_input) 
            filename = sub_input.split('.')[0]
            data = sc.textFile(IN_S + filename) 
            data = data.filter(lambda line: not get_dumb_lines(line))
            sub_data = data.filter(lambda line: 'subreddit' in json.loads(line) and \
                        json.loads(line)['subreddit'].lower() not in relevant_subs)
        
            filename = com_input.split('.')[0]
            data = sc.textFile(IN_C + filename) 
            data = data.filter(lambda line: not get_dumb_lines(line))
            com_data = data.filter(lambda line: 'subreddit' in json.loads(line) and \
                        json.loads(line)['subreddit'].lower() not in relevant_subs)

            # sample from posts and comments
            sample_size = month_totals[month]
            print("Sampling", sample_size, "from", month)
            all_data = com_data.union(sub_data)
            sampled_data = sc.parallelize(all_data.takeSample(False, sample_size, seed))
            sampled_data.coalesce(1).saveAsTextFile(DATA + 'reddit_control/' + month)
    
            # pack posts and comments
            pack_file(IN_S, sub_input) 
            pack_file(IN_C, com_input) 
        print("TIME:", time.time() - start)
        
def extract_subreddits_main(): 
    '''
    Get relevant subreddits for comments and posts.
    '''
    in_d = '/mnt/data0/corpora/reddit/comments/'
    out_d = '/mnt/data0/lucy/manosphere/data/comments/'
    extract_relevant_subreddits(in_d, out_d)
    in_d = '/mnt/data0/corpora/reddit/submissions/'
    out_d = '/mnt/data0/lucy/manosphere/data/submissions/'
    extract_relevant_subreddits(in_d, out_d)
    
def check_duplicates_main(): 
    '''
    In the downloaded pushshift data, a few months
    appear twice. This is to check that they are the same.
    '''
    check_duplicate_months(IN_C, [('RC_2018-10.xz', 'RC_2018-10.zst')])
    check_duplicate_months(IN_S, [('RS_2017-11.bz2', 'RS_2017-11.xz')])
    check_duplicate_months(IN_S, [('RS_2017-07.bz2', 'RS_2017-07.xz')])
    
def get_n_gramlist(nngramlist, toks, author, n=2):   
    # stack overflow said this was fastest
    for s in ngrams(toks,n=n):        
        nngramlist.append((author, ' '.join(s)))                
    return nngramlist
    
def get_ngrams(line): 
    '''
    Gets 10-grams for each post/comment.
    This is using white-space splitting because it is faster and
    tokenization should not affect things if there are large amounts 
    of copied text between posts/comments
    '''
    d = json.loads(line)
    if 'author' not in d: return []
    author = d['author'].lower()
    if 'body' in d: 
        toks = d['body'].split()
    elif 'selftext' in d: 
        toks = d['selftext'].split()
    else: 
        return []
    all_grams = get_n_gramlist([], toks, author, 10)
    all_grams = list(set(all_grams))
    return all_grams

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
    
def detect_bots(): 
    '''
    This function finds users who tend to write
    the same 10-gram over and over (some bots customize
    responses to a post so we splice up their comments
    to get a better idea of repetition)
    '''
    bot_users = set()
    for filename in os.listdir(COMS): 
        if filename == 'bad_jsons': continue
        m = filename.replace('RC_', '')
        cdata = sc.textFile(COMS + filename + '/part-00000')
        cdata = cdata.filter(check_valid_comment)
        cdata = cdata.flatMap(get_ngrams).map(lambda n: (n, 1)).reduceByKey(lambda n1, n2: n1 + n2)
        
        if os.path.exists(SUBS + 'RS_' + m + '/part-00000'): 
            post_path = SUBS + 'RS_' + m + '/part-00000'
        else: 
            post_path = SUBS + 'RS_v2_' + m + '/part-00000'
        pdata = sc.textFile(post_path)
        pdata = pdata.flatMap(get_ngrams).map(lambda n: (n, 1)).reduceByKey(lambda n1, n2: n1 + n2)
        data = cdata.union(pdata).reduceByKey(lambda n1, n2: n1 + n2)

        file_data = sc.textFile(CONTROL + m + '/part-00000')
        cdata = file_data.filter(check_valid_comment)
        pdata = file_data.filter(check_valid_post)
        cdata = cdata.flatMap(get_ngrams).map(lambda n: (n, 1)).reduceByKey(lambda n1, n2: n1 + n2)
        pdata = pdata.flatMap(get_ngrams).map(lambda n: (n, 1)).reduceByKey(lambda n1, n2: n1 + n2)
        data_control = cdata.union(pdata).reduceByKey(lambda n1, n2: n1 + n2)

        data = data.union(data_control).reduceByKey(lambda n1, n2: n1 + n2)
        
        data = data.filter(lambda tup: tup[1] > 100) # extensive repetition
        data = data.map(lambda n: n[0][0]) # user
        bot_users.update(data.collect())
    with open(LOGS + 'reddit_bots.txt', 'w') as outfile: 
        for user in bot_users: 
            outfile.write(user + '\n')
            
def count_posts_per_subreddit(): 
    '''
    For the control dataset, we want to focus on
    the top 1000 subreddits, based on post count. 
    '''
    # get subreddits that are in our dataset 
    relevant_subs = set()
    with open(DATA + 'subreddit_names.txt', 'r') as infile: 
        for line in infile: 
            name = line.strip().lower()
            if name.startswith('/r/'): name = name[3:]
            if name.startswith('r/'): name = name[2:]
            if name.endswith('/'): name = name[:-1]
            relevant_subs.add(name)

    for f in os.listdir(IN_S):
        start = time.time() 
        filename = f.split('.')[0]
        if os.path.isdir(DATA + 'all_reddit_post_counts/' + filename): continue # skip ones we already have

        unpack_file(IN_S, f)
        data = sc.textFile(IN_S + filename) 
        data = data.filter(lambda line: not get_dumb_lines(line))
        sub_data = data.filter(lambda line: 'subreddit' in json.loads(line) and \
                    json.loads(line)['subreddit'].lower() not in relevant_subs)
        sub_data = sub_data.map(lambda line: (json.loads(line)['subreddit'].lower(), 1))
        sub_data = sub_data.reduceByKey(lambda n1, n2: n1 + n2).map(lambda tup: tup[0] + ' ' + str(tup[1]))
        sub_data.coalesce(1).saveAsTextFile(DATA + 'all_reddit_post_counts/' + filename)

        pack_file(IN_S, f) 
        print("TIME:", time.time() - start)
        
def get_top_subreddits(): 
    '''
    count_posts_per_subreddit()
    needs to be run before this function. 
    
    Then, each part-00000 file in DATA + 'all_reddit_post_counts/'
    needs to be concatenated, and that file is called 'all_post_counts'
    Then, we use spark to read in that file, map, and reduce by key, and write out
    the top 1000 subreddits that do not start with 'u' (user). 
    '''
    data = sc.textFile(DATA + 'all_reddit_post_counts/all_post_counts')
    data = data.filter(lambda x: not x.startswith('u_') and ' ' in x)
    data = data.map(lambda x: (x.split(' ')[0], int(x.split(' ')[1]))).reduceByKey(lambda n1, n2: n1 + n2)
    data = data.filter(lambda tup: tup[1] > 100000)
    data = Counter(data.collectAsMap())
    with open(DATA + 'all_reddit_post_counts/top_subreddits.txt', 'w') as outfile: 
        for tup in data.most_common(): 
            outfile.write(tup[0] + ' ' + str(tup[1]) + '\n')
            
def content_has_vocab(line, vocab=set()): 
    '''
    @inputs: 
    - vocab: vocab words to find
    
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
    bigrams = set()
    for s in ngrams(toks,n=2):        
        bigrams.add(' '.join(s))  
    overlap = (unigrams & vocab) | (bigrams & vocab)
    if len(overlap) > 0: 
        return True
    return False
            
def extract_mainstream_subreddits(in_d, out_d, vocab, relevant_subs): 
    """
    Creates new files containing 
    jsons of only relevant subreddits
    @inputs: 
    - in_d: folder with inputs
    - out_d: folder with outputs
    """
    all_files = sorted(os.listdir(in_d))
    for f in all_files:
        year = f.split('-')[0].split('_')[-1]
        if year in ['2005', '2006', '2020', '2021']: continue
        filename = f.split('.')[0]
        if os.path.isdir(out_d + filename): continue # skip ones we already have
        unpack_file(in_d, f)
        data = sc.textFile(in_d + filename)
        not_wanted = data.filter(get_dumb_lines).collect()
        data = data.filter(lambda line: not get_dumb_lines(line))
        rel_data = data.filter(lambda line: 'subreddit' in json.loads(line) and \
                    json.loads(line)['subreddit'].lower() in relevant_subs)
        rel_data = rel_data.filter(partial(content_has_vocab, vocab=vocab))
        rel_data.coalesce(1).saveAsTextFile(out_d + filename)
        if len(not_wanted) > 0: 
            # write bad lines to bad_jsons
            with open(out_d + 'bad_jsons/' + filename + '.txt', 'w') as outfile: 
                for line in not_wanted:
                    outfile.write(line + '\n') 
        pack_file(in_d, f)
        
def extract_lexical_innovations(): 
    '''
    Get relevant subreddits for comments and posts.
    '''
    # load top subreddits
    N = 500
    top_n_subreddits = set()
    with open(DATA + 'all_reddit_post_counts/top_subreddits.txt', 'r') as infile: 
        line_count = 0
        for line in infile: 
            top_n_subreddits.add(line.strip().split(' ')[0])
            line_count += 1
            if line_count == N: break
    vocab = set()
    with open(LOGS + 'lexical_innovations.txt', 'r') as infile: 
        for line in infile: 
            vocab.add(line.strip())
    in_d = '/mnt/data0/corpora/reddit/comments/'
    out_d = '/mnt/data0/lucy/manosphere/data/mainstream/'
    extract_mainstream_subreddits(in_d, out_d, vocab, top_n_subreddits)
    in_d = '/mnt/data0/corpora/reddit/submissions/'
    out_d = '/mnt/data0/lucy/manosphere/data/mainstream/'
    extract_mainstream_subreddits(in_d, out_d, vocab, top_n_subreddits)
    
def extract_select_subreddits(in_d, out_d, relevant_subs): 
    """
    Creates new files containing 
    jsons of only relevant subreddits
    @inputs: 
    - in_d: folder with inputs
    - out_d: folder with outputs
    """
    all_files = sorted(os.listdir(in_d))
    for f in all_files:
        year = f.split('-')[0].split('_')[-1]
        if year in ['2005', '2006', '2020', '2021']: continue
        filename = f.split('.')[0]
        if os.path.isdir(out_d + filename): continue # skip ones we already have
        unpack_file(in_d, f)
        data = sc.textFile(in_d + filename)
        not_wanted = data.filter(get_dumb_lines).collect()
        data = data.filter(lambda line: not get_dumb_lines(line))
        rel_data = data.filter(lambda line: 'subreddit' in json.loads(line) and \
                    json.loads(line)['subreddit'].lower() in relevant_subs)
        rel_data.coalesce(1).saveAsTextFile(out_d + filename)
        if len(not_wanted) > 0: 
            # write bad lines to bad_jsons
            with open(out_d + 'bad_jsons/' + filename + '.txt', 'w') as outfile: 
                for line in not_wanted:
                    outfile.write(line + '\n') 
        pack_file(in_d, f)
    
def filter_reddit_dating(): 
    subreddit_list = ['relationships', 'relationship_advice', 'dating_advice', 'breakups', 'dating']
    in_d = '/mnt/data0/corpora/reddit/comments/'
    out_d = '/mnt/data0/lucy/manosphere/data/reddit_dating/'
    extract_select_subreddits(in_d, out_d, subreddit_list)
    in_d = '/mnt/data0/corpora/reddit/submissions/'
    out_d = '/mnt/data0/lucy/manosphere/data/reddit_dating/'
    extract_select_subreddits(in_d, out_d, subreddit_list)

def main(): 
    #check_duplicates_main()
    #extract_subreddits_main()
    #sample_reddit_control()
    #detect_bots()
    #count_posts_per_subreddit()
    #get_top_subreddits()
    #extract_lexical_innovations()
    filter_reddit_dating()
    sc.stop()

if __name__ == '__main__':
    main()
