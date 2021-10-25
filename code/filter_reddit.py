"""
File for using Spark to organize the data,
gather statistics about the dataset

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

conf = SparkConf()
sc = SparkContext(conf=conf)

IN_S = '/mnt/data0/corpora/reddit/submissions/'
IN_C = '/mnt/data0/corpora/reddit/comments/'
UD = '/mnt/data0/corpora/urban_dictionary/UD2019/Oct19/all_definitions.dat'
#ROOT = '/mnt/data0/lucy/manosphere/'
ROOT = '/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
SUBS = ROOT + 'data/submissions/'
COMS = ROOT + 'data/comments/'

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
            
def get_language(line): 
    #d = json.loads(line)
    #if 'body' in d: # comment
    #    text = d['body']
    #elif 'selftext' in d and 'title' in d: # submission
    #    text = d['title'] + '\n' + d['selftext']
    text = line
    lang = equilid.get_langs(text)
    if len(lang) > 1: return u''
    if len(lang) == 0: return u''
    return lang[0]

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
    
def detect_bots(): 
    '''
    This function finds users who tend to write
    the same 10-gram over and over (some bots customize
    responses to a post so we splice up their comments
    to get a better idea of repetition)
    '''
    all_data = sc.emptyRDD()
    for filename in os.listdir(COMS): 
        if filename == 'bad_jsons': continue
        m = filename.replace('RC_', '')
        cdata = sc.textFile(COMS + filename + '/part-00000')
        cdata = cdata.filter(check_valid_comment)
        cdata = cdata.flatMap(get_ngrams)
        cdata = cdata.map(lambda n: (n, 1))
        cdata = cdata.reduceByKey(lambda n1, n2: n1 + n2)
        
        if os.path.exists(SUBS + 'RS_' + m + '/part-00000'): 
            post_path = SUBS + 'RS_' + m + '/part-00000'
        else: 
            post_path = SUBS + 'RS_v2_' + m + '/part-00000'
        pdata = sc.textFile(post_path)
        pdata = pdata.flatMap(get_ngrams)
        pdata = pdata.map(lambda n: (n, 1))
        data = cdata.union(pdata)

        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        all_data = all_data.union(data)
        all_data = all_data.reduceByKey(lambda n1, n2: n1 + n2)
    all_data = all_data.filter(lambda tup: tup[1] > 100) # extensive repetition
    all_data = all_data.map(lambda n: n[0][0]) # user
    bot_users = set(all_data.collect())
    with open(LOGS + 'reddit_bots.txt', 'w') as outfile: 
        for user in bot_users: 
            outfile.write(user + '\n')

def main(): 
    detect_bots()
    sc.stop()

if __name__ == '__main__':
    main()
