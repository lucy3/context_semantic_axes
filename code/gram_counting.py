'''
This script does the following: 
- produce post and comment counts per month in jsons
- produce unigram and bigram parquets for all three discussion datasets
'''

from transformers import BasicTokenizer
import sys
import json 
from nltk import ngrams
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.sql import Row, SQLContext
from functools import partial
from helpers import check_valid_comment, check_valid_post, remove_bots, get_bot_set, get_sr_cats
import os

ROOT = '/mnt/data0/lucy/manosphere/' 
SUBS = ROOT + 'data/submissions/'
COMS = ROOT + 'data/comments/'
CONTROL = ROOT + 'data/reddit_control/'
FORUMS = ROOT + 'data/cleaned_forums/'
WORD_COUNT_DIR = ROOT + 'logs/gram_counts/'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def get_num_comments(): 
    '''
    Get number of comments per subreddit per month
    
    This is used to visualize comment counts in the count_viz notebook. 
    '''
    sr_month = defaultdict(Counter)
    for filename in os.listdir(COMS): 
        if filename == 'bad_jsons': continue
        f = filename.replace('RS_', '').replace('RC_', '').replace('v2_', '').split('.')[0]
        sr_counts = Counter()
        for part in os.listdir(COMS + filename):
            if not part.startswith('part-'): continue
            data = sc.textFile(COMS + filename + '/' + part)
            data = data.map(lambda line: (json.loads(line)['subreddit'].lower(), 1))
            data = data.reduceByKey(lambda n1, n2: n1 + n2)
            sr_counts += data.collectAsMap()
        sr_month[f] = sr_counts
    with open(LOGS + 'comment_counts.json', 'w') as outfile:
        json.dump(sr_month, outfile)
    sc.stop()
    
def get_num_posts(): 
    '''
    Get the number of posts per subreddit per month
    '''
    sr_month = defaultdict(Counter)
    for filename in tqdm(os.listdir(SUBS)): 
        if filename == 'bad_jsons': continue
        f = filename.replace('RS_', '').replace('RC_', '').replace('v2_', '').split('.')[0]
        subreddits = Counter()
        for part in os.listdir(SUBS + filename):
            if not part.startswith('part-'): continue
            with open(SUBS + filename + '/' + part, 'r') as infile: 
                for line in infile: 
                    d = json.loads(line)
                    subreddits[d['subreddit'].lower()] += 1
        for sr in subreddits: 
            sr_month[f][sr] = subreddits[sr]
    with open(LOGS + 'submission_counts.json', 'w') as outfile:
        json.dump(sr_month, outfile)

def get_n_gramlist(nngramlist, toks, sr, n=2):   
    # stack overflow said this was fastest
    for s in ngrams(toks,n=n):        
        nngramlist.append((sr, ' '.join(s)))                
    return nngramlist

def get_ngrams_comment(line, tokenizer=None, per_comment=True): 
    '''
    Bigrams and unigrams in Reddit comment
    '''
    d = json.loads(line)
    sr = d['subreddit'].lower()
    toks = tokenizer.tokenize(d['body'])
    all_grams = [(sr, i) for i in toks]
    all_grams = get_n_gramlist(all_grams, toks, sr, 2)
    if per_comment: 
        all_grams = list(set(all_grams))
    return all_grams

def get_ngrams_post(line, tokenizer=None, per_comment=True): 
    '''
    Bigrams and unigrams in Reddit comment
    '''
    d = json.loads(line)
    all_grams = set()
    sr = d['subreddit'].lower()
    toks = tokenizer.tokenize(d['selftext'])
    all_grams = [(sr, i) for i in toks]
    all_grams = get_n_gramlist(all_grams, toks, sr, 2)
    if per_comment: 
        all_grams = list(set(all_grams))
    return all_grams

def count_sr(per_comment=True): 
    '''
    Creates parquet for unigrams and bigrams in Reddit data 
    '''
    bots = get_bot_set()
    tokenizer = BasicTokenizer(do_lower_case=True)
    schema = StructType([
      StructField('word', StringType(), True),
      StructField('count', IntegerType(), True),
      StructField('community', StringType(), True),
      StructField('month', StringType(), True)
      ])
    df = sqlContext.createDataFrame([],schema)
    
    for filename in os.listdir(COMS): 
        if filename == 'bad_jsons': continue
        m = filename.replace('RC_', '')
        cdata = sc.textFile(COMS + filename + '/part-00000')
        cdata = cdata.filter(check_valid_comment)
        cdata = cdata.filter(partial(remove_bots, bot_set=bots))
        cdata = cdata.flatMap(partial(get_ngrams_comment, tokenizer=tokenizer, per_comment=per_comment))
        cdata = cdata.map(lambda n: (n, 1))
        cdata = cdata.reduceByKey(lambda n1, n2: n1 + n2)
        
        if os.path.exists(SUBS + 'RS_' + m + '/part-00000'): 
            post_path = SUBS + 'RS_' + m + '/part-00000'
        else: 
            post_path = SUBS + 'RS_v2_' + m + '/part-00000'
        pdata = sc.textFile(post_path)
        pdata = pdata.filter(partial(remove_bots, bot_set=bots))
        pdata = pdata.flatMap(partial(get_ngrams_post, tokenizer=tokenizer, per_comment=per_comment))
        pdata = pdata.map(lambda n: (n, 1))
        data = cdata.union(pdata)
        
        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        data = data.map(lambda tup: Row(word=tup[0][1], count=tup[1], community=tup[0][0], month=m))
        data_df = sqlContext.createDataFrame(data, schema)
        df = df.union(data_df)
    if per_comment: 
        outpath = WORD_COUNT_DIR + 'subreddit_counts_set'
    else: 
        outpath = WORD_COUNT_DIR + 'subreddit_counts'
    df.write.mode('overwrite').parquet(outpath)    
    
def count_control(per_comment=True):
    '''
    Creates parquet for unigrams and bigrams in Reddit control
    @inputs: 
    - per_comment: flag, where if False, counts all instances of a word 
    in a comment, otherwise if True, counts each word just once per comment
    ''' 
    bots = get_bot_set()
    tokenizer = BasicTokenizer(do_lower_case=True)
    schema = StructType([
      StructField('word', StringType(), True),
      StructField('count', IntegerType(), True),
      StructField('community', StringType(), True),
      StructField('month', StringType(), True)
      ])
    df = sqlContext.createDataFrame([],schema)
    
    for filename in os.listdir(CONTROL): 
        if filename == 'bad_jsons': continue
        m = filename.replace('RC_', '')
        file_data = sc.textFile(CONTROL + filename + '/part-00000')
        file_data = file_data.filter(partial(remove_bots, bot_set=bots))
        cdata = file_data.filter(check_valid_comment)
        pdata = file_data.filter(check_valid_post)
        
        cdata = cdata.flatMap(partial(get_ngrams_comment, tokenizer=tokenizer, per_comment=per_comment))
        cdata = cdata.map(lambda n: (n, 1))
        cdata = cdata.reduceByKey(lambda n1, n2: n1 + n2)
        
        pdata = pdata.flatMap(partial(get_ngrams_post, tokenizer=tokenizer, per_comment=per_comment))
        pdata = pdata.map(lambda n: (n, 1))
        data = cdata.union(pdata)
        
        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        data = data.map(lambda tup: Row(word=tup[0][1], count=tup[1], community=tup[0][0], month=m))
        data_df = sqlContext.createDataFrame(data, schema)
        df = df.union(data_df)
    if per_comment:
        outpath = WORD_COUNT_DIR + 'control_counts_set'
    else: 
        outpath = WORD_COUNT_DIR + 'control_counts'
    df.write.mode('overwrite').parquet(outpath)   
    
def get_ngrams_comment_forum(line, tokenizer=None, per_comment=True): 
    '''
    Gets bigrams and unigrams for forum
    '''
    d = json.loads(line)
    if d['date_post'] is None: 
        year = "None"
        month = "None"
    else: 
        date_time_str = d["date_post"].split('-')
        year = date_time_str[0]
        month = date_time_str[1]
    date_month = year + '-' + month
    toks = tokenizer.tokenize(d['text_post'])
    all_grams = [(date_month, i) for i in toks]
    all_grams = get_n_gramlist(all_grams, toks, date_month, 2)
    if per_comment: 
        all_grams = list(set(all_grams))
    return all_grams
    
def count_forum(per_comment=True): 
    '''
    We attach "FORUM_" the beginning of the community name
    to avoid incels the forum and incels the subreddit from clashing
    later when we combine dataframes. 
    
    Creates parquet for unigrams and bigrams in forums
    '''
    tokenizer = BasicTokenizer(do_lower_case=True)
    schema = StructType([
      StructField('word', StringType(), True),
      StructField('count', IntegerType(), True),
      StructField('community', StringType(), True),
      StructField('month', StringType(), True)
      ])
    df = sqlContext.createDataFrame([],schema)
    for filename in os.listdir(FORUMS):
        data = sc.textFile(FORUMS + filename)
        data = data.flatMap(partial(get_ngrams_comment_forum, tokenizer=tokenizer, per_comment=per_comment))
        data = data.map(lambda n: (n, 1))
        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        data = data.map(lambda tup: Row(word=tup[0][1], count=tup[1], community='FORUM_' + filename, month=tup[0][0]))
        data_df = sqlContext.createDataFrame(data, schema)
        df = df.union(data_df)
    if per_comment: 
        outpath = WORD_COUNT_DIR + 'forum_counts_set'
    else: 
        outpath = WORD_COUNT_DIR + 'forum_counts'
    df.write.mode('overwrite').parquet(outpath)
    
def get_total_tokens(): 
    '''
    Sum up unigrams for each dataset 
    '''
    outfile = open(WORD_COUNT_DIR + 'total_unigram_counts.txt', 'w')
    categories = get_sr_cats()
    reddit_df = sqlContext.read.parquet(WORD_COUNT_DIR + 'subreddit_counts')
    leave_out = []
    for sr in categories: 
        if categories[sr] == 'Health' or categories[sr] == 'Criticism': 
            leave_out.append(sr)
    reddit_df = reddit_df.filter(~reddit_df.community.isin(leave_out))
    unigrams = reddit_df.rdd.filter(lambda x: len(x[0].split(' ')) == 1)
    # map to month : total
    um_totals = unigrams.map(lambda x: (x[3], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    um_totals = sum(list(um_totals.values()))
    outfile.write('reddit_rel:' + str(um_totals) + '\n')
    
    forum_df = sqlContext.read.parquet(WORD_COUNT_DIR + 'forum_counts')
    unigrams = forum_df.rdd.filter(lambda x: len(x[0].split(' ')) == 1)
    um_totals = unigrams.map(lambda x: (x[3], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    um_totals = sum(list(um_totals.values()))
    outfile.write('forum_rel:' + str(um_totals) + '\n')
    
    control_df = sqlContext.read.parquet(WORD_COUNT_DIR + 'control_counts')
    unigrams = control_df.rdd.filter(lambda x: len(x[0].split(' ')) == 1)
    um_totals = unigrams.map(lambda x: (x[3], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    um_totals = sum(list(um_totals.values()))
    outfile.write('control:' + str(um_totals) + '\n')
    outfile.close()

def main(): 
#     count_control(per_comment=False)
#     count_sr(per_comment=False)
#     count_forum(per_comment=False)
#     count_control()
#     count_sr()
#     count_forum()
    get_total_tokens()
    sc.stop()

if __name__ == "__main__":
    main()
