from transformers import BasicTokenizer
import sys
import json 
from nltk import ngrams
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.sql import Row, SQLContext
from functools import partial
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

def get_n_gramlist(nngramlist, toks, sr, n=2):   
    # stack overflow said this was fastest
    for s in ngrams(toks,n=n):        
        nngramlist.append((sr, ' '.join(s)))                
    return nngramlist

def get_ngrams_comment(line, tokenizer=None, per_comment=True): 
    '''
    Reddit comment
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
    Reddit post
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
        cdata = cdata.flatMap(partial(get_ngrams_comment, tokenizer=tokenizer, per_comment=per_comment))
        cdata = cdata.map(lambda n: (n, 1))
        cdata = cdata.reduceByKey(lambda n1, n2: n1 + n2)
        
        if os.path.exists(SUBS + 'RS_' + m + '/part-00000'): 
            post_path = SUBS + 'RS_' + m + '/part-00000'
        else: 
            post_path = SUBS + 'RS_v2_' + m + '/part-00000'
        pdata = sc.textFile(post_path)
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
    @inputs: 
    - per_comment: flag, where if False, counts all instances of a word 
    in a comment, otherwise if True, counts each word just once per comment
    ''' 
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

def main(): 
    count_control()
    count_sr()
    count_forum()
    sc.stop()

if __name__ == "__main__":
    main()
