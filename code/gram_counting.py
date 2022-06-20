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
from pyspark.sql.functions import col
from functools import partial
from helpers import check_valid_comment, check_valid_post, remove_bots, get_bot_set, get_sr_cats
from collections import defaultdict
import os
from tqdm import tqdm

ROOT = '/mnt/data0/lucy/manosphere/' 
SUBS = ROOT + 'data/submissions/'
COMS = ROOT + 'data/comments/'
CONTROL = ROOT + 'data/reddit_dating/'
FORUMS = ROOT + 'data/cleaned_forums/'
WORD_COUNT_DIR = ROOT + 'logs/gram_counts/'
LOGS = ROOT + 'logs/'

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
        m = filename.replace('RC_', '').replace('RS_v2_', '').replace('RS_', '')
        file_data = sc.textFile(CONTROL + filename + '/part-00000')
        file_data = file_data.filter(partial(remove_bots, bot_set=bots))
        
        if filename.startswith('RC_'): 
            cdata = file_data.filter(check_valid_comment)
            cdata = cdata.flatMap(partial(get_ngrams_comment, tokenizer=tokenizer, per_comment=per_comment))
            cdata = cdata.map(lambda n: (n, 1))
            data = cdata.reduceByKey(lambda n1, n2: n1 + n2)
        else: 
            pdata = file_data.filter(check_valid_post)
            pdata = pdata.flatMap(partial(get_ngrams_post, tokenizer=tokenizer, per_comment=per_comment))
            pdata = pdata.map(lambda n: (n, 1))
            data = pdata.reduceByKey(lambda n1, n2: n1 + n2)
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
    
def count_vocab_mainstream(line, tokenizer=None, vocab=set()): 
    '''
    Counts vocab words for mainstream reddit
    These are PER-COMMENT counts
    '''
    d = json.loads(line)
    if 'selftext' in d: 
        text = d['selftext'].lower()
    elif 'body' in d: 
        text = d['body'].lower()
    else: 
        text = ''
    sr = d['subreddit']
    toks = tokenizer.tokenize(text)
    unigrams = set(toks)
    bigrams = set()
    for s in ngrams(toks,n=2):        
        bigrams.add(' '.join(s))  
    overlap = (unigrams & vocab) | (bigrams & vocab)
    ret = []
    for w in overlap: 
        ret.append(((sr, w), 1))
    return ret
    
def count_lexical_innovations(): 
    '''
    get monthly counts of each lexical innovation in mainstream dataset
    '''
    mainstream_path = ROOT + 'data/mainstream/'
    schema = StructType([
      StructField('word', StringType(), True),
      StructField('count', IntegerType(), True),
      StructField('community', StringType(), True),
      StructField('month', StringType(), True)
      ])
    df = sqlContext.createDataFrame([],schema)
    bots = get_bot_set()
    tokenizer = BasicTokenizer(do_lower_case=True)
    vocab = set()
    with open(LOGS + 'lexical_innovations.txt', 'r') as infile: 
        for line in infile: 
            vocab.add(line.strip())
    for folder in sorted(os.listdir(mainstream_path)):
        if not folder.startswith('RC_'): continue 
        com_input = mainstream_path + folder + '/part-00000'
        month = folder.replace('RC_', '')
        sub_input = ''
        if os.path.exists(mainstream_path + 'RS_' + month): 
            sub_input = mainstream_path + 'RS_' + month + '/part-00000'
        elif os.path.exists(mainstream_path + 'RS_v2_' + month): 
            sub_input = mainstream_path + 'RS_v2_' + month + '/part-00000'
        assert sub_input != ''
        if not os.path.exists(com_input) and not os.path.exists(sub_input): continue

        if os.path.exists(sub_input): 
            pdata = sc.textFile(sub_input)
            pdata = pdata.filter(partial(remove_bots, bot_set=bots))
            pdata = pdata.flatMap(partial(count_vocab_mainstream, tokenizer=tokenizer, vocab=vocab))
            pdata = pdata.reduceByKey(lambda n1, n2: n1 + n2)
        else: 
            pdata = sc.emptyRDD()
        
        if os.path.exists(com_input): 
            cdata = sc.textFile(com_input)
            cdata = cdata.filter(partial(remove_bots, bot_set=bots))
            cdata = cdata.flatMap(partial(count_vocab_mainstream, tokenizer=tokenizer, vocab=vocab))
            cdata = cdata.reduceByKey(lambda n1, n2: n1 + n2)
        else: 
            cdata = sc.emptyRDD()
        
        data = cdata.union(pdata)
        data = data.reduceByKey(lambda n1, n2: n1 + n2) 
        
        data = data.map(lambda tup: Row(word=tup[0][1], count=tup[1], community=tup[0][0], month=month))
        data_df = sqlContext.createDataFrame(data, schema)
        df = df.union(data_df)
    df.write.mode('overwrite').parquet(LOGS + 'word_dest/mainstream_counts')
    
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
        
def get_sustained_periods(df): 
    srs = set(df['community'].to_list()) 
    words = set(df['word'].to_list()) 
    sustained_periods = defaultdict(dict) # {w : {sr : (start, end)}}
    df = df[df['month'] != 'None-None']
    for sr in tqdm(srs): 
        for w in words: 
            this_df = df[df['community'] == sr]
            this_df = this_df[this_df['word'] == w]
            if len(this_df) >= 3: 
                month_counts = this_df[['month', 'count']].set_index('month').to_dict()['count']
                month_range = sorted(month_counts.keys())
                start = None
                num_months = 0
                prev_month = None
                end = None
                for month in month_year_iter(min(month_range), max(month_range)): 
                    if month in month_counts: # increment months, assign start if needed
                        if start is None: 
                            start = month
                        num_months += 1
                    elif num_months < 3: # not a long enough sustained period, restart
                        start = None
                        num_months = 0
                    else: # end of earliest sustained period
                        end = prev_month 
                        break
                    prev_month = month
                if start and not end: 
                    end = max(month_range)
                if start and end: 
                    sustained_periods[w][sr] = (start, end)
    return sustained_periods
    
def mainstream_sustained_periods(): 
    '''
    Get words that have sustained presence in mainstream reddit.
    '''
    vocab = set()
    with open(LOGS + 'lexical_innovations.txt', 'r') as infile: 
        for line in infile: 
            vocab.add(line.strip())
            
    mainstream_df = sqlContext.read.parquet(LOGS + 'word_dest/mainstream_counts')
    mainstream_df = mainstream_df.filter(mainstream_df['count'] > 20) # 32665 rows
    mainstream_df = mainstream_df.toPandas()
    sustained_periods = get_sustained_periods(mainstream_df)
                    
    with open(LOGS + 'sustained_mainstream.json', 'w') as outfile: 
        json.dump(sustained_periods, outfile)
        
def manosphere_sustained_periods(): 
    with open(LOGS + 'sustained_mainstream.json', 'r') as infile: 
        sustained_periods = json.load(infile)
        
    vocab = set(sustained_periods.keys()) # 107 words
    manosphere_df = sqlContext.read.parquet(LOGS + 'gram_counts/combined_counts_set')
    manosphere_df = manosphere_df.filter(manosphere_df.word.isin(vocab))
    manosphere_df = manosphere_df.filter(manosphere_df['count'] > 20) # 4938 rows
    manosphere_df = manosphere_df.toPandas()
    sustained_periods = get_sustained_periods(manosphere_df)
                    
    with open(LOGS + 'sustained_manosphere.json', 'w') as outfile: 
        json.dump(sustained_periods, outfile)

def main(): 
#     count_control(per_comment=False)
#     count_sr(per_comment=False)
#     count_forum(per_comment=False)
    count_control()
#     count_sr()
#     count_forum()
    get_total_tokens()
#     count_lexical_innovations()
#     mainstream_sustained_periods()
#     manosphere_sustained_periods()
    sc.stop()

if __name__ == "__main__":
    main()
