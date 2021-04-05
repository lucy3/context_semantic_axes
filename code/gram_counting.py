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
WORD_COUNT_DIR = ROOT + 'logs/gram_counts/'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def check_valid_comment(line): 
    comment = json.loads(line)
    return 'body' in comment and comment['body'].strip() != '[deleted]' \
            and comment['body'].strip() != '[removed]'

def get_n_gramlist(nngramlist, toks, sr, n=2):   
    # stack overflow said this was fastest
    for s in ngrams(toks,n=n):        
        nngramlist.append((sr, ' '.join(s)))                
    return nngramlist

def get_ngrams_comment(line, tokenizer=None): 
    d = json.loads(line)
    sr = d['subreddit'].lower()
    toks = tokenizer.tokenize(d['body'])
    all_grams = [(sr, i) for i in toks]
    all_grams = get_n_gramlist(all_grams, toks, sr, 2)
    all_grams = get_n_gramlist(all_grams, toks, sr, 3)
    return all_grams

def get_ngrams_post(line, tokenizer=None): 
    d = json.loads(line)
    all_grams = set()
    sr = d['subreddit'].lower()
    toks = tokenizer.tokenize(d['selftext'])
    all_grams = [(sr, i) for i in toks]
    all_grams = get_n_gramlist(all_grams, toks, sr, 2)
    all_grams = get_n_gramlist(all_grams, toks, sr, 3)
    return all_grams

def count_sr(): 
    tokenizer = BasicTokenizer(do_lower_case=True)
    schema = StructType([
      StructField('word', StringType(), True),
      StructField('count', IntegerType(), True),
      StructField('community', StringType(), True),
      StructField('month', StringType(), True)
      ])
    df = sqlContext.createDataFrame([],schema)
    for filename in ['RC_2013-11', 'RC_2005-12', 'RC_2015-10', 'RC_2019-06']: 
    #for filename in os.listdir(COMS): 
        if filename == 'bad_jsons': continue
        m = filename.replace('RC_', '')
        data = sc.textFile(COMS + filename + '/part-00000')
        data = data.filter(check_valid_comment)
        data = data.flatMap(partial(get_ngrams_comment, tokenizer=tokenizer))
        data = data.map(lambda n: (n, 1))
        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        data = data.map(lambda tup: Row(word=tup[0][1], count=tup[1], community=tup[0][0], month=m))
        data_df = sqlContext.createDataFrame(data, schema)
        df = df.union(data_df)
        
        if os.path.exists(SUBS + 'RS_' + m + '/part-00000'): 
            post_path = SUBS + 'RS_' + m + '/part-00000'
        else: 
            post_path = SUBS + 'RS_v2_' + m + '/part-00000'
        data = sc.textFile(post_path)
        data = data.flatMap(partial(get_ngrams_post, tokenizer=tokenizer))
        data = data.map(lambda n: (n, 1))
        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        data = data.map(lambda tup: Row(word=tup[0][1], count=tup[1], community=tup[0][0], month=m))
        data_df = sqlContext.createDataFrame(data, schema)
        df = df.union(data_df)
    outpath = WORD_COUNT_DIR + 'subreddit_counts'
    df.write.mode('overwrite').parquet(outpath)    
    
def count_forum(): 
    '''
    Need to group counts by month
    '''
    pass

def main(): 
    count_sr()
    count_forum()
    sc.stop()

if __name__ == "__main__":
    main()