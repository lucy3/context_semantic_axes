"""
Find growth and decline words 
"""
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from helpers import get_sr_cats
import math
from scipy.stats import spearmanr
from collections import Counter

ROOT = '/mnt/data0/lucy/manosphere/'
#ROOT = '/global/scratch/lucy3_li/manosphere/'
LOGS = ROOT + 'logs/'
COMMENTS = ROOT + 'data/comments/'
POSTS = ROOT + 'data/submissions/'
FORUMS = ROOT + 'data/cleaned_forums/'
WORD_COUNT_DIR = LOGS + 'gram_counts/'

def load_gram_counts(categories, sqlContext): 
    reddit_df = sqlContext.read.parquet(WORD_COUNT_DIR + 'subreddit_counts')
    leave_out = []
    for sr in categories: 
        if categories[sr] == 'Health' or categories[sr] == 'Criticism': 
            leave_out.append(sr)
    reddit_df = reddit_df.filter(~reddit_df.community.isin(leave_out))
    
    forum_df = sqlContext.read.parquet(WORD_COUNT_DIR + 'forum_counts')
    df = forum_df.union(reddit_df)
    return df

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

def get_time_series(word, df, um_totals, bm_totals): 
    '''
    Gets normalized and log transformed
    frequencies for each word 
    
    Normalized means it is word count in month / total words in month
    For bigrams, it's bigram count in month / total bigrams in month
    Log transformed is log10. 
    '''
    # df filter for word 
    word_df = df.rdd.filter(lambda x: x[0] == word)
    # sum counts per month 
    word_counts = word_df.map(lambda x: (x[3], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    word_counts = Counter(word_counts)
    # check if word is unigram OR bigram 
    assert word.split(' ') < 3
    totals = None
    if word.split(' ') == 1: 
        totals = um_totals
    elif word.split(' ') == 2: 
        totals = bm_totals
    # divide month counts by total month count 
    min_month = 'z'
    max_month = '0'
    for m in word_counts: 
        if month != 'None-None' and month != '1970-01': 
            min_month = min(m, min_month)
            max_month = max(m, max_month)

    # trim zeros off start and end 
    ts = []
    start = False
    for m in month_year_iter(min_month, max_month): 
        print(m, word_counts[m])
        if word_counts[m] > 0: 
            start = True
        if start: 
            ts.append(math.log(word_counts[m] / totals[m], 10))
    end = len(ts)
    for i in range(len(ts) - 1, -1, -1): 
        if ts[i] != 0: 
            end = i
            break
    return ts[:end+1]
    
def calculate_growth_words(ts): 
    # spearman correlation 
    spearmanr(ts, range(len(ts)))
    
def calculate_decline_words(ts): 
    pass

def main(): 
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    
    categories = get_sr_cats()
    df = load_gram_counts(categories, sqlContext)
    unigrams = df.rdd.filter(lambda x: len(x[0].split(' ')) == 1)
    bigrams = df.rdd.filter(lambda x: len(x[0].split(' ')) == 2)
    
    # {month : total word count}
    um_totals = unigrams.map(lambda x: (x[3], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    um_totals = Counter(um_totals)
    bm_totals = bigrams.map(lambda x: (x[3], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    bm_totals = Counter(bm_totals)
    
    words = ['incel', 'roastie', 'femoid', 'femcel', 'amog']
    for w in words: 
        ts = get_time_series(w, df, um_totals, bm_totals)
        print(ts)
        break
        #calculate_growth_words(ts)
        #calculate_decline_words(ts)

if __name__ == '__main__':
    main()
