"""
Find growth and decline words 
"""
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import col
from helpers import get_sr_cats
import math
from scipy.stats import spearmanr
from collections import Counter
import json

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

def get_time_series(word, df, um_totals, bm_totals, vocab_sizes): 
    '''
    Gets normalized and log transformed
    frequencies for each word. 
    Normalized means it is word count in month / total words in month
    For bigrams, it's bigram count in month / total bigrams in month
    Log transformed is log10. 
    
    - word: word to get time series for
    - df: dataframe containing word counts
    - um_totals: total number of unigrams per month
    - bm_totals: total number of bigrams per month
    - vocab_sizes: total number of unique unigrams and bigrams 
    '''
    # df filter for word 
    word_df = df.rdd.filter(lambda x: x[0] == word)
    # sum counts per month 
    word_counts = word_df.map(lambda x: (x[3], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    word_counts = Counter(word_counts)
    # check if word is unigram OR bigram 
    assert len(word.split(' ')) < 3
    totals = None
    uniq_count = None
    if len(word.split(' ')) == 1: 
        totals = um_totals
    elif len(word.split(' ')) == 2: 
        totals = bm_totals
    # divide month counts by total month count 
    min_month = 'z'
    max_month = '0'
    for m in word_counts: 
        if m != 'None-None' and m != '1970-01': 
            min_month = min(m, min_month)
            max_month = max(m, max_month)

    # trim zeros off start and end 
    # add smoothing to avoid zero counts
    ts = []
    start = False
    for m in month_year_iter(min_month, max_month): 
        if word_counts[m] > 0: 
            start = True
        if start: 
            prob = word_counts[m] / totals[m]
            prob += 1 / max(totals.values())
            ts.append(math.log(prob, 10))
    end = len(ts)
    for i in range(len(ts) - 1, -1, -1): 
        if ts[i] != 0: 
            end = i
            break
    return ts[:end+1]
    
def calculate_growth_words(ts): 
    # spearman correlation 
    # spearmanr(ts, range(len(ts)))
    pass
    
def calculate_decline_words(ts): 
    pass

def save_word_count_data(sqlContext): 
    '''
    This function is used to save a combined
    word count dataframe (Reddit, excluding Health and Criticism, and forums)
    and total word counts per month (bm_totals, um_totals)
    '''
    categories = get_sr_cats()
    df = load_gram_counts(categories, sqlContext)
    unigrams = df.rdd.filter(lambda x: len(x[0].split(' ')) == 1)
    bigrams = df.rdd.filter(lambda x: len(x[0].split(' ')) == 2)
    
    # {month : total word count}
    um_totals = unigrams.map(lambda x: (x[3], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    um_totals = Counter(um_totals)
    bm_totals = bigrams.map(lambda x: (x[3], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    bm_totals = Counter(bm_totals)
    
    # total unique bigrams and unigrams
    unigram_size = unigrams.toDF().select(col("word")).distinct().count()
    bigram_size = bigrams.toDF().select(col("word")).distinct().count()
    
    with open(WORD_COUNT_DIR + 'unique_counts', 'w') as outfile: 
        outfile.write('UNIGRAMS,' + str(unigram_size))
        outfile.write('BIGRAMS,' + str(bigram_size))
    
    df.write.mode('overwrite').parquet(WORD_COUNT_DIR + 'combined_counts')
    
    with open(WORD_COUNT_DIR + 'bigram_totals.json', 'w') as outfile: 
        json.dump(bm_totals, outfile)
        
    with open(WORD_COUNT_DIR + 'unigram_totals.json', 'w') as outfile: 
        json.dump(um_totals, outfile)

def get_word_count_data(sqlContext): 
    '''
    This loads the saved word count data that was produced and
    precalculated by save_word_count_data() above. 
    '''
    df = sqlContext.read.parquet(WORD_COUNT_DIR + 'combined_counts')
    
    with open(WORD_COUNT_DIR + 'bigram_totals.json', 'r') as infile: 
        bm_totals = json.load(infile)
        
    with open(WORD_COUNT_DIR + 'unigram_totals.json', 'r') as infile: 
        um_totals = json.load(infile)
        
    word_months = df.select(col("word"),col("month")).distinct().groupBy('month').count()
    
    vocab_sizes = {}
    with open(WORD_COUNT_DIR + 'unique_counts', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split(',')
            vocab_sizes[contents[0]] = int(contents[1])
        
    return df, um_totals, bm_totals, vocab_sizes

def main(): 
    '''
    Note, if you want to be able to see the
    printed output more clearly, you can write it to a file 
    '''
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    
    df, um_totals, bm_totals, vocab_sizes = get_word_count_data(sqlContext)
    
    words = ['incel', 'roastie', 'femoid', 'femcel', 'amog']
    for w in words: 
        print(w)
        ts = get_time_series(w, df, um_totals, bm_totals, vocab_sizes)
        print(ts)
        break
        #calculate_growth_words(ts)
        #calculate_decline_words(ts)

if __name__ == '__main__':
    main()
