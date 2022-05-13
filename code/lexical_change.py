"""
Find growth and decline words 
"""
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, split, sum
from helpers import get_sr_cats
import math
from scipy.stats import spearmanr
from collections import Counter, defaultdict
from functools import partial
import json
import numpy as np
import csv

ROOT = '/mnt/data0/lucy/manosphere/'
#ROOT = '/global/scratch/lucy3_li/manosphere/'
LOGS = ROOT + 'logs/'
COMMENTS = ROOT + 'data/comments/'
POSTS = ROOT + 'data/submissions/'
FORUMS = ROOT + 'data/cleaned_forums/'
WORD_COUNT_DIR = LOGS + 'gram_counts/'
TIME_SERIES_DIR = LOGS + 'time_series/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'

def load_gram_counts(categories, sqlContext): 
    reddit_df = sqlContext.read.parquet(WORD_COUNT_DIR + 'subreddit_counts_set')
    leave_out = []
    for sr in categories: 
        if categories[sr] == 'Health' or categories[sr] == 'Criticism': 
            leave_out.append(sr)
    reddit_df = reddit_df.filter(~reddit_df.community.isin(leave_out))
    
    forum_df = sqlContext.read.parquet(WORD_COUNT_DIR + 'forum_counts_set')
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
    Gets normalized frequencies for each word. 
    Normalized means it is word count in month / total words in month
    For bigrams, it's bigram count in month / total bigrams in month
    
    - word: word to get time series for
    - df: dataframe containing word counts
    - um_totals: total number of unigrams per month
    - bm_totals: total number of bigrams per month
    '''
    # df filter for word 
    word_rdd = df.rdd.filter(lambda x: x[0] == word)
    # sum counts per month 
    word_counts = word_rdd.map(lambda x: (x[3], x[1])).reduceByKey(lambda x,y: x + y).collectAsMap()
    word_counts = Counter(word_counts)
    # check if word is unigram OR bigram 
    assert len(word.split(' ')) < 3
    totals = None
    if len(word.split(' ')) == 1: 
        totals = um_totals
    elif len(word.split(' ')) == 2: 
        totals = bm_totals
    # divide month counts by total month count 
    min_month = '2005-11'
    max_month = '2019-12'

    ts = []
    
    for m in month_year_iter(min_month, max_month): 
        if m not in totals: 
            ts.append(0)
        else: 
            prob = word_counts[m] / totals[m]
            ts.append(prob)
    return ts

def save_word_count_data(sqlContext, dataset): 
    '''
    This function is used to save a combined
    word count dataframe (Reddit, excluding Health and Criticism, and forums)
    and total word counts per month (bm_totals, um_totals)
    '''
    if dataset == 'manosphere': 
        parquet_file = WORD_COUNT_DIR + 'combined_counts_set'
        bigram_file = WORD_COUNT_DIR + 'bigram_totals_set.json'
        unigram_file = WORD_COUNT_DIR + 'unigram_totals_set.json'
        unique_file = WORD_COUNT_DIR + 'unique_counts_set'
        
        categories = get_sr_cats()
        df = load_gram_counts(categories, sqlContext)
        df.write.mode('overwrite').parquet(parquet_file)
        
    elif dataset == 'control': 
        parquet_file = WORD_COUNT_DIR + 'control_counts_set'
        bigram_file = WORD_COUNT_DIR + 'bigram_totals_control_set.json'
        unigram_file = WORD_COUNT_DIR + 'unigram_totals_control_set.json'
        unique_file = WORD_COUNT_DIR + 'unique_counts_control_set'
        
        df = sqlContext.read.parquet(parquet_file)
    
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
    
    with open(unique_file, 'w') as outfile: 
        outfile.write('UNIGRAMS,' + str(unigram_size) + '\n')
        outfile.write('BIGRAMS,' + str(bigram_size))
    
    with open(bigram_file, 'w') as outfile: 
        json.dump(bm_totals, outfile)
        
    with open(unigram_file, 'w') as outfile: 
        json.dump(um_totals, outfile)

def get_word_count_data(sqlContext, dataset): 
    '''
    This loads the saved word count data that was produced and
    precalculated by save_word_count_data() above. 
    '''
    if dataset == 'manosphere': 
        parquet_file = WORD_COUNT_DIR + 'combined_counts_set'
        bigram_file = WORD_COUNT_DIR + 'bigram_totals_set.json'
        unigram_file = WORD_COUNT_DIR + 'unigram_totals_set.json'
        unique_file = WORD_COUNT_DIR + 'unique_counts_set'
    elif dataset == 'control': 
        parquet_file = WORD_COUNT_DIR + 'control_counts_set'
        bigram_file = WORD_COUNT_DIR + 'bigram_totals_control_set.json'
        unigram_file = WORD_COUNT_DIR + 'unigram_totals_control_set.json'
        unique_file = WORD_COUNT_DIR + 'unique_counts_control_set'
        
    df = sqlContext.read.parquet(parquet_file)
    with open(bigram_file, 'r') as infile: 
        bm_totals = json.load(infile)
    with open(unigram_file, 'r') as infile: 
        um_totals = json.load(infile)

    word_months = df.select(col("word"),col("month")).distinct().groupBy('month').count()
        
    return df, um_totals, bm_totals

def get_multiple_time_series(dataset, sqlContext): 
    '''
    @input: 
    - dataset: str that can be "manosphere" or "control" 
    '''
    
    df, um_totals, bm_totals = get_word_count_data(sqlContext, dataset)

    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                words.append(row['entity'])
    
    # filter the count dataframe just to the words we care about
    word_df = df.filter(df.word.isin(words))

    matrix = []
    with open(TIME_SERIES_DIR + 'vocab_' + dataset + '_set.txt', 'w') as outfile: 
        for w in words: 
            outfile.write(w + '\n')
            ts = get_time_series(w, word_df, um_totals, bm_totals)
            matrix.append(ts)
    matrix = np.array(matrix)
    np.save(TIME_SERIES_DIR + 'time_series_' + dataset + '_set.npy', matrix)
            
def smooth_time_series(dataset): 
    matrix = np.load(TIME_SERIES_DIR + 'time_series_' + dataset + '_set.npy')
    kernel_size = 3 # make sure to modify the code below if you change this
    kernel = np.ones(kernel_size) / kernel_size
    for i in range(matrix.shape[0]): 
        smoothed = np.convolve(matrix[i], kernel, mode='valid')
        smoothed = np.concatenate(([matrix[i][0]/2 + matrix[i][1]/2], smoothed, [matrix[i][-1]/2 + matrix[i][-2]/2]))
        assert smoothed.shape == matrix[i].shape
        matrix[i] = smoothed
    np.save(TIME_SERIES_DIR + 'time_series_' + dataset + '_smoothed_set.npy', matrix) 
    
def time_series_prep_and_run(): 
    '''
    Run this to get time series for each word
    that can later be clustered using ksc. 
    '''
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    
    save_word_count_data(sqlContext, 'manosphere')
    save_word_count_data(sqlContext, 'control')
    
    get_multiple_time_series('manosphere', sqlContext)
    smooth_time_series('manosphere')
    
    sc.stop()
    
def map_to_category(x, categories={}): 
    if x.community in categories: 
        cat = categories[x.community]
    else: 
        assert x.community.startswith('FORUM')
        cat = x.community
    return x.word,cat,x.year,x.summed_count
    
def calc_frequency_per_cat_year(): 
    '''
    This is used to accompany visualization of word embeddings
    over time in reddit and forum datasets. 
    
    @input: combined_counts_set for reddit + forum
    @output: nested dictionary of { cat_year: { term : count } }
    '''
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    
    parquet_file = WORD_COUNT_DIR + 'combined_counts_set'
    # word, count, community, month
    df = sqlContext.read.parquet(parquet_file)
    
    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                words.append(row['entity'])
                
    # filter for words we care about
    df = df.filter(df.word.isin(words))
    df = df.withColumn("year", split(df.month, '-').getItem(0)).drop('month')
    
    df = df.groupby(['word', 'community', 'year']).agg(sum("count").alias("summed_count"))
    categories = get_sr_cats()
    df = df.rdd.map(partial(map_to_category, categories=categories)).toDF(['word', 'category', 'year', 'summed_count'])
    df = df.collect()
    
    count_dict = defaultdict(dict) # { cat_year { term : count } }
    for row in df: 
        count_dict[row.category + '_' + str(row.year)][row.word] = row.summed_count
    with open(WORD_COUNT_DIR + 'combined_catyear_word_count.json', 'w') as outfile: 
        json.dump(count_dict, outfile)
    
    sc.stop()
    
def main(): 
    # time_series_prep_and_run()
    calc_frequency_per_cat_year()

if __name__ == '__main__':
    main()
