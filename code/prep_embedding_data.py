from transformers import BasicTokenizer
import sys
import json 
from nltk import ngrams
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.sql import Row, SQLContext
from functools import partial
from nltk import tokenize
import sys
sys.path.insert(0, '/mnt/data0/lucy/manosphere/code')
from helpers import check_valid_comment, check_valid_post, remove_bots, get_bot_set, get_vocab
import os
import csv
from collections import defaultdict
import random

ROOT = '/mnt/data0/lucy/manosphere/'
LOGS = ROOT + 'logs/'
SUBS = ROOT + 'data/submissions/'
COMS = ROOT + 'data/comments/'
CONTROL = ROOT + 'data/reddit_control/'
FORUMS = ROOT + 'data/cleaned_forums/'
SUB_META = ROOT + 'data/subreddits.txt'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
sc.addPyFile('/mnt/data0/lucy/manosphere/code/helpers.py')

def get_subreddit_categories(): 
    categories = {}
    with open(SUB_META, 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in reader: 
            name = row['Subreddit'].strip().lower()
            if name.startswith('/r/'): name = name[3:]
            if name.startswith('r/'): name = name[2:]
            if name.endswith('/'): name = name[:-1]
            cat = row['Category after majority agreement']
            if cat != 'Criticism' and cat != 'Health': 
                categories[name] = cat
    return categories

def preprocess_text(text, idx, cat, tokenizer=None, vocab=set()): 
    '''
    idx is the comment/post's ID, and id_suffix is the sentence ID
    cat is category + year 
    '''
    sents = tokenize.sent_tokenize(text)
    id2sent = [] # (idx + id_suffix, sent)
    word2id = [] # ((term, cat), idx + id_suffix)
    id_suffix = 0
    for sent in sents: 
        tokens = tokenizer.tokenize(sent)
        if len(tokens) < 5 or len(tokens) > 150: 
            continue
        words_in_sent = set()
        for i in range(len(tokens)): 
            if i > 0 and tokens[i-1] + ' ' + tokens[i] in vocab: 
                # bigram:
                term = tokens[i-1] + ' ' + tokens[i]
                word2id.append(((term, cat), idx + str(id_suffix)))
                words_in_sent.add(term)
            if tokens[i] in vocab: 
                # unigram: 
                word2id.append(((tokens[i], cat), idx + str(id_suffix)))
                words_in_sent.add(tokens[i])
        if len(words_in_sent) > 0: 
            id2sent.append((idx + str(id_suffix), sent))
        id_suffix += 1
    return word2id, id2sent

def preprocess_comment(line, tokenizer=None, year='', vocab=set(), categories={}): 
    d = json.loads(line)
    sr = d['subreddit'].lower()
    if sr not in categories: 
        # health or criticism
        return ([], [])
    idx = d['id']
    cat = categories[sr] + '_' + year
    word2id, id2sent = preprocess_text(d['body'], idx, cat, tokenizer=tokenizer, vocab=vocab)
    return (word2id, id2sent)

def preprocess_comment_no_cat(line, tokenizer=None, year='', vocab=set()): 
    d = json.loads(line)
    idx = d['id']
    cat = 'CONTROL_' + year
    word2id, id2sent = preprocess_text(d['body'], idx, cat, tokenizer=tokenizer, vocab=vocab)
    return (word2id, id2sent)

def preprocess_post_no_cat(line, tokenizer=None, year='', vocab=set()): 
    d = json.loads(line)
    idx = d['id']
    cat = 'CONTROL_' + year
    word2id, id2sent = preprocess_text(d['selftext'], idx, cat, tokenizer=tokenizer, vocab=vocab)
    return (word2id, id2sent)

def preprocess_post(line, tokenizer=None, year='', vocab=set(), categories={}): 
    d = json.loads(line)
    sr = d['subreddit'].lower()
    idx = d['id']
    if sr not in categories: 
        # health or criticism
        return ([], [])
    cat = categories[sr] + '_' + year
    word2id, id2sent = preprocess_text(d['selftext'], idx, cat, tokenizer=tokenizer, vocab=vocab)
    return (word2id, id2sent)

def exact_sample(tup): 
    w = '_'.join(tup[0])
    occur = list(set(tup[1]))
    if len(occur) < 500: 
        return (w, occur)
    else: 
        return (w, random.sample(occur, 500))

def preprocess_dataset_reddit(): 
    '''
    Preprocesses Reddit manosphere data with sampling 
    We have up to 500 samples of each word in an 
    ideology (e.g. MRA/PUA) in a year (e.g. 2008). 
    '''
    vocab = get_vocab()
    tokenizer = BasicTokenizer(do_lower_case=True)
    bots = get_bot_set()
    
    categories = get_subreddit_categories()
    year_month = defaultdict(list) # {year : [months]}
    for filename in os.listdir(COMS): 
        if not filename.startswith('RC_'): continue
        y = filename.replace('RC_', '').split('-')[0]
        year_month[y].append(filename)

    for y in year_month: 
        all_word2id = sc.emptyRDD() # []
        all_id2sent = sc.emptyRDD() # [(id, sent)]
        for filename in year_month[y]: 
            m = filename.replace('RC_', '')
            cdata = sc.textFile(COMS + filename + '/part-00000')
            cdata = cdata.filter(check_valid_comment)
            cdata = cdata.filter(partial(remove_bots, bot_set=bots))
            cdata = cdata.map(partial(preprocess_comment, tokenizer=tokenizer, year=y, vocab=vocab, categories=categories))
            cword2id = cdata.flatMap(lambda x: x[0]).map(lambda tup: (tup[0], [tup[1]]))
            cword2id = cword2id.reduceByKey(lambda n1, n2: n1 + n2)
            cid2sent = cdata.flatMap(lambda x: x[1])

            if os.path.exists(SUBS + 'RS_' + m + '/part-00000'): 
                post_path = SUBS + 'RS_' + m + '/part-00000'
            else: 
                post_path = SUBS + 'RS_v2_' + m + '/part-00000'
            pdata = sc.textFile(post_path)
            pdata = pdata.filter(check_valid_post)
            pdata = pdata.filter(partial(remove_bots, bot_set=bots))
            pdata = pdata.map(partial(preprocess_post, tokenizer=tokenizer, year=y, vocab=vocab, categories=categories))
            pword2id = pdata.flatMap(lambda x: x[0]).map(lambda tup: (tup[0], [tup[1]]))
            pword2id = pword2id.reduceByKey(lambda n1, n2: n1 + n2)
            pid2sent = pdata.flatMap(lambda x: x[1])
            all_word2id = sc.union([all_word2id, cword2id, pword2id])
            all_word2id = all_word2id.reduceByKey(lambda n1, n2: n1 + n2)
            all_id2sent = sc.union([all_id2sent, cid2sent, pid2sent])

        all_word2id = all_word2id.map(exact_sample).collectAsMap()
        ids_to_keep = set()
        for k in all_word2id: 
            ids_to_keep.update(all_word2id[k])
        all_id2sent = all_id2sent.filter(lambda tup: tup[0] in ids_to_keep).collectAsMap()
        with open(LOGS + 'semantics_mano/reddit_' + y + '_word2id.json', 'w') as outfile: 
            json.dump(all_word2id, outfile)
        with open(LOGS + 'semantics_mano/reddit_' + y + '_id2sent.json', 'w') as outfile: 
            json.dump(all_id2sent, outfile)
                
    sc.stop()
    
def preprocess_dataset_control(): 
    '''
    Preprocesses Reddit control dataset with sampling.  
    We have up to 500 samples of each word in the control
    set in a year (e.g. 2008). 
    This function was modified off of preprocess_dataset_reddit()
    '''
    vocab = get_vocab()
    tokenizer = BasicTokenizer(do_lower_case=True)
    bots = get_bot_set()
    year_month = defaultdict(list) # {year : [months]}

    for filename in os.listdir(CONTROL): 
        if not filename.startswith('2'): continue
        y = filename.split('-')[0]
        year_month[y].append(filename)
    for y in year_month: 
        all_word2id = sc.emptyRDD() # []
        all_id2sent = sc.emptyRDD() # [(id, sent)]
        for filename in year_month[y]: 
            m = filename
            file_data = sc.textFile(CONTROL + filename + '/part-00000')
            file_data = file_data.filter(partial(remove_bots, bot_set=bots))
            cdata = file_data.filter(check_valid_comment)
            pdata = file_data.filter(check_valid_post)
            cdata = cdata.map(partial(preprocess_comment_no_cat, tokenizer=tokenizer, year=y, vocab=vocab))
            cword2id = cdata.flatMap(lambda x: x[0]).map(lambda tup: (tup[0], [tup[1]]))
            cword2id = cword2id.reduceByKey(lambda n1, n2: n1 + n2)
            cid2sent = cdata.flatMap(lambda x: x[1])
            pdata = pdata.map(partial(preprocess_post_no_cat, tokenizer=tokenizer, year=y, vocab=vocab))
            pword2id = pdata.flatMap(lambda x: x[0]).map(lambda tup: (tup[0], [tup[1]]))
            pword2id = pword2id.reduceByKey(lambda n1, n2: n1 + n2)
            pid2sent = pdata.flatMap(lambda x: x[1])
            all_word2id = sc.union([all_word2id, cword2id, pword2id])
            all_word2id = all_word2id.reduceByKey(lambda n1, n2: n1 + n2)
            all_id2sent = sc.union([all_id2sent, cid2sent, pid2sent])
        all_word2id = all_word2id.map(exact_sample).collectAsMap()
        ids_to_keep = set()
        for k in all_word2id: 
            ids_to_keep.update(all_word2id[k])
        all_id2sent = all_id2sent.filter(lambda tup: tup[0] in ids_to_keep).collectAsMap()
        with open(LOGS + 'semantics_mano/control_' + y + '_word2id.json', 'w') as outfile: 
            json.dump(all_word2id, outfile)
        with open(LOGS + 'semantics_mano/control_' + y + '_id2sent.json', 'w') as outfile: 
            json.dump(all_id2sent, outfile)
                
    sc.stop()
    
def preprocess_post(line, tokenizer=None, year='', vocab=set(), categories={}): 
    d = json.loads(line)
    sr = d['subreddit'].lower()
    idx = d['id']
    if sr not in categories: 
        # health or criticism
        return ([], [])
    cat = categories[sr] + '_' + year
    word2id, id2sent = preprocess_text(d['selftext'], idx, cat, tokenizer=tokenizer, vocab=vocab)
    return (word2id, id2sent)
    
def preprocess_forum_post(line, tokenizer=None, forum='', vocab=set(), categories={}): 
    d = json.loads(line)
    idx = str(d['id_post'])
    if d['date_post'] is None: 
        year = "None"
    else: 
        date_time_str = d["date_post"].split('-')
        year = date_time_str[0]
    cat = forum + '_' + year
    word2id, id2sent = preprocess_text(d['text_post'], idx, cat, tokenizer=tokenizer, vocab=vocab)
    return (word2id, id2sent)
    
def preprocess_dataset_forums(): 
    vocab = get_vocab()
    tokenizer = BasicTokenizer(do_lower_case=True)
    
    for filename in os.listdir(FORUMS):
        data = sc.textFile(FORUMS + filename)
        data = data.map(partial(preprocess_forum_post, tokenizer=tokenizer, forum=filename, vocab=vocab))
        word2id = data.flatMap(lambda x: x[0]).map(lambda tup: (tup[0], [tup[1]]))
        word2id = word2id.reduceByKey(lambda n1, n2: n1 + n2)
        id2sent = data.flatMap(lambda x: x[1])
        all_word2id = word2id.map(exact_sample).collectAsMap()
        ids_to_keep = set()
        for k in all_word2id: 
            ids_to_keep.update(all_word2id[k])
        all_id2sent = id2sent.filter(lambda tup: tup[0] in ids_to_keep).collectAsMap()
        with open(LOGS + 'semantics_mano/forum_' + filename + '_word2id.json', 'w') as outfile: 
            json.dump(all_word2id, outfile)
        with open(LOGS + 'semantics_mano/forum_' + filename + '_id2sent.json', 'w') as outfile: 
            json.dump(all_id2sent, outfile)

def main(): 
    #preprocess_dataset_reddit()
    #preprocess_dataset_control()
    preprocess_dataset_forums()

if __name__ == '__main__':
    main()