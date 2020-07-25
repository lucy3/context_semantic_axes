from pyspark import SparkConf, SparkContext
from transformers import BertTokenizer, BasicTokenizer
from collections import defaultdict, Counter
import json
import os

conf = SparkConf()
sc = SparkContext(conf=conf)

ROOT = '/mnt/data0/lucy/manosphere/'
LOGS = ROOT + 'logs/'
SUBS = ROOT + 'data/submissions/'
COMS = ROOT + 'data/comments/'

def get_num_tokens(): 
    '''
    Get total number of tokens in data
    '''
    tokenizer = BasicTokenizer(do_lower_case=True)
    total = 0
    for filename in os.listdir(COMS): 
        if filename == 'bad_jsons': continue
        for part in os.listdir(COMS + filename):
            if not part.startswith('part-'): continue
            data = sc.textFile(COMS + filename + '/' + part)
            data = data.filter(check_valid_comment)
            data = data.map(lambda line: len(tokenizer.tokenize(json.loads(line)['body'])))
            total += sum(data.collect())
    print("TOTAL:", total)
    
def check_valid_comment(line): 
    comment = json.loads(line)
    return 'body' in comment and comment['body'].strip() != '[deleted]' \
            and comment['body'].strip() != '[removed]'
    
def get_num_comments(): 
    '''
    Get number of comments per subreddit per month
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
        
def main(): 
    #get_num_comments()
    get_num_tokens()

if __name__ == '__main__':
    main()
            