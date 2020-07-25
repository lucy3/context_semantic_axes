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

conf = SparkConf()
sc = SparkContext(conf=conf)

IN_S = '/mnt/data0/corpora/reddit/submissions/'
IN_C = '/mnt/data0/corpora/reddit/comments/'
UD = '/mnt/data0/corpora/urban_dictionary/UD2019/Oct19/all_definitions.dat'
ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
MANUAL_PEOPLE = '/mnt/data0/lucy/manosphere/data/manual_people.csv'

def get_manual_people(): 
    """
    get list of words, add plural forms
    """
    words = set()
    with open(MANUAL_PEOPLE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['community'].strip() != 'generic': 
                word_sing = row['word (singular)']
                if len(word_sing.split()) == 0: continue
                last_word = word_sing.split()[-1]
                other_words = word_sing.split()[:-1]
                if last_word == 'man': 
                    plural = ' '.join(other_words + ['men'])
                elif last_word == 'woman': 
                    plural = ' '.join(other_words + ['women'])
                else:
                    plural = word_sing + 's'
                words.add(word_sing.lower())
                words.add(plural.lower())
    return words

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
    and a same-size sample of the
    rest of Reddit 
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
        if os.path.isdir(out_d + filename): continue
        unpack_file(in_d, f)
        data = sc.textFile(in_d + filename)
        not_wanted = data.filter(get_dumb_lines).collect()
        data = data.filter(lambda line: not get_dumb_lines(line))
        rel_data = data.filter(lambda line: 'subreddit' in json.loads(line) and \
                    json.loads(line)['subreddit'].lower() in relevant_subs)
        rel_data.coalesce(1).saveAsTextFile(out_d + filename)
        if len(not_wanted) > 0: 
            with open(out_d + 'bad_jsons/' + filename + '.txt', 'w') as outfile: 
                for line in not_wanted:
                    outfile.write(line + '\n') 
        pack_file(in_d, f)

def find_urban_dictionary(): 
    """
    Takes a list of words and finds them in urban dictionary
    """
    people = get_manual_people() 
    data = sc.textFile(UD)
    data = data.filter(lambda line: line.strip().split('|')[0].lower() in people)
    filtered_data = data.collect()
    with open(LOGS + 'urban_dict.csv', 'w') as outfile: 
        for line in filtered_data: 
            outfile.write(line + '\n')

def main(): 
    #check_duplicate_months(IN_C, [('RC_2018-10.xz', 'RC_2018-10.zst')])
    #check_duplicate_months(IN_S, [('RS_2017-11.bz2', 'RS_2017-11.xz')])
    #check_duplicate_months(IN_S, [('RS_2017-07.bz2', 'RS_2017-07.xz')])
    #find_urban_dictionary()
    in_d = '/mnt/data0/corpora/reddit/comments/'
    out_d = '/mnt/data0/lucy/manosphere/data/comments/'
    extract_relevant_subreddits(in_d, out_d)
    #in_d = '/mnt/data0/corpora/reddit/submissions/'
    #out_d = '/mnt/data0/lucy/manosphere/data/submissions/'
    #extract_relevant_subreddits(in_d, out_d)
    sc.stop()

if __name__ == '__main__':
    main()
