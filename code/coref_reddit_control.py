
'''
Goal: get pronouns that refer to
a word in our vocabulary.
'''

import spacy
import csv
import json
import os
import time
import sys
import neuralcoref
from collections import defaultdict


ROOT = '/global/scratch/users/dtadimeti/manosphere/'
LOGS = ROOT + 'logs/'
REDDIT_CONTROL = ROOT + 'data/reddit_control/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'
BOTS = LOGS + 'reddit_bots.txt'

def main():
    '''
    Output format: subreddit \t cluster1word1$cluster1word2 \t cluster2word1$cluster2word2$cluster2word3$cluster2word4 \n
    '''
    # load vocabulary
    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['keep'] == 'Y':
                # *CHANGED TO LOWERCASE*
                words.append(row['entity'].lower())

    # *CHANGED HOW I REMOVE 'SHE' AND 'HE' FROM VOCAB*
    pattern1 = "he"
    pattern2 = "she"

    if pattern1 in words:
        words.remove(pattern1)
    if pattern2 in words:
        words.remove(pattern2)


    # load coref
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)

    f = sys.argv[1]
    month = f.replace('RC_', '')


    outfile = open(LOGS + 'coref_reddit_control/' + month, 'w')

    error_outfile = open(LOGS + "reddit_control_errors", 'w')

    with open(REDDIT_CONTROL + month + '/part-00000', 'r') as infile:
        for line in infile:
            d = json.loads(line)
            sr = d['subreddit']

            if check_valid_post(line) and not check_valid_comment(line):
                text = d['selftext']

            if check_valid_comment(line) and not check_valid_post(line):
                text = d['body']

            if not check_valid_post(line) and not check_valid_comment(line):
                outfile.write(sr.lower())
                outfile.write("\n")
                continue


            try:
                # run the coref on text
                doc = nlp(text)

            except MemoryError:
                error_outfile.write(line + '\n')
                outfile.write(sr.lower())
                outfile.write("\n")
                continue

            else:
                outstring = ''
                for c in doc._.coref_clusters: # for coref cluster in doc
                    keep_cluster = False
                    for s in c.mentions: # for span in cluster
                        if s.text.lower() in words: # SCENARIO 2
                            keep_cluster = True
                            break
                        if s[0].dep_ in {'det','poss'}: # SCENARIO 1
                            new_s = s[1:]
                            if new_s.text.lower() in words:
                                keep_cluster = True
                                break
                    if keep_cluster:
                        curr_cluster = []
                        for s in c.mentions: # for span in cluster
                            entity = s.text.lower()
                            entity = entity.replace("\n", "")
                            curr_cluster.append(entity)
                        outstring += "$".join(curr_cluster) + "\t"

                outfile.write(sr.lower() + "\t" + outstring)
                outfile.write("\n")


    outfile.close()

def check_valid_comment(line):
    '''
    For Reddit comments
    '''
    comment = json.loads(line)
    if 'body' not in comment: return False
    text = comment['body']
    if len(text) > 1000000: return False
    if text == "" or text == "[deleted]" or text == "[removed]": return False

    # read in bots from reddit_bots.txt, create list
    bots = []
    with open(BOTS, 'r') as infile:
        for line in infile:
            bots.append(line)

    author = comment['author']
    if author in bots: return False

    return True

def check_valid_post(line):
    '''
    For Reddit posts
    '''
    post = json.loads(line)
    if 'selftext' not in post: return False
    text = post['selftext']
    if len(text) > 1000000: return False
    if text == "" or text == "[deleted]" or text == "[removed]": return False

    # read in bots from reddit_bots.txt, create list
    bots = []
    with open(BOTS, 'r') as infile:
        for line in infile:
            bots.append(line)
    author = post['author']
    if author in bots: return False

    return True


if __name__ == '__main__':
    main()

