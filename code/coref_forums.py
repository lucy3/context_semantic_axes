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


ROOT = '/global/scratch/users/lucy3_li/manosphere/'
LOGS = ROOT + 'logs/'
FORUMS = ROOT + 'data/cleaned_forums/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'

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
    forum_name = sys.argv[1]
    outfile = open(LOGS + 'coref_forums/' + forum_name, 'w')
    writer = csv.writer(outfile, delimiter='\t')

    error_outfile = open(LOGS + "forum_errors", 'w')

    with open(FORUMS + forum_name,'r') as infile:
        for line in infile:
            d = json.loads(line)

            # skip the really long post (id_post: 2380578)
            if forum_name == "incels" and d["id_post"] == 2380578: continue

            text = d['text_post']
            date_post = d['date_post']
            if d['date_post'] is None: continue
                
            date = date_post[0:10]

            if not check_valid_forum(line):
                writer.writerow([date])
                continue

            try:
                # run the coref on text
                doc = nlp(text)

            except MemoryError:
                error_outfile.write(line + '\n')
                writer.writerow([date])
                continue

            else:
                outstring = [date]
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
                        outstring.append("$".join(curr_cluster))

                writer.writerow(outstring)


    outfile.close()


def check_valid_forum(line):
    '''
    For forums posts
    '''
    post = json.loads(line)
    if 'text_post' not in post: return False
    text = post['text_post']
    if len(text) > 1000000: return False
    if text == "" or text == "[deleted]" or text == "[removed]": return False

    return True



if __name__ == '__main__':
    main()

