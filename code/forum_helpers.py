from sqlitedict import SqliteDict
import os
import datetime
from collections import defaultdict, Counter
import json
import tqdm

ROOT = '/mnt/data0/lucy/manosphere/'
FORUMS = ROOT + 'data/forums/'
CLEAN_FORUMS = ROOT + 'data/cleaned_forums/'
LOGS = ROOT + 'logs/'

def get_num_forum_comments_old():
    forum_month = defaultdict(Counter)
    for filename in os.listdir(FORUMS): 
        if not filename.endswith('.sqlite'): continue
        forum_name = filename.replace('.sqlite', '')
        print(forum_name)
        processed_posts = SqliteDict(FORUMS + filename, tablename="processed_posts")
        for key, posts in processed_posts.items(): 
            for post in posts:
                if post['date_post'] is None: 
                    year = "None"
                    month = "None"
                else: 
                    date_time_str = post["date_post"].split('-')
                    year = date_time_str[0]
                    month = date_time_str[1]
                forum_month[year + '-' + month][forum_name] += 1
    with open(LOGS + 'old_forum_count.json', 'w') as outfile: 
        json.dump(forum_month, outfile)
        
def get_num_forum_comments(): 
    forum_month = defaultdict(Counter)
    for forum_name in os.listdir(CLEAN_FORUMS): 
        print(forum_name)
        i = 0
        with open(CLEAN_FORUMS + forum_name, 'r') as infile: 
            for line in infile: 
                post = json.loads(line.strip())
                if post['date_post'] is None: 
                    year = "None"
                    month = "None"
                else: 
                    date_time_str = post["date_post"].split('-')
                    year = date_time_str[0]
                    month = date_time_str[1]
                forum_month[year + '-' + month][forum_name] += 1
    with open(LOGS + 'forum_count.json', 'w') as outfile: 
        json.dump(forum_month, outfile)  

def remove_quotes_and_duplicates(): 
    # incels
    processed_posts = SqliteDict(FORUMS + 'incels.sqlite', tablename="processed_posts")
    IDs2authors = defaultdict(set) # list just in case IDs are not unique
    for key, posts in tqdm.tqdm(processed_posts.items()): 
        for post in posts:
            IDs2authors[(key, post["id_post"])].add(post['author'])
    start_quote = ' said:'
    end_quote = 'Click to expand...'

    already_seen = set()
    outfile = open(CLEAN_FORUMS + 'incels', 'w')
    for key, posts in processed_posts.items(): 
        for post in posts:
            if len(post["id_post_interaction"]) != 0: 
                text = post["text_post"]
                for quoted_id in post["id_post_interaction"]: 
                    if len(IDs2authors[(key, quoted_id)]) > 0: 
                        # remove quotes
                        quote_author = list(IDs2authors[(key, quoted_id)])[0]
                        start_id = text.find(quote_author + start_quote)
                        end_id = text.find(end_quote) + len(end_quote)
                        if start_id != -1 and end_id != -1: 
                            text = text[:start_id] + ' ' + text[end_id+1:]
                            post["text_post"] = text
            if (key, post["id_post"]) not in already_seen: 
                # remove duplicates
                d_string = json.dumps(post)
                outfile.write(d_string + '\n')
            already_seen.add((key, post["id_post"]))
    outfile.close()  
    
    # rooshv
    processed_posts = SqliteDict(FORUMS + 'rooshv.sqlite', tablename="processed_posts")
    IDs2authors = defaultdict(set) # list just in case IDs are not unique
    # IDs are actually unique, was checked later
    IDs2post = defaultdict(str)
    for key, posts in tqdm.tqdm(processed_posts.items()): 
        for post in posts:
            IDs2authors[(key, post["id_post"])].add(post['author'])
            IDs2post[(key, post["id_post"])] = post["text_post"]
            
    start_quote = " Wrote: "
    already_seen = set()
    outfile = open(CLEAN_FORUMS + 'rooshv', 'w')
    for key, posts in processed_posts.items(): 
        for post in posts:
            if (key, post["id_post"]) in already_seen: continue
            if len(post["id_post_interaction"]) != 0 and "Wrote" in post["text_post"]:
                text = post["text_post"]
                for quoted_id in post["id_post_interaction"]:
                    if len(IDs2authors[(key, quoted_id)]) > 0: 
                        assert len(IDs2authors[(key, quoted_id)]) == 1
                        # remove quotes
                        quote_author = list(IDs2authors[(key, quoted_id)])[0]
                        if quote_author is None: continue
                        start_id = text.find(quote_author + start_quote)
                        quote_start = start_id + len(quote_author + start_quote)
                        if start_id == -1: continue
                        datetime = text[start_id-21:start_id]
                        if datetime.startswith('('): 
                            start_id = start_id-21
                        the_rest = text[quote_start:]
                        quoted_post = IDs2post[(key, quoted_id)]
                        # get the longest quote
                        end_id = 0
                        for i in range(len(the_rest)): 
                            excerpt = the_rest[:i]
                            if excerpt not in quoted_post: 
                                break
                            end_id = i
                        # some matches are too small to be sure
                        if end_id < 5: continue
                        text = text[:start_id] + ' ' + text[quote_start + end_id:]
                        post["text_post"] = text
            d_string = json.dumps(post)
            outfile.write(d_string + '\n')
            already_seen.add((key, post["id_post"]))
    outfile.close()                    
    
def remove_duplicates(): 
    '''
    Make sure there are no duplicate posts in dataset
    '''
    for filename in os.listdir(FORUMS): 
        if not filename.endswith('.sqlite'): continue
        forum_name = filename.replace('.sqlite', '')
        if forum_name == 'incels' or forum_name == 'rooshv': continue
        print(forum_name)
        already_seen = set()
        processed_posts = SqliteDict(FORUMS + filename, tablename="processed_posts")
        outfile = open(CLEAN_FORUMS + forum_name, 'w')
        for key, posts in processed_posts.items(): 
            for post in posts:
                if (key, post["id_post"]) in already_seen: continue
                d_string = json.dumps(post)
                outfile.write(d_string + '\n')
                already_seen.add((key, post["id_post"]))
        outfile.close()

def main(): 
    get_num_forum_comments()
    #remove_quotes_and_duplicates()
    #remove_duplicates()
    

if __name__ == '__main__':
    main()
