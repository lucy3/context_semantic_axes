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

def get_num_forum_comments():
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
    with open(LOGS + 'forum_count.json', 'w') as outfile: 
        json.dump(forum_month, outfile)

def remove_quotes_and_duplicates(): 
    # incels
    '''
    processed_posts = SqliteDict(FORUMS + 'incels.sqlite', tablename="processed_posts")
    IDs2authors = defaultdict(set) # list just in case IDs are not unique
    for key, posts in tqdm.tqdm(processed_posts.items()): 
        for post in posts:
            IDs2authors[(key, post["id_post"])].add(post['author'])
            if post["id_post"] == 1266588: 
                print(post)
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
                        text = text[:start_id] + text[end_id+1:]
                        post["text_post"] = text
            if (key, post["id_post"]) not in already_seen: 
                # remove duplicates
                d_string = json.dumps(post)
                outfile.write(d_string + '\n')
            already_seen.add((key, post["id_post"]))
    outfile.close()
    '''
    
    # rooshv
    processed_posts = SqliteDict(FORUMS + 'rooshv.sqlite', tablename="processed_posts")
    for key, posts in processed_posts.items(): 
        for post in posts:
            print(post["id_post"], post["text_post"], post["id_post_interaction"])
    
    
def remove_duplicates(): 
    '''
    TODO: do this for not incels or rooshv
    '''
    pass
        
def main(): 
    #get_num_forum_comments()
    remove_quotes_and_duplicates()
    

if __name__ == '__main__':
    main()
