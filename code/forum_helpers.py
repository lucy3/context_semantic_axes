from sqlitedict import SqliteDict
import os
import datetime
from collections import defaultdict, Counter
import json

ROOT = '/mnt/data0/lucy/manosphere/'
FORUMS = ROOT + 'data/forums/'
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
                
def main(): 
    get_num_forum_comments()

if __name__ == '__main__':
    main()
