import csv
import json
import os
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from helpers import get_sr_cats, valid_line

ROOT = '/mnt/data0/lucy/manosphere/'
POSTS = ROOT + 'data/submissions/'
COMMENTS = ROOT + 'data/comments/'
FORUMS = ROOT + 'data/cleaned_forums/'
LOGS = ROOT + 'logs/'

def get_users_per_subreddit(): 
    '''
    Get the set of users posting or commenting
    in each subreddit 
    '''
    categories = get_sr_cats()
    sr_users = defaultdict(set)
    for f in sorted(os.listdir(COMMENTS)):
        if f == 'bad_jsons': continue 
        month = f.replace('RC_', '')
        print(month)
        with open(COMMENTS + f + '/part-00000', 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                text = d['body']
                sr = d['subreddit'].lower()
                cat = categories[sr]
                if cat == 'Health' or cat == 'Criticism': continue
                if valid_line(text) and d['author'] != '[deleted]': 
                    sr_users[sr].add(d['author'])
                
        if os.path.exists(POSTS + 'RS_' + month + '/part-00000'): 
            post_path = POSTS + 'RS_' + month + '/part-00000'
        else: 
            post_path = POSTS + 'RS_v2_' + month + '/part-00000'
        with open(post_path, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                text = d['selftext']
                sr = d['subreddit'].lower()
                cat = categories[sr]
                if cat == 'Health' or cat == 'Criticism': continue
                if valid_line(text) and d['author'] != '[deleted]': 
                    sr_users[sr].add(d['author'])
    
    sr_users_list = defaultdict(list)
    for k in sr_users: 
        sr_users_list[k] = list(sr_users[k])
    with open(LOGS + 'users/sr_users.json', 'w') as outfile: 
        json.dump(sr_users_list, outfile)
        
def get_community_network(): 
    categories = get_sr_cats()
    with open(LOGS + 'users/sr_users.json', 'r') as infile: 
        sr_users = json.load(infile)
    overlap_values = {}
    for sr1 in sr_users: 
        for sr2 in sr_users: 
            if sr1 != sr2: 
                overlap = len(set(sr_users[sr1]) & set(sr_users[sr2]))
                overlap_values[tuple(sorted([sr1, sr2]))] = overlap 
    G = nx.Graph()
    for sr in sr_users: 
        G.add_node(sr)
    color_map = []
    colors = {
        'MRA': 'blue', 
        'TRP': 'red', 
        'PUA': 'orange', 
        'Incels': 'yellow',
        'FDS': 'purple', 
        'Femcels': 'gray', 
        'MGTOW': 'green'
        }
    for node in G:
        color_map.append(colors[categories[node]])
    for tup in overlap_values: 
        G.add_edge(tup[0], tup[1], weight=overlap_values[tup])
        
    nx.draw_spectral(G, node_color=color_map, with_labels=True)
    plt.savefig(LOGS + 'users/community_network.png', dpi=300, bbox_inches='tight')

def main(): 
    #get_users_per_subreddit()
    get_community_network()

if __name__ == '__main__':
    main()