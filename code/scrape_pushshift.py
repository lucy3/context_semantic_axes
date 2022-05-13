'''
This file downloads submissions and posts
from Pushshift website html, and
then checks that all months are covered. 
'''

from bs4 import BeautifulSoup
import wget
import os
from collections import Counter

# html of pushshift website
S_INPUT = '/mnt/data0/lucy/manosphere/data/submissions.txt'
C_INPUT = '/mnt/data0/lucy/manosphere/data/comments.txt'
# folders of downloaded files
OUT_S = '/mnt/data0/corpora/reddit/submissions/'
OUT_C = '/mnt/data0/corpora/reddit/comments/'
CLEANED_S = '/mnt/data0/lucy/manosphere/data/submissions/'

def get_submissions(): 
    links = set()
    with open(S_INPUT, 'r') as infile:
        contents = infile.read()
        soup = BeautifulSoup(contents, 'lxml')
        for a in soup.find_all('a', href=True):
            if a['href'].endswith('.xz') or a['href'].endswith('.zst') or a['href'].endswith('.bz2'):
                links.add(a['href'])
    for link in sorted(links):
        full_link = 'https://files.pushshift.io/reddit/submissions/' + link[2:]
        print(full_link)
        filename = wget.download(full_link, out=OUT_S)
        
def check_files(d): 
    '''
    Check that we have exactly one copy of each month,
    and if there are duplicates or missing ones, print it out. 
    '''
    months = Counter()
    num_files = 0
    for filename in os.listdir(d): 
        f = filename.replace('RS_', '').replace('RC_', '').replace('v2_', '').split('.')[0]
        months[f] += 1
        num_files += 1
    for f in months: 
        if months[f] != 1: 
            print("Too many files:", f)
    for m in range(1, 13): 
        for y in range(2005, 2020): 
            if len(str(m)) == 1: 
                d = str(y) + '-0' + str(m)
            else:
                d = str(y) + '-' + str(m)
            if d not in months: 
                print("Missing:", d)
                
def get_comments(): 
    links = set()
    with open(C_INPUT, 'r') as infile:
        contents = infile.read()
        soup = BeautifulSoup(contents, 'lxml')
        for a in soup.find_all('a', href=True):
            if a['href'].endswith('.xz') or a['href'].endswith('.zst') or a['href'].endswith('.bz2'):
                links.add(a['href'])
    for link in sorted(links):
        full_link = 'https://files.pushshift.io/reddit/comments/' + link[2:]
        if os.path.exists(OUT_C + link[2:]): 
            continue
        else: 
            print(full_link)
            filename = wget.download(full_link, out=OUT_C)

def main(): 
    get_submissions()
    check_files(OUT_S)
    print()
    get_comments()
    check_files(OUT_C)
    # check for filtered submissions
    check_files(CLEANED_S) 

if __name__ == '__main__':
    main()
