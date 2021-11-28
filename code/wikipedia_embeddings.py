"""
For getting BERT embeddings of key words from wikipedia
"""
import requests
import json
from tqdm import tqdm
import wikitextparser as wtp
from transformers import BasicTokenizer

ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'

def get_adj(): 
    '''
    Read in adjectives
    '''
    axes_vocab = set()
    with open(LOGS + 'semantics_val/wordnet_axes.txt', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            if len(contents) < 3: continue # no antonyms
            synset = contents[0]
            axis1 = contents[1].split(',')
            axis2 = contents[2].split(',')
            axes_vocab.update(axis1)
            axes_vocab.update(axis2)
    return axes_vocab
    
def get_occupations(): 
    '''
    Read in occupation words
    '''
    pass

def get_titles(vocab, vocab_name): 
    '''
    Get titles of pages that contain the keywords we
    are interested in, and download the wikitext of those pages.
    
    This is deprecated/unused because the wikipedia pages that get retrieved
    are usually biased towards proper entities that contain the keyword 
    rather than general uses of the keyword. 
    '''
    titles = set()
    for w in tqdm(vocab): 
        response = requests.get('https://en.wikipedia.org/w/api.php?action=query&list=search&srwhat=text&srsearch=' + w + '&format=json')
        if not response.ok: 
            print("Problem with", w)
        res = json.loads(response.text)['query']['search']
        for r in res: 
            titles.add(r['title'])
    with open(LOGS + 'wikipedia/' + vocab_name + '_titles.txt', 'w') as outfile: 
        for title in titles: 
            outfile.write(title + '\n')
            
def download_pages(title_file): 
    """
    This is deprecated/unused because the wikipedia pages that get retrieved
    are usually biased towards proper entities that contain the keyword 
    rather than general uses of the keyword. 
    """
    titles = []
    with open(title_file, 'r') as infile: 
        for line in infile:
            titles.append(line.strip())
    page_num = 0
    for wiki_title in tqdm(titles): 
        page_num += 1
        wiki_title = wiki_title.replace('&', '%26').replace('+', '%2B')
        response = requests.get('https://en.wikipedia.org/w/api.php?action=parse&page=' + wiki_title + '&prop=wikitext&formatversion=2&format=json')
        if not response.ok: 
            print("Problem with", wiki_title)
            continue
        wikitext = json.loads(response.text)
        if 'parse' not in wikitext or 'wikitext' not in wikitext['parse']: 
            print("Problem with dictionary", wiki_title)
            continue
        try: 
            wikitext = wikitext['parse']['wikitext']
            wikitext = wtp.remove_markup(wikitext)
            with open(LOGS + 'wikipedia/pages/' + str(page_num), 'w') as outfile: 
                outfile.write(wiki_title + '\n')
                outfile.write(wikitext)
        except: 
            print("Something went wrong with", wiki_title)
            
def sample_from_wikipedia(vocab, vocab_name): 
    '''
    
    '''
    tokenizer = BasicTokenizer(do_lower_case=True)

def main(): 
    vocab = get_adj()
    sample_from_wikipedia(vocab, 'adj')

if __name__ == '__main__':
    main()
