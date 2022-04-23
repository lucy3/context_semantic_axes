'''
This script does the following: 
- prep_datasets() calls functions for gathering
  Wikipedia pages that contain occupations 
- retrieve_wordnet_axes() creates WordNet axes
- prep_person_exp() creates input for seeing if the embedding for
'person' changes across contexts 
'''
from collections import defaultdict
import json
from nltk.corpus import wordnet as wn
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import spatial
import math
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
import re
from nltk import tokenize

ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
GLOVE = DATA + 'glove/'
LOGS = ROOT + 'logs/'
        
def occupations(): 
    '''
    We gather occupations from Wikipedia pages whose content was
    slightly cleaned up (removed content in parantheses) and placed in csvs. 
    '''
    classes = defaultdict(dict)
    categories = ['art', 'health', 'other', 'sports', 'stem']
    
    curr_cat = None
    for clss in categories: 
        with open(DATA + 'semantics/Occupations_' + clss + '.csv', 'r') as infile: 
            for line in infile: 
                if line.startswith('#http') or line.startswith('#Note:'): continue
                if line.startswith('#'): 
                    curr_cat = line.strip().replace('#', '')
                    classes[curr_cat]['high'] = []
                    classes[curr_cat]['low'] = []
    for clss in categories: 
        with open(DATA + 'semantics/Occupations_' + clss + '.csv', 'r') as infile: 
            for line in infile: 
                if line.startswith('#http') or line.startswith('#Note:'): continue
                if line.startswith('#'): 
                    curr_cat = line.strip().replace('#', '')
                    continue
                job = line.strip()
                clean_job = re.sub("[\(].*?[\)]", "", job)
                if len(clean_job.split()) >= 3: continue
                for clss in classes: 
                    if clss == curr_cat: 
                        classes[clss]['high'].append(job)
                    else: 
                        classes[clss]['low'].append(job)
                        
    for clss in classes: 
        # remove duplicates
        classes[clss]['high'] = list(set(classes[clss]['high']))
        print(clss.upper())
        print(classes[clss]['high'])
        classes[clss]['low'] = list(set(classes[clss]['low']))
    
    # This version keeps wikipedia job formatting, which helps us find pages
    with open(DATA + 'semantics/cleaned/occupations.json', 'w') as outfile:
        json.dump(classes, outfile)
        
def get_occupation_pages_part1(): 
    '''
    For each occupation, get its wikipedia page and download wikitext
    Some occupations do not have a wikipedia page, in which
    case we leave them out. 
    '''
    glove_vocab = set()
    with open(GLOVE + 'glove.6B.300d.txt', 'r') as infile:
        for line in infile: 
            contents = line.split()
            glove_vocab.add(contents[0])
            
    with open(DATA + 'semantics/cleaned/occupations.json', 'r') as infile:
        classes = json.load(infile)
        
    all_occs = set()
    for clss in classes: 
        all_occs.update(classes[clss]['high'])
        all_occs.update(classes[clss]['low'])
            
    wiki_pages = {} # title : request response
    for wiki_title in all_occs: 
        toks = set(wiki_title.lower().split(' '))
        if len(toks) > 2: 
            # only unigrams and bigrams
            continue
        if toks & glove_vocab != toks: 
            # tokens need to be in glove
            continue
        response = requests.get('https://en.wikipedia.org/w/api.php?action=parse&page=' + wiki_title + '&prop=wikitext&formatversion=2&format=json&redirects')
        if not response.ok: 
            print("Problem with", wiki_title)
        response_dict = json.loads(response.text)
        if 'error' in response_dict: 
            print("Problem with", wiki_title, response_dict)
        else: 
            title = response_dict['parse']['title']
            if 'List of' in title or 'Lists of' in title: 
                continue
            pageid = response_dict['parse']['pageid']
            wiki_pages[wiki_title] = [title, pageid]
    # save wikipedia page IDs
    with open(DATA + 'semantics/occupation_wikipages.json', 'w') as outfile: 
        json.dump(wiki_pages, outfile)         
        
    # This version removes wikipedia job formatting
    for clss in classes: 
        for g in classes[clss]: 
            new_list = []
            old_list = classes[clss][g]
            for occ in old_list: 
                new_occ = re.sub("[\(].*?[\)]", "", occ).lower()
                if new_occ.endswith('s'): 
                    # the sports people are plural, look for singular
                    new_occ = new_occ[:-1]
                new_list.append(new_occ)
            classes[clss][g] = new_list
    with open(DATA + 'semantics/cleaned/occupations.json', 'w') as outfile:
        json.dump(classes, outfile)
    
def get_occupation_pages_part2(): 
    '''
    Use the page IDs from occupation_wikipages.json, 
    grep for the file containing the cleaned wikitext, 
    and then extracts the cleaned wikitext to a separate file. 
    '''
            
    WIKI_TEXT = '/mnt/data0/corpora/wikipedia/text/'
    
    with open(DATA + 'semantics/occupation_wikipages.json', 'r') as infile: 
        occ_pages = json.load(infile)
    pages_occ = {}
    for occ in occ_pages: 
        pages_occ[str(occ_pages[occ][1])] = occ
        
    occ_sents = defaultdict(list)
            
    for folder in tqdm(os.listdir(WIKI_TEXT)): 
        if folder == 'all_files.txt': continue
        for f in os.listdir(WIKI_TEXT + folder): 
            path = WIKI_TEXT + folder + '/' + f
            with open(path, 'r') as infile: 
                soup = BeautifulSoup(infile.read(), features="lxml")
                docs = soup.find_all('doc')
                for doc in docs: 
                    idx = doc.get('id')
                    if idx in pages_occ: 
                        # found an occupation page
                        occ = pages_occ[idx]
                        occ = re.sub("[\(].*?[\)]", "", occ).lower()
                        if occ.endswith('s'): 
                            # the sports people are plural, look for singular
                            occ = occ[:-1]
                        text = doc.get_text().split('\n')
                        sents = []
                        for l in text: 
                            sents.extend(tokenize.sent_tokenize(l))
                        for sent in sents: 
                            sent_lower = sent.lower()
                            matches = re.search(r'\b' + occ + r'\b', sent)
                            if matches is not None: 
                                occ_sents[occ].append(sent)
    
    with open(DATA + 'semantics/occupation_sents.json', 'w') as outfile: 
        json.dump(occ_sents, outfile)
    
def prep_datasets():
    '''
    This gathers and prepares occupation pages for axes evaluation. 
    '''
    #occupations()
    get_occupation_pages_part1()
    get_occupation_pages_part2()

def retrieve_wordnet_axes(): 
    '''
    This function creates WordNet axes. 
    Like in the semaxis paper where poles are expanded using
    neighbors, here poles are expanded using synsets, or groups of synonymous words 
    '''
    glove_vocab = set()
    with open(GLOVE + 'glove.6B.300d.txt', 'r') as infile:
        for line in infile: 
            contents = line.split()
            glove_vocab.add(contents[0])
            
    i = 0
    seen = set() # adjective clusters already seen
    with open(LOGS + 'semantics_val/wordnet_axes.txt', 'w') as outfile: 
        for ss in wn.all_synsets():
            if ss.pos() == 'a': 
                synonyms = set()
                antonyms = set()
                similar = ss.similar_tos() # similar synsets
                for sim_ss in similar: 
                    synonyms.update(sim_ss.lemma_names())
                synonyms.update(ss.lemma_names())
                for lem in ss.lemmas(): # lemmas in this synset             
                    # get antonym lemmas and antonym's similar lemmas
                    ants = lem.antonyms() # list of lemmas
                    for ant in lem.antonyms(): 
                        antonyms.update(ant.synset().lemma_names())
                        for ant_sim_ss in ant.synset().similar_tos(): 
                            antonyms.update(ant_sim_ss.lemma_names())
                # check that word appears in GloVe
                synonyms = synonyms & glove_vocab
                antonyms = antonyms & glove_vocab
                # remove '.' acronyms 
                synonyms = [w for w in synonyms if '.' not in w]
                antonyms = [w for w in antonyms if '.' not in w]
                # check that pole is "robust"
                if len(synonyms) < 3 or len(antonyms) < 3: continue
                synonyms = ','.join(sorted(synonyms))
                antonyms = ','.join(sorted(antonyms))
                if synonyms in seen or antonyms in seen: continue
                outfile.write(ss.name() + '\t' + synonyms + '\t' + antonyms + '\n')
                seen.add(synonyms)
                seen.add(antonyms)
                
def axes_stats(): 
    '''
    Outputs # of axes, and average # of adj per pole. 
    '''
    num_adj = []
    num_axes = 0
    with open(LOGS + 'semantics_val/wordnet_axes.txt', 'r') as infile: 
        for line in infile: 
            num_axes += 1
            contents = line.strip().split('\t')
            synset = contents[0]
            axis1 = contents[1].split(',')
            num_adj.append(len(axis1))
            axis2 = contents[2].split(',')
            num_adj.append(len(axis2))
    print("avg # of adj per pole:", np.mean(num_adj))
    print("# of axes:", num_axes)
    
def prep_person_exp(): 
    with open(DATA + 'semantics/occupation_sents.json', 'r') as infile: 
        occ_sents = json.load(infile) 
        
    new_occ_sents = defaultdict(list)
    for occ in occ_sents: 
        for sent in occ_sents[occ]: 
            new_sent = re.sub(r'\b' + occ + r'\b', 'person', sent)
            new_occ_sents[occ].append(new_sent)
            
    with open(DATA + 'semantics/person_occupation_sents.json', 'w') as outfile: 
        json.dump(new_occ_sents, outfile)
                
def main():
    prep_datasets()
    #retrieve_wordnet_axes()
    #axes_stats()
    prep_person_exp()
    
if __name__ == '__main__':
    main()