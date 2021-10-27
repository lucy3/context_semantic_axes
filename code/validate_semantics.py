'''
Compare semantic differences between nouns.
Filter each dataset for nouns that occur in WordNet. 
'''
from collections import defaultdict
import json
from nltk.corpus import wordnet as wn
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
GLOVE = DATA + 'glove/'
LOGS = ROOT + 'logs/'

def nrc_vad():
    data_file = DATA + 'semantics/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt'
    classes = {'valence' : {'high': [], 'low': []}, 
               'arousal' : {'high': [], 'low': []}, 
               'dominance' : {'high': [], 'low': []}, 
                }
    with open(data_file, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            word = contents[0]
            syns = wn.synsets(word)
            if 'n' not in set([x.pos() for x in syns]): continue
            val = float(contents[1])
            if val > 0.75: 
                classes['valence']['high'].append(word)
            elif val < 0.25: 
                classes['valence']['low'].append(word)
            aro = float(contents[2])
            if aro > 0.75: 
                classes['arousal']['high'].append(word)
            elif aro < 0.25: 
                classes['arousal']['low'].append(word)
            dom = float(contents[3])
            if dom > 0.75: 
                classes['dominance']['high'].append(word)
            elif dom < 0.25: 
                classes['dominance']['low'].append(word)
    with open(DATA + 'semantics/cleaned/nrc_vad.json', 'w') as outfile:
        json.dump(classes, outfile)
    
def nrc_emotion(): 
    classes = {}
    emotions = ['anger', 'fear', 'joy', 'sadness']
    
def prep_datasets():
    nrc_vad()
    nrc_emotion()
    
def get_axes(): 
    axes_file = DATA + 'semantics/732_semaxis_axes.tsv'
    axes = []
    with open(axes_file, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t') 
            axes.append(contents[0])
            axes.append(contents[1])
    return axes
    
def type_embeddings():
    with open(DATA + 'semantics/cleaned/nrc_vad.json', 'r') as infile:
        vad = json.load(infile)
    vocab = set()
    for c in vad: 
        for score in vad[c]: 
            vocab.update(vad[c][score])
            
    antonyms = ['dominant', 'submissive', 'active', 'passive', 'positive', 'negative', 
                'great', 'terrible', 'excited', 'unexcited', 'worthy', 'useless']
    axes = get_axes()
    
    glove_vecs = {}
    with open(GLOVE + 'glove.6B.300d.txt', 'r') as infile:
        for line in infile: 
            contents = line.split()
            word = contents[0]
            if word in vocab or word in antonyms or word in axes: 
                vec = np.array([float(i) for i in contents[1:]])
                glove_vecs[word] = vec
                
    adj_matrix = []
    for adj in antonyms: 
        adj_matrix.append(glove_vecs[adj])
    adj_matrix = np.array(adj_matrix)
    
    axes_matrix = []
    axes_order = []
    for i, w in enumerate(axes): 
        if i % 2 == 0 and w in glove_vecs and axes[i+1] in glove_vecs:
            axes_matrix.append(glove_vecs[w])
            axes_matrix.append(glove_vecs[axes[i+1]])
            axes_order.append(w)
            axes_order.append(axes[i+1])
            
    with open(LOGS + 'semantics_val/axes_order.txt', 'w') as outfile: 
        for w in axes_order: 
            outfile.write(w + '\n')
                
    for c in vad: 
        word_matrix = []
        score_matrix = []
        word_order = []
        for score in vad[c]: 
            for word in vad[c][score]: 
                if word not in glove_vecs: continue
                word_matrix.append(glove_vecs[word])
                word_order.append(word)
                if score == 'high': 
                    score_matrix.append(1)
                elif score == 'low': 
                    score_matrix.append(0)
        score_matrix = np.array(score_matrix)
        word_matrix = np.array(word_matrix)
        
        clf = LinearDiscriminantAnalysis()
        t_matrix = clf.fit_transform(word_matrix, score_matrix)
        np.save(LOGS + 'semantics_val/' + c + '.npy', t_matrix)
        with open(LOGS + 'semantics_val/' + c + '_vocab.txt', 'w') as outfile: 
            for word in word_order: 
                outfile.write(word + '\n')
                
        t_adj_matrix = clf.transform(adj_matrix)
        np.save(LOGS + 'semantics_val/' + c + '_adj.npy', t_adj_matrix)
        
        t_axes_matrix = clf.transform(axes_matrix)
        np.save(LOGS + 'semantics_val/' + c + '_axes.npy', t_axes_matrix)
    
def main(): 
    prep_datasets()
    type_embeddings()

if __name__ == '__main__':
    main()