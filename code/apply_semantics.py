"""
Applies axes to word embeddings
"""
from validate_semantics import load_wordnet_axes, get_poles_bert
from tqdm import tqdm
from scipy.spatial.distance import cosine
import json
import numpy as np
from collections import Counter, defaultdict
from fastdist import fastdist
from helpers import get_vocab
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
EMBED_PATH = LOGS + 'semantics_mano/embed/'
AGG_EMBED_PATH = LOGS + 'semantics_mano/agg_embed/'

def load_manosphere_vecs(): 
    '''
    Load z-scored embeddings for each vocabulary term
    '''
    bert_mean = np.load(LOGS + 'wikipedia/mean_BERT.npy')
    bert_std = np.load(LOGS + 'wikipedia/std_BERT.npy')
    
    vocab_order = []
    full_reps = []
    with open(AGG_EMBED_PATH + 'mano_overall.json', 'r') as infile: 
        d = json.load(infile) # {term : vector}
    for term in sorted(d.keys()): 
        standard_vec = (np.array(d[term]) - bert_mean) / bert_std
        vocab_order.append(term)
        full_reps.append(standard_vec)
    full_reps = np.array(full_reps)
    print("Number of reps", full_reps.shape)
    return full_reps, vocab_order

def get_good_axes(): 
    '''
    This is copied from axes_occupation_viz.ipynb. 
    '''
    quality_file_path = LOGS + 'semantics_val/axes_quality_bert-base-prob-zscore.txt'
    scores = defaultdict(dict) # {synset: {word : (predicted, true)}}
    with open(quality_file_path, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            scores[contents[0]][contents[1]] = (float(contents[2]), contents[3])
    avg_scores = Counter()
    good_synsets = set()
    for synset in scores: 
        left_scores = []
        right_scores = []
        for w in scores[synset]: 
            if scores[synset][w][1] == 'left': 
                left_scores.append(-1*scores[synset][w][0])
            else: 
                right_scores.append(scores[synset][w][0])
        if left_scores != []: 
            # some are empty since they only had one word with reps
            avg_scores[synset + '_left'] = np.mean(left_scores) 
        if right_scores != []: 
            avg_scores[synset + '_right'] = np.mean(right_scores) 
        if avg_scores[synset + '_left'] >= 0 and avg_scores[synset + '_right'] >= 0: 
            good_synsets.add(synset)
    return good_synsets

def project_onto_axes(): 
    '''
    The output is a dictionary of axis: list of scores, in order of full_reps
    '''
    print("getting axes...")
    axes, axes_vocab = load_wordnet_axes()
    # synset : (right_vec, left_vec)
    adj_poles = get_poles_bert(axes, 'bert-base-prob-zscore')
    good_axes = get_good_axes()
    
    print("getting word vectors...")
    full_reps, vocab_order = load_manosphere_vecs()
    
    print("calculating bias of every word to every axis...")
    variances = Counter()
    scores = defaultdict(list) 
    for pole in tqdm(adj_poles): 
        if pole not in good_axes: continue
        left_vecs, right_vecs = adj_poles[pole]
        left_pole = left_vecs.mean(axis=0)
        right_pole = right_vecs.mean(axis=0)
        microframe = right_pole - left_pole
        # note that this is cosine distance, not cosine similarity
        c_w_f = fastdist.vector_to_matrix_distance(microframe, full_reps, fastdist.cosine, "cosine")
        scores[pole] = list(c_w_f)
        
    with open(LOGS + 'semantics_mano/results/scores.json', 'w') as outfile: 
        json.dump(scores, outfile)
        
    with open(LOGS + 'semantics_mano/results/vocab_order.txt', 'w') as outfile: 
        outfile.write('\n'.join(vocab_order))
            
def get_overall_embeddings(): 
    '''
    Reaggregates based on per-year, per-community/platform
    embeddings and their counts
    This way each vocab word has one embedding. 
    '''
    total_count = Counter() # {term : count}
    overall_vec = {}
    
    # go through reddit
    years = range(2008, 2020)
    for y in tqdm(years):
        with open(EMBED_PATH + 'reddit_' + str(y) + '.json', 'r') as infile: 
            d = json.load(infile) # { term_category_year : vector }
        with open(EMBED_PATH + 'reddit_' + str(y) + '_wordcounts.json', 'r') as infile: 
            word_counts = json.load(infile)
        for key in sorted(d.keys()): 
            count = word_counts[key]
            parts = key.split('_')
            term = parts[0]
            vec = np.array(d[key])*count
            total_count[term] += count
            if term not in overall_vec: 
                overall_vec[term] = np.zeros(3072)
            overall_vec[term] += vec
    forums = ['avfm', 'mgtow', 'incels', 'pua_forum', 'red_pill_talk', 'rooshv', 'the_attraction']
    # go through forum 
    for f in tqdm(forums): 
        with open(EMBED_PATH + 'forum_' + f + '.json', 'r') as infile: 
            d = json.load(infile) # { term_category_year : vector }
        with open(EMBED_PATH + 'forum_' + f + '_wordcounts.json', 'r') as infile: 
            word_counts = json.load(infile)
        for key in sorted(d.keys()): 
            count = word_counts[key]
            parts = key.split('_')
            term = parts[0]
            vec = np.array(d[key])*count
            total_count[term] += count
            if term not in overall_vec: 
                overall_vec[term] = np.zeros(3072)
            overall_vec[term] += vec
    
    for term in overall_vec: 
        overall_vec[term] = list(overall_vec[term] / total_count[term]) 
    with open(AGG_EMBED_PATH + 'mano_overall.json', 'w') as outfile: 
        json.dump(overall_vec, outfile)
        
def pca_experiment(): 
    print("getting axes...")
    axes, axes_vocab = load_wordnet_axes()
    # synset : (right_vec, left_vec)
    adj_poles = get_poles_bert(axes, 'bert-base-prob-zscore')
    good_axes = get_good_axes()
    
    print("getting word vectors...")
    full_reps, vocab_order = load_manosphere_vecs()
    
    with open(LOGS + 'coref_results/mano_gender_labels.json', 'r') as infile: 
        gender_labels = json.load(infile)
    
    fem_reps = []
    masc_reps = []
    for i, term in enumerate(vocab_order): 
        if term not in gender_labels: continue
        if gender_labels[term] > 0.75: 
            fem_reps.append(full_reps[i])
        elif gender_labels[term] < 0.25: 
            masc_reps.append(full_reps[i])
    fem_reps = np.array(fem_reps)
    masc_reps = np.array(masc_reps)
    
    print("applying PCA transformation on manosphere vecs") 
    pca_masc = PCA()
    scaler_masc = StandardScaler()
    scaler_masc.fit(masc_reps)
    masc_reps = scaler_masc.transform(masc_reps)
    pca_masc.fit(masc_reps)
    new_masc_reps = pca_masc.transform(masc_reps)
    np.save(AGG_EMBED_PATH + 'pca_mano_masc.npy', new_masc_reps)
    
    pca_fem = PCA()
    scaler_fem = StandardScaler()
    scaler_fem.fit(fem_reps)
    fem_reps = scaler_fem.transform(fem_reps)
    pca_fem.fit(fem_reps)
    new_fem_reps = pca_fem.transform(fem_reps)
    np.save(AGG_EMBED_PATH + 'pca_mano_fem.npy', new_fem_reps)
    
    print("applying PCA transformation on poles")
    variances = Counter()
    scores = defaultdict(list) 
    fem_ret = {} # {pole: [new_left, new_right]}
    masc_ret = {} # {pole: [new_left, new_right]}
    for pole in tqdm(adj_poles): 
        if pole not in good_axes: continue
        left_vecs, right_vecs = adj_poles[pole]
        left_pole = left_vecs.mean(axis=0).reshape(1, -1)
        right_pole = right_vecs.mean(axis=0).reshape(1, -1)
        
        scaled_left = scaler_fem.transform(left_pole)
        scaled_right = scaler_fem.transform(right_pole)
        new_left = pca_fem.transform(scaled_left).flatten()
        new_right = pca_fem.transform(scaled_right).flatten()
        fem_ret[pole] = [list(new_left), list(new_right)]
        
        scaled_left = scaler_masc.transform(left_pole)
        scaled_right = scaler_masc.transform(right_pole)
        new_left = pca_masc.transform(scaled_left).flatten()
        new_right = pca_masc.transform(scaled_right).flatten()
        masc_ret[pole] = [list(new_left), list(new_right)]
    with open(LOGS + 'semantics_mano/pca_fem_poles.json', 'w') as outfile: 
        json.dump(fem_ret, outfile)
    with open(LOGS + 'semantics_mano/pca_masc_poles.json', 'w') as outfile: 
        json.dump(masc_ret, outfile)

def main(): 
    #get_overall_embeddings()
    #project_onto_axes()
    pca_experiment()

if __name__ == '__main__':
    main()
