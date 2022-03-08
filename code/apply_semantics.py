"""
Applies axes to word embeddings
"""
from validate_semantics import load_wordnet_axes, get_poles_bert
from tqdm import tqdm
from scipy.spatial.distance import cosine
import json
import numpy as np
from collections import Counter, defaultdict
from sklearn.neighbors.kde import KernelDensity
from fastdist import fastdist

ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
EMBED_PATH = LOGS + 'semantics_mano/embed/'

def load_manosphere_vecs(): 
    '''
    In the reddit dataset, the size of
    full_reps is (257646, 3072)
    '''
    bert_mean = np.load(LOGS + 'wikipedia/mean_BERT.npy')
    bert_std = np.load(LOGS + 'wikipedia/std_BERT.npy')
    
    vocab_order = []
    full_reps = []
    years = range(2008, 2020)
    for y in tqdm(years): 
        with open(EMBED_PATH + 'reddit_' + str(y) + '.json', 'r') as infile: 
            d = json.load(infile) # { term_category_year : vector }
        for key in sorted(d.keys()): 
            parts = key.split('_')
            term = parts[0]
            cat_year = '_'.join(parts[1:3])
            standard_vec = (np.array(d[key]) - bert_mean) / bert_std
            full_reps.append(standard_vec)
            vocab_order.append(key)
    full_reps = np.array(full_reps)
    print("Number of reps", full_reps.shape)
    return full_reps, vocab_order

def quantify_axes_behavior(): 
    '''
    Finds the axes that showcase the most variance in how
    word biases are distributed. 
    Instead of using 
    '''
    print("getting axes...")
    axes, axes_vocab = load_wordnet_axes()
    # synset : (right_vec, left_vec)
    adj_poles = get_poles_bert(axes, 'bert-base-prob-zscore')
    
    print("getting word vectors...")
    full_reps, vocab_order = load_manosphere_vecs()
    
    print("calculating bias diversity...")
    variances = Counter()
    scores = defaultdict(list) 
    for pole in tqdm(adj_poles): 
        left_vecs, right_vecs = adj_poles[pole]
        left_pole = left_vecs.mean(axis=0)
        right_pole = right_vecs.mean(axis=0)
        microframe = right_pole - left_pole
        # note that this is cosine distance, not cosine similarity
        c_w_f = fastdist.vector_to_matrix_distance(microframe, full_reps, fastdist.cosine, "cosine")
        variances[pole] = np.var(c_w_f)
        scores[pole] = list(c_w_f)

    with open(LOGS + 'semantics_mano/results/variances.json', 'w') as outfile: 
        json.dump(variances, outfile)
        
    with open(LOGS + 'semantics_mano/results/scores.json', 'w') as outfile: 
        json.dump(scores, outfile)
        
    with open(LOGS + 'semantics_mano/results/vocab_order.txt', 'w') as outfile: 
        outfile.write('\n'.join(vocab_order))
        
    print(variances.most_common(20))
    
def examine_outliers(): 
    with open(LOGS + 'semantics_mano/results/scores.json', 'r') as infile: 
        scores = json.load(infile)
        
    with open(LOGS + 'semantics_mano/results/variances.json', 'r') as infile: 
        variances = Counter(json.load(infile))
        
    vocab_order = []
    with open(LOGS + 'semantics_mano/results/vocab_order.txt', 'r') as infile:
        vocab_order = infile.readlines()
        
    N = 10
    with open(LOGS + 'semantics_mano/results/top_and_bottom_words.txt', 'w') as outfile: 
        for tup in variances.most_common(20): 
            pole = tup[0]
            var = tup[1]
            outfile.write(pole + ' ' + str(var) + '\n')
            score_list = scores[pole]
            indices = np.argpartition(score_list, -N)[-N:]
            topN = [vocab_order[idx].strip() for idx in indices]
            outfile.write(', '.join(topN) + '\n')
            indices = np.argpartition(score_list, N)[:N]
            bottomN = [vocab_order[idx].strip() for idx in indices]
            outfile.write(', '.join(bottomN) + '\n')
            outfile.write('\n')

def main(): 
    #quantify_axes_behavior()
    examine_outliers()

if __name__ == '__main__':
    main()