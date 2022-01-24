'''
Compare semantic differences between nouns.
Filter each dataset for nouns that occur in WordNet. 
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

ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
GLOVE = DATA + 'glove/'
LOGS = ROOT + 'logs/'

def nrc_vad():
    '''
    Input: NRC VAD file
    Output: a json where keys are the different lexicons
    and the inner dictionaries contain high and low value words 
    '''
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
        
def occupations(): 
    '''
    Occupations from labour bureau and wikipedia
    TODO: rewrite
    '''
    data_file = DATA + 'semantics/job_demographics.csv'
    classes = {'stem' : {'high': [], 'low': []},
               'art' : {'high': [], 'low': []},
               'health' : {'high': [], 'low': []}, 
                }
    stem = False
    art = False
    health = False
    with open(data_file, 'r', encoding='utf-8-sig') as infile: 
        reader = csv.DictReader(infile)
        for row in reader:
            job = row['job'].lower().strip()
            if job == '': continue
            if job == 'professional and related occupations': 
                stem = True
            elif job == 'community and social service occupations': 
                stem = False
            elif job == 'arts, design, entertainment, sports, and media occupations': 
                art = True
            elif job == 'healthcare practitioners and technical occupations': 
                art = False
                health = True
            elif job == 'protective service occupations': 
                health = False
            if len(job.split()) < 3: 
                if stem: 
                    classes['stem']['high'].append(job)
                else: 
                    classes['stem']['low'].append(job)
                if art: 
                    classes['art']['high'].append(job)
                else: 
                    classes['art']['low'].append(job)
                if health: 
                    classes['health']['high'].append(job)
                else: 
                    classes['health']['low'].append(job)
    
    for clss in classes: 
        with open(DATA + 'semantics/Occupations_' + clss + '.csv', 'r') as infile: 
            for line in infile: 
                if line.startswith('#'): continue
                job = line.strip().lower()
                if len(job.split()) >= 3: continue
                for other_clss in classes: 
                    if clss == other_clss: 
                        classes[other_clss]['high'].append(job)
                    else: 
                        classes[other_clss]['low'].append(job)
                        
    for clss in classes: 
        # remove duplicates
        classes[clss]['high'] = list(set(classes[clss]['high']))
        classes[clss]['low'] = list(set(classes[clss]['low']))
    
    print(classes['stem']['high'])
    print()
    print(classes['art']['high'])
    print()
    print(classes['health']['high'])
    with open(DATA + 'semantics/cleaned/occupations.json', 'w') as outfile:
        json.dump(classes, outfile)
        
def get_occupation_pages(): 
    '''
    For each occupation, get its wikipedia page and download wikitext
    Some occupations do not have a wikipedia page, in which
    case we leave them out. 
    '''
    classes = ['stem', 'art', 'health']
    wiki_pages = {} # title : request response
    for clss in classes: 
        with open(DATA + 'semantics/Occupations_' + clss + '.csv', 'r') as infile: 
            for line in infile: 
                if line.startswith('#http'): continue
                if line.startswith('#'): 
                    curr_cat = line.strip()
                else: 
                    wiki_title = line.strip().lower()
                    response = requests.get('https://en.wikipedia.org/w/api.php?action=parse&page=' + wiki_title + '&prop=wikitext&formatversion=2&format=json&redirects')
                    if not response.ok: 
                        print("Problem with", wiki_title)
                    response_dict = json.loads(response.text)
                    if 'error' in response_dict: 
                        print("Problem with", wiki_title, response_dict)
                    else: 
                        wikitext = response_dict['parse']['wikitext']
                        wiki_pages[wiki_title] = wikitext
    with open(DATA + 'semantics/occupation_wikipages.json', 'w') as outfile: 
        json.dump(wiki_pages, outfile)         
    
def prep_datasets():
    #nrc_vad()
    #occupations()
    get_occupation_pages()
    
def get_semaxes(): 
    '''
    Default single word axes 
    '''
    axes_file = DATA + 'semantics/732_semaxis_axes.tsv'
    axes = []
    with open(axes_file, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t') 
            axes.append(contents[0])
            axes.append(contents[1])
    return axes

def retrieve_wordnet_axes(): 
    '''
    Uses the wider set of axes from WordNet, as seen
    in the frameaxis paper. 
    
    Like in the semaxis paper where poles are expanded using
    neighbors, here poles are expanded using synonyms
    
    Synsets are groups of synonymous words 
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
                
def load_wordnet_axes(): 
    axes = {}
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
            axes[synset] = (axis1, axis2)
    return axes, axes_vocab

def get_pole_matrix(glove_vecs, axes): 
    '''
    @input: 
    - glove vectors, wordnet axes
    @output: 
    - a matrix of wordnet axes where each row is a pole
    - axes_order.txt: the pole that each row of the matrix refers to
    '''
    poles = []
    adj_matrix = []
    for pole in sorted(axes.keys()): 
        left = axes[pole][0]
        left_vec = []
        for w in left: 
            if w in glove_vecs: 
                left_vec.append(glove_vecs[w])
        if len(left_vec) == 0: continue
        left_vec = np.array(left_vec).mean(axis=0)
        
        right = axes[pole][1]
        right_vec = []
        for w in right: 
            if w in glove_vecs: 
                right_vec.append(glove_vecs[w])
        if len(right_vec) == 0: continue
        right_vec = np.array(right_vec).mean(axis=0)
        
        adj_matrix.append(left_vec)
        adj_matrix.append(right_vec)
        poles.append(pole + '_synonym')
        poles.append(pole + '_antonym')
        
    adj_matrix = np.array(adj_matrix)
    
    with open(LOGS + 'semantics_val/axes_order.txt', 'w') as outfile: 
        for pole in poles: 
            outfile.write(pole + '\n')
    
    np.save(LOGS + 'semantics_val/wordnet_axes.npy', adj_matrix)
    
def get_poles(vec_dict, axes): 
    '''
    @output: 
    - a dictionary of each synset and the vectors that are part of each pole
    '''
    adj_poles = {} # synset : (right_vec, left_vec)
    for pole in sorted(axes.keys()): 
        left = axes[pole][0]
        left_vec = []
        for w in left: 
            if w in vec_dict: 
                left_vec.append(vec_dict[w])
        if len(left_vec) == 0: continue
        left_vec = np.array(left_vec)
        
        right = axes[pole][1]
        right_vec = []
        for w in right: 
            if w in vec_dict: 
                right_vec.append(vec_dict[w])
        if len(right_vec) == 0: continue
        right_vec = np.array(right_vec)
        
        adj_poles[pole] = (left_vec, right_vec)
    return adj_poles

def get_glove_vecs(vocab, axes_vocab): 
    '''
    - get glove representations of a vocab, e.g. occupations
    '''
    bigram_tokens = set()
    for w in vocab: 
        tokens = w.split()
        if len(tokens) == 2: 
            bigram_tokens.update(tokens)
    glove_vecs = {}
    with open(GLOVE + 'glove.6B.300d.txt', 'r') as infile:
        for line in infile: 
            contents = line.split()
            word = contents[0]
            if word in vocab or word in axes_vocab or word in bigram_tokens: 
                vec = np.array([float(i) for i in contents[1:]])
                glove_vecs[word] = vec
    # average representations for bigrams
    for w in vocab: 
        tokens = w.split()
        if len(tokens) == 2: 
            rep = []
            for tok in tokens: 
                if tok in glove_vecs: 
                    rep.append(glove_vecs[tok])
            if len(rep) != 2: continue
            rep = np.mean(np.array(rep), axis=0)
            glove_vecs[w] = rep
    return glove_vecs

def save_inputs_from_json(file_path, lexicon_name): 
    '''
    For a lexicon that has scores for different words,
    saves vectors associated with that lexicon.
    '''
    with open(file_path, 'r') as infile:
        lexicon_dict = json.load(infile)
    vocab = set()
    for c in lexicon_dict: 
        for score in lexicon_dict[c]: 
            vocab.update(lexicon_dict[c][score])
    
    axes, axes_vocab = load_wordnet_axes()
    glove_vecs = get_glove_vecs(vocab, axes_vocab)
    get_pole_matrix(glove_vecs, axes)
    
    for c in lexicon_dict.keys(): 
        word_matrix = []
        score_matrix = []
        word_order = []
        for score in lexicon_dict[c]: 
            for word in lexicon_dict[c][score]: 
                if word not in glove_vecs: continue
                word_matrix.append(glove_vecs[word])
                word_order.append(word)
                if score == 'high': 
                    score_matrix.append(1)
                elif score == 'low': 
                    score_matrix.append(0)
        score_matrix = np.array(score_matrix)
        word_matrix = np.array(word_matrix)
        np.save(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '_scores.npy', score_matrix)
        np.save(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '_words.npy', word_matrix)
        
        with open(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '_vocab.txt', 'w') as outfile: 
            for word in word_order: 
                outfile.write(word + '\n')
    
def load_inputs(file_path, lexicon_name): 
    with open(file_path, 'r') as infile:
        lexicon_dict = json.load(infile)
    adj_matrix = np.load(LOGS + 'semantics_val/wordnet_axes.npy')
    score_matrices = {}
    word_matrices = {} 
    for c in lexicon_dict: 
        score_matrices[c] = np.load(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '_scores.npy')
        word_matrices[c] = np.load(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '_words.npy')

    return adj_matrix, score_matrices, word_matrices
        
def frameaxis_glove(file_path, lexicon_name, calc_effect=False, exp_name=''): 
    with open(file_path, 'r') as infile:
        lexicon_dict = json.load(infile)
    vocab = set()
    for c in lexicon_dict: 
        for score in lexicon_dict[c]: 
            vocab.update(lexicon_dict[c][score])
    axes, axes_vocab = load_wordnet_axes()
    glove_vecs = get_glove_vecs(vocab, axes_vocab)
    adj_poles = get_poles(glove_vecs, axes)
    _, score_matrices, word_matrices = load_inputs(file_path, lexicon_name)
    
    N = 1000 # number of bootstrap samples
    biases = defaultdict(dict) # {c : { pole : (bias_sep, effect, bias1, bias2) } }
    for c in score_matrices: 
        score_matrix = score_matrices[c]
        word_matrix = word_matrices[c]
        
        for pole in adj_poles: 
            left_vecs, right_vecs = adj_poles[pole]
            this_adj_matrix = np.concatenate((left_vecs, right_vecs), axis=0)
            this_word_matrix = word_matrix
            if exp_name == 'pca' or exp_name == 'scaler': 
                scaler = StandardScaler()
                this_adj_matrix = scaler.fit_transform(this_adj_matrix)
                this_word_matrix = scaler.transform(this_word_matrix)
            if exp_name == 'pca': 
                pca = PCA(n_components=5)
                this_adj_matrix = pca.fit_transform(this_adj_matrix)
                this_word_matrix = pca.transform(this_word_matrix)
            left_vecs = this_adj_matrix[:left_vecs.shape[0], :]
            right_vecs = this_adj_matrix[left_vecs.shape[0]:, :]
                
            microframe = left_vecs.mean(axis=0) - right_vecs.mean(axis=0)
            c_w_f = cosine_similarity(this_word_matrix, microframe.reshape(1, -1))
            c_w_f1 = c_w_f[score_matrix == 0]
            c_w_f2 = c_w_f[score_matrix == 1]
            b_t_f1 = np.mean(c_w_f1) # bias 
            b_t_f2 = np.mean(c_w_f2) # bias
            bias_sep = abs(b_t_f1 - b_t_f2)

            if calc_effect: 
                samples = []
                for i in range(N): 
                    idx1 = np.random.choice(c_w_f.shape[0], size=c_w_f1.shape[0], replace=False)
                    sample1 = c_w_f[idx1, :]
                    b_t_sample1 = np.mean(sample1)

                    idx2 = np.random.choice(c_w_f.shape[0], size=c_w_f2.shape[0], replace=False)
                    sample2 = c_w_f[idx2, :]
                    b_t_sample2 = np.mean(sample2)

                    bias_sep_sample = abs(b_t_sample1 - b_t_sample2)
                    samples.append(bias_sep_sample)
                effect = bias_sep - np.mean(samples)
            else: 
                effect = 0
            biases[c][pole] = (bias_sep, effect, b_t_f1, b_t_f2)
                
    if exp_name == '': 
        with open(LOGS + 'semantics_val/' + lexicon_name + '/frameaxis.json', 'w') as outfile:
            json.dump(biases, outfile)
    else: 
        with open(LOGS + 'semantics_val/' + lexicon_name + '/frameaxis_' + exp_name + '.json', 'w') as outfile:
            json.dump(biases, outfile)
        
def loo_val_helper(arr, left_vec, right_vec, exp_name):
    if exp_name == 'pca': 
        this_adj_matrix = np.concatenate((left_vec, right_vec), axis=0)
        scaler = StandardScaler()
        this_adj_matrix = scaler.fit_transform(this_adj_matrix)
        pca = PCA(n_components=5)
        this_adj_matrix = pca.fit_transform(this_adj_matrix)
        left_vec = this_adj_matrix[:left_vec.shape[0], :]
        right_vec = this_adj_matrix[left_vec.shape[0]:, :]
        arr = scaler.transform(arr.reshape(1, -1))
        arr = pca.transform(arr)
    if exp_name == 'scaler': 
        # TODO
        pass
    if exp_name == 'kbest': 
        this_adj_matrix = np.concatenate((left_vec, right_vec), axis=0)
        this_adj_scores = [1] * left_vec.shape[0] + [0] * right_vec.shape[0]
        selector = SelectKBest(f_classif, k=100)
        this_adj_matrix = selector.fit_transform(this_adj_matrix, this_adj_scores)
        left_vec = this_adj_matrix[:left_vec.shape[0], :]
        right_vec = this_adj_matrix[left_vec.shape[0]:, :]
        arr = selector.transform(arr.reshape(1, -1))
    left_pole = left_vec.mean(axis=0)
    right_pole = right_vec.mean(axis=0)
    microframe = right_pole - left_pole
    sim = 1 - spatial.distance.cosine(arr, microframe)
    if math.isnan(sim): print(microframe, arr, sim)
    return sim

def loo(pole, left_vec, right_vec, right_vocab, left_vocab, outfile, exp_name): 
    '''
    called by loo_val_static() and loo_val()
    '''
    left_vec = np.ma.array(left_vec, mask=False)
    right_vec = np.ma.array(right_vec, mask=False)
    for i in range(left_vec.shape[0]): 
        mask = np.ones(left_vec.shape[0], dtype=bool)
        mask[i] = False
        new_left = left_vec[mask,:]
        arr = left_vec[i]
        sim = loo_val_helper(arr, new_left, right_vec, exp_name=exp_name)
        outfile.write(pole + '\t' + left_vocab[i] + '\t' + str(sim) + '\tleft\n')

    for i in range(right_vec.shape[0]): 
        mask = np.ones(right_vec.shape[0], dtype=bool)
        mask[i] = False
        new_right = right_vec[mask,:]
        arr = right_vec[i]
        sim = loo_val_helper(arr, left_vec, new_right, exp_name=exp_name)
        outfile.write(pole + '\t' + right_vocab[i]
                      + '\t' + str(sim) + '\tright\n')
        
def loo_val_static(vec_dict, axes, exp_name): 
    '''
    leave-one-out validation where we calculate the simlarity of 
    one adjective to microframes 
    '''
    with open(LOGS + 'semantics_val/axes_quality_' + exp_name + '.txt', 'w') as outfile: 
        for pole in sorted(axes.keys()): 
            left = axes[pole][0] # list of words
            left_vec = [] # list of vectors 
            left_vocab = []
            for w in left: 
                if w in vec_dict: 
                    left_vec.append(vec_dict[w])
                    left_vocab.append(w)
            right = axes[pole][1]
            right_vec = [] # list of vectors 
            right_vocab = []
            for w in right: 
                if w in vec_dict: 
                    right_vec.append(vec_dict[w])
                    right_vocab.append(w)

            # leave one out 
            loo(pole, left_vec, right_vec, right_vocab, left_vocab, outfile, exp_name)
                
def loo_val(vec_dict, axes, exp_name): 
    with open(LOGS + 'semantics_val/axes_quality_' + exp_name + '.txt', 'w') as outfile: 
        for pole in sorted(axes.keys()): 
            left = axes[pole][0] # list of words
            left_vec = [] # list of vectors 
            left_vocab = []
            for w in left: 
                ss_adj = w + '@' + pole + '_left'
                if ss_adj in vec_dict: 
                    left_vec.append(vec_dict[ss_adj])
                    left_vocab.append(w)
            right = axes[pole][1]
            right_vec = [] # list of vectors 
            right_vocab = []
            for w in right: 
                ss_adj = w + '@' + pole + '_right'
                if ss_adj in vec_dict:  
                    right_vec.append(vec_dict[ss_adj])
                    right_vocab.append(w)

            # leave one out 
            loo(pole, left_vec, right_vec, right_vocab, left_vocab, outfile, exp_name)

def get_bert_vecs(exp_name='bert-default'): 
    with open(LOGS + 'semantics_val/adj_BERT.json', 'r') as infile: 
        bert_vecs = json.load(infile) # { adj@ss : [vector] } 
    if 'zscore' in exp_name: 
        bert_mean = np.load(LOGS + 'wikipedia/mean_BERT.npy')
        bert_std = np.load(LOGS + 'wikipedia/std_BERT.npy')
        for vec in bert_vecs: 
            bert_vecs[vec] = (np.array(bert_vecs[vec]) - bert_mean) / bert_std
    else: 
        for vec in bert_vecs: 
            bert_vecs[vec] = np.array(bert_vecs[vec])
    return bert_vecs

def get_vecs_and_map(in_folder, side, side_pole, vec_dict, word_rep_keys, exp_name, singleton_subs): 
    '''
    Gets vectors for one side of a synset. 
    @outputs
    side_vec: np.ma.array where each row is a vector
    rep_keys_map: { word : [indices that make up the word's agg vector] } 
    sub_map: { word : [additional indices to leave out] }
    '''
    pole = side_pole.split('_')[0]
    sub_map = defaultdict(list) 
    if side_pole not in word_rep_keys: 
        # fall back on bert random
        side_vec = [] # list of vectors 
        rep_keys_map = defaultdict(list) 
        i = 0
        for w in side: 
            ss_adj = w + '@' + side_pole
            if ss_adj in vec_dict:  
                side_vec.append(vec_dict[ss_adj])
                rep_keys_map[w].append(i)
                i += 1      
    else: 
        rep_keys = word_rep_keys[side_pole] # [[line_num, word]]
        rep_keys_map = defaultdict(list) 
        for i, rk in enumerate(rep_keys): 
            line_num = rk[0]
            w = rk[1]
            rep_keys_map[w].append(i)
            for s in singleton_subs[pole + '_' + line_num + '_' + w]: 
                sub_map[s].append(i)
        side_vec = np.load(in_folder + side_pole + '.npy')
        if 'zscore' in exp_name: 
            bert_mean = np.load(LOGS + 'wikipedia/mean_BERT.npy')
            bert_std = np.load(LOGS + 'wikipedia/std_BERT.npy')
            side_vec = (side_vec - bert_mean) / bert_std
    side_vec = np.ma.array(side_vec, mask=False)
    return side_vec, rep_keys_map, sub_map

def loo_val_subs(vec_dict, in_folder, axes, exp_name): 
    if exp_name.startswith('bert-base-sub'): 
        subfile_name = 'sub_lines_base-substitutes.json'
    with open(LOGS + 'wikipedia/' + subfile_name, 'r') as infile:
        # cases where a substitute is deciding factor in overlap
        singleton_subs = defaultdict(list, json.load(infile)) # {pole + '_' + substitute : [[line_ID, adj]]}
    with open(in_folder + 'word_rep_key.json', 'r') as infile: 
        word_rep_keys = json.load(infile)
    with open(LOGS + 'semantics_val/axes_quality_' + exp_name + '.txt', 'w') as outfile: 
        for pole in tqdm(sorted(axes.keys())): 
            left = axes[pole][0] # list of words
            left_pole = pole + '_left'
            left_vec, lrep_keys_map, lsub_map = get_vecs_and_map(in_folder, left, left_pole, vec_dict, \
                                                               word_rep_keys, exp_name, singleton_subs)
            
            right = axes[pole][1]
            right_pole = pole + '_right'
            right_vec, rrep_keys_map, rsub_map = get_vecs_and_map(in_folder, right, right_pole, vec_dict, \
                                                               word_rep_keys, exp_name, singleton_subs)
            
            for w in lrep_keys_map: 
                idx = lrep_keys_map[w]
                idx_sub = lsub_map[w]
                mask = np.ones(left_vec.shape[0], dtype=bool)
                mask[idx] = False # mask out vectors corresponding to word
                mask[idx_sub] = False # mask out vectors from contexts where word is sole substitute
                new_left = left_vec[mask,:]
                if new_left.shape[0] == 0: continue
                arr = left_vec[idx,:].mean(axis=0) 
                sim = loo_val_helper(arr, new_left, right_vec, exp_name=exp_name)
                outfile.write(pole + '\t' + w + '\t' + str(sim) + '\tleft\n')
                
            for w in rrep_keys_map: 
                idx = rrep_keys_map[w]
                idx_sub = rsub_map[w]
                mask = np.ones(right_vec.shape[0], dtype=bool)
                mask[idx] = False
                mask[idx_sub] = False
                new_right = right_vec[mask,:]
                if new_right.shape[0] == 0: continue
                arr = right_vec[idx,:].mean(axis=0)
                sim = loo_val_helper(arr, left_vec, new_right, exp_name=exp_name)
                outfile.write(pole + '\t' + w + '\t' + str(sim) + '\tright\n')

def inspect_axes(exp_name): 
    axes, axes_vocab = load_wordnet_axes()
    vocab = set()
    if exp_name in ['default', 'kbest', 'scaler', 'pca']: 
        vec_dict = get_glove_vecs(vocab, axes_vocab)
        loo_val_static(vec_dict, axes, exp_name)
    elif exp_name in ['bert-default', 'bert-zscore']: 
        vec_dict = get_bert_vecs(exp_name)
        loo_val(vec_dict, axes, exp_name)
    elif 'sub' in exp_name and 'bert' in exp_name: 
        vec_dict = get_bert_vecs(exp_name)
        in_folder = LOGS + 'wikipedia/substitutes/' + exp_name + '/'
        loo_val_subs(vec_dict, in_folder, axes, exp_name)
    
def main(): 
#     prep_datasets()
#     retrieve_wordnet_axes()
    #inspect_axes('default')
    #inspect_axes('bert-default')
    #inspect_axes('bert-zscore')
    #inspect_axes('bert-base-sub')
    inspect_axes('bert-base-sub-mask')
#     save_inputs_from_json(DATA + 'semantics/cleaned/occupations.json', 'occupations')
#     frameaxis_glove(DATA + 'semantics/cleaned/occupations.json', 'occupations')
#     frameaxis_glove(DATA + 'semantics/cleaned/nrc_vad.json', 'vad')
#     frameaxis_glove(DATA + 'semantics/cleaned/occupations.json', 'occupations', exp_name='pca')
#     frameaxis_glove(DATA + 'semantics/cleaned/nrc_vad.json', 'vad', exp_name='pca')
#     frameaxis_glove(DATA + 'semantics/cleaned/occupations.json', 'occupations', exp_name='scaler')
#     frameaxis_glove(DATA + 'semantics/cleaned/nrc_vad.json', 'vad', exp_name='scaler')
    #prep_datasets()

if __name__ == '__main__':
    main()
