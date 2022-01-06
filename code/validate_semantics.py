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
    '''
    data_file = DATA + 'semantics/job_demographics.csv'
    classes = {'gender' : {'high': [], 'low': []}, 
               'stem' : {'high': [], 'low': []},
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
                if row['Women'] != '–':
                    fem_percent = float(row['Women'])
                    if fem_percent < 25: 
                        classes['gender']['high'].append(job)
                    elif fem_percent > 75: 
                        classes['gender']['low'].append(job)
                
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
        if clss == 'gender': continue
        with open(DATA + 'semantics/Occupations_' + clss + '.csv', 'r') as infile: 
            for line in infile: 
                if line.startswith('#'): continue
                job = line.strip().lower()
                if len(job.split()) >= 3: continue
                for other_clss in classes: 
                    if other_clss == 'gender': continue
                    if clss == other_clss: 
                        classes[other_clss]['high'].append(job)
                    else: 
                        classes[other_clss]['low'].append(job)
                        
    for clss in classes: 
        # remove duplicates
        classes[clss]['high'] = list(set(classes[clss]['high']))
        classes[clss]['low'] = list(set(classes[clss]['low']))
    
    print(classes['gender']['high'])
    print(classes['gender']['low'])
    print()
    print(classes['stem']['high'])
    print()
    print(classes['art']['high'])
    print()
    print(classes['health']['high'])
    with open(DATA + 'semantics/cleaned/occupations.json', 'w') as outfile:
        json.dump(classes, outfile)
    
def prep_datasets():
    #nrc_vad()
    occupations()
    
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

def get_glove_vecs(vocab, axes_vocab): 
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
            
def lda_glove(file_path, lexicon_name):
    adj_matrix, score_matrices, word_matrices = load_inputs(file_path, lexicon_name)

    for c in score_matrices: 
        score_matrix = score_matrices[c]
        word_matrix = word_matrices[c]
        print(c)
        
        clf = LinearDiscriminantAnalysis()
        scaler = StandardScaler()
        #pca = PCA(n_components=100) # PCA to number of samples
        word_matrix = scaler.fit_transform(word_matrix)
        #word_matrix = pca.fit_transform(word_matrix)
        t_matrix = clf.fit_transform(word_matrix, score_matrix)
        np.save(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '.npy', t_matrix)
                
        this_adj_matrix = scaler.transform(adj_matrix)
        #this_adj_matrix = pca.transform(this_adj_matrix)
        this_adj_matrix = clf.transform(this_adj_matrix)
        np.save(LOGS + 'semantics_val/' + lexicon_name + '/' + c + '_axes.npy', this_adj_matrix)
        
def frameaxis_glove(file_path, lexicon_name, calc_effect=False): 
    adj_matrix, score_matrices, word_matrices = load_inputs(file_path, lexicon_name)
    N = 1000 # number of bootstrap samples
    biases = defaultdict(dict) # {c : { pole : (bias, effect) } }
    for c in score_matrices: 
        score_matrix = score_matrices[c]
        word_matrix = word_matrices[c]
        
        for i in range(adj_matrix.shape[0]): 
            if (i - 1) % 2 == 0:
                microframe = adj_matrix[i] - adj_matrix[i-1]
                c_w_f = cosine_similarity(word_matrix, microframe.reshape(1, -1))
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
                biases[c][i] = (bias_sep, effect, b_t_f1, b_t_f2)
                
    with open(LOGS + 'semantics_val/' + lexicon_name + '/frameaxis.json', 'w') as outfile:
        json.dump(biases, outfile)
    
def main(): 
    #save_inputs_from_json(DATA + 'semantics/cleaned/occupations.json', 'occupations')
    #save_inputs_from_json(DATA + 'semantics/cleaned/nrc_vad.json', 'vad')
    #lda_glove(DATA + 'semantics/cleaned/occupations.json', 'occupations')
    frameaxis_glove(DATA + 'semantics/cleaned/occupations.json', 'occupations')
    #frameaxis_glove(DATA + 'semantics/cleaned/nrc_vad.json', 'vad')
    #lda_glove(DATA + 'semantics/cleaned/nrc_vad.json', 'vad')
    #prep_datasets()

if __name__ == '__main__':
    main()