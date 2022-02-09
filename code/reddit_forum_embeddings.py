"""
Script for getting word embeddings 
from reddit and forum data 

One embedding for a word in each
ideology and year 

Example of use: 
python reddit_forum_embeddings.py --dataset reddit --subset 2005
"""
from transformers import BasicTokenizer, BertTokenizerFast, BertModel, BertTokenizer
import argparse
from gram_counting import check_valid_comment, check_valid_post, remove_bots, get_bot_set

ROOT = '/mnt/data0/lucy/manosphere/' 
SUBS = ROOT + 'data/submissions/'
COMS = ROOT + 'data/comments/'
CONTROL = ROOT + 'data/reddit_control/'
FORUMS = ROOT + 'data/cleaned_forums/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str,
                    help='reddit, control, or forum')
parser.add_argument('--subset', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='for reddit/control, should be a year, for forum, should be a forum')

args = parser.parse_args()

def get_vocab(): 
    words = []
    with open(ANN_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            if row['keep'] == 'Y': 
                words.append(row['entity'].lower())
    return words

def batch_reddit(): 
    vocab = get_vocab()
    tokenizer = BasicTokenizer(do_lower_case=True)
    bots = get_bot_set()

    batch_size = 8
    batch_sentences = [] # each item is a list
    batch_words = [] # each item is a list
    batch_sr = []
    curr_batch = []
    curr_words = []
    curr_meta = []
    # go through comment file
    for filename in os.listdir(COMS): 
        if filename.startwith('RC_' + args.subset): 
            m = filename.replace('RC_', '')
            with open(COMS + filename + '/part-00000', 'r') as infile: 
                for line in infile: 
                    if not check_valid_comment(line): continue
                    if not remove_bots(line, bot_set=bots): continue
                    d = json.loads(line)
                    sr = d['subreddit'].lower()
                    tokens = tokenizer.tokenize(d['body'])
                    words_in_line = []
                    for i in range(len(tokens)): 
                        if tokens[i] in vocab: 
                            words_in_line.append(tokens[i])
                    curr_batch.append(tokens)
                    curr_words.append(words_in_line)
                    curr_meta.append(sr)
                    if len(curr_batch) == batch_size: 
                        batch_sentences.append(curr_batch)
                        batch_words.append(curr_words)
                        batch_sr.append(curr_meta)
                        curr_batch = []
                        curr_words = []   
                        curr_meta = []
            if os.path.exists(SUBS + 'RS_' + m + '/part-00000'): 
                post_path = SUBS + 'RS_' + m + '/part-00000'
            else: 
                post_path = SUBS + 'RS_v2_' + m + '/part-00000'
            with open(post_path, 'r') as infile: 
                # go through submission file 
                for line in file: 
                    if not check_valid_post(line): continue
                    if not remove_bots(line, bot_set=bots): continue
                    d = json.loads(line)
                    sr = d['subreddit'].lower()
                    tokens = tokenizer.tokenize(d['selftext'])
                    words_in_line = []
                    for i in range(len(tokens)): 
                        if tokens[i] in vocab: 
                            words_in_line.append(tokens[i])
                    curr_batch.append(tokens)
                    curr_words.append(words_in_line)
                    curr_meta.append(sr)
                    if len(curr_batch) == batch_size: 
                        batch_sentences.append(curr_batch)
                        batch_words.append(curr_words)
                        batch_sr.append(curr_meta)
                        curr_batch = []
                        curr_words = [] 
                        curr_meta = []
                        
            if len(curr_batch) != 0: # fence post
                batch_sentences.append(curr_batch)
                batch_words.append(curr_words)
                batch_sr.append(curr_meta)
    return batch_sentences, batch_words, batch_sr

def get_reddit_embeddings(): 
    year = args.subset
    batch_sentences, batch_words, batch_sr = batch_reddit()
    
    word_reps = {}
    word_counts = Counter()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    layers = [-4, -3, -2, -1] # last four layers
    model.to(device)
    model.eval()
    
    for i, batch in enumerate(tqdm(batch_sentences)): # for every batch
        word_tokenids = {} # { j : { word : [token ids] } }
        encoded_inputs = tokenizer(batch, is_split_into_words=True, padding=True, truncation=True, 
             return_tensors="pt")
        encoded_inputs.to(device)
        outputs = model(**encoded_inputs, output_hidden_states=True)
        states = outputs.hidden_states # tuple
        # batch_size x seq_len x 3072
        vector = torch.cat([states[i] for i in layers], 2) # concatenate last four
        for j in range(len(batch)): # for every example
            # TODO: get category of subreddit 
            word_ids = encoded_inputs.word_ids(j)
            word_tokenids = defaultdict(list) # {word : [token ids]}
            for k, word_id in enumerate(word_ids): # for every token
                if word_id is not None: 
                    curr_word = batch[j][word_id]
                    if curr_word in batch_words[i][j]: 
                        word_tokenids[curr_word].append(k)
            for word in word_tokenids: 
                token_ids_word = np.array(word_tokenids[word]) 
                word_embed = vector[j][token_ids_word]
                word_embed = word_embed.mean(dim=0).detach().cpu().numpy() # average word pieces
                if np.isnan(word_embed).any(): 
                    print("PROBLEM!!!", word, batch[j])
                    return 
                word_cat = word + '_' + cat
                if word_cat not in word_reps: 
                    word_reps[word_cat] = np.zeros(3072)
                word_reps[word_cat] += word_embed
                word_counts[word_cat] += 1
        torch.cuda.empty_cache()
        
    res = {}
    for w in word_counts: 
        res[w] = list(word_reps[w] / word_counts[w])
    with open(LOGS + 'semantics_mano/embed/' + args.dataset + '_' + args.subset + '.json', 'w') as outfile: 
        json.dump(res, outfile)

def get_forum_embeddings(): 
    pass

def get_control_embeddings(): 
    pass

def main(): 
    if 'reddit' == args.dataset: 
        get_reddit_embeddings()
    elif 'forum' == args.dataset: 
        get_forum_embeddings() 
    elif 'control' == args.dataset: 
        get_control_embeddings()

if __name__ == '__main__':
    main()