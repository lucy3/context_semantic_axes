"""
This file is for getting BERT
and RoBERTa substitutes for adjectives
in Wikipedia sentences.

Some code modified from (Bill) Yuchen Lin's example:
https://gist.github.com/yuchenlin/a2f42d3c4378ed7b83de65c7a2222eb2
"""
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm

def predict_masked_sent(tokens, masked_index, tokenizer, top_k=10):
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda') 

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))

def predict_substitutes(): 
    with open(LOGS + 'wikipedia/adj_lines.json', 'r') as infile: 
        adj_lines = json.load(infile) # {adj : [line IDs]}
        
    lines_adj = defaultdict(list) # {line ID: [adj]}
    for adj in adj_lines: 
        for line in adj_lines[adj]: 
            lines_adj[str(line)].append(adj.replace('xqxq', '-'))
            
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    model.to('cuda')
    
    num_lines = len(lines_adj)
            
    with open(LOGS + 'wikipedia/' + vocab_name + '_data/part-00000', 'r') as infile: 
        for line in tqdm(infile, total=num_lines):
            contents = line.split('\t')
            line_num = contents[0]
            text = '\t'.join(contents[1:])
            text = wtp.remove_markup(text).lower()
            words_in_line = lines_adj[line_num]
            new_words_in_line = []
            for w in words_in_line: 
                if '-' in w:  
                    sub_w = w.replace('-', 'xqxq')
                    text = text.replace(w, sub_w)
                    new_words_in_line.append(sub_w)
                else: 
                    new_words_in_line.append(w)
            
            text = "[CLS] " + text + " [SEP]"
            tokens = tokenizer.tokenize(text)
            for 
            masked_index = tokens.index("[MASK]")
            predict_masked_sent(tokens, masked_index, tokenizer)

def main(): 
    predict_substitutes()

if __name__ == '__main__':
    main()
