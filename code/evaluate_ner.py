import spacy
import csv
import re

ROOT = '/mnt/data0/lucy/manosphere/'
FOLDER = ROOT + 'logs/manual_annotations/'
REDDIT_SAMPLE = FOLDER + 'reddit_sample.txt'
REDDIT_TEXT = FOLDER + 'reddit_text.txt'
REDDIT_ANN = FOLDER + 'reddit_sample.ann'
FORUM_SAMPLE = FOLDER + 'forum_sample.txt'
FORUM_ANN = FOLDER + 'forum_sample.ann'
FORUM_TEXT = FOLDER + 'forum_text.txt'

def fix_emojis(text): 
    '''
    Some emojis count as extra characters
    '''
    text = text.replace('👏', 'xx').replace('😹', 'xx').replace('🤧', 'xx').replace('😫', 'xx')
    return text

def reformat_text_only(): 
    # output a file without metadata
    txt = []
    with open(REDDIT_SAMPLE, 'r') as infile: 
        reader = csv.reader(infile, delimiter='\t')
        for row in reader: 
            txt.append(row[4])
    with open(REDDIT_TEXT, 'w') as outfile: 
        for comment in txt: 
            outfile.write(comment + '\n')
    txt = []
    with open(FORUM_SAMPLE, 'r') as infile: 
        reader = csv.reader(infile, delimiter='\t')
        for row in reader: 
            txt.append(row[2])
    with open(FORUM_TEXT, 'w') as outfile: 
        for comment in txt: 
            outfile.write(comment + '\n')
    
def match_annotations_to_text(): 
    '''
    Align characters in Python and text
    '''
    spans = []
    with open(REDDIT_ANN, 'r') as infile: 
        for line in infile: 
            contents = line.split('\t') 
            entity = contents[1].split()
            spans.append((int(entity[1]), int(entity[2])))
            
    with open(REDDIT_SAMPLE, 'r', newline='\n', encoding='utf-8') as infile:
        doc = infile.read()
        doc = fix_emojis(doc)
        for span in spans: 
            print(span, doc[span[0]:span[1]])

def evaluate_annotations(manual_path, text_path, ner_path): 
    spacy_nlp = spacy.load("en_core_web_sm")
    gold_spans = [] # character spans
    with open(manual_path, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t') 
            entity = contents[1].split()
            gold_spans.append((int(entity[1]), int(entity[2])))
    tok_spans = []
    with open(ner_path, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split(' ')
            tok_spans.append((int(contents[0]), int(contents[1]), ' '.join(contents[2:])))
            
    tagged_spans = [] # character spans
    with open(text_path, 'r', newline='\n', encoding='utf-8') as infile: 
        text = infile.read()
        #text = re.sub("\s+", " ", text)
        text = fix_emojis(text)
        doc = spacy_nlp(text)
        for span in tok_spans: 
            start = span[0]
            end = span[1]
            entity_name = span[2]
            end_idx = doc[end].idx + len(doc[end])
            #print(doc[start:end+1])
            tagged_spans.append((doc[start].idx, doc[end].idx + len(doc[end])))
    #print(sorted(gold_spans)[-10:])
    #print(sorted(tagged_spans)[-10:])   
    
    gold_spans = set(gold_spans)
    tagged_spans = set(tagged_spans)
    precision = len(gold_spans & tagged_spans) / len(tagged_spans)
    recall = len(gold_spans & tagged_spans) / len(gold_spans)
    f1 = 2*(precision*recall) / (precision + recall)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    

def main(): 
    print("Combined")
    evaluate_annotations(FOLDER + 'reddit_text.ann', REDDIT_TEXT, FOLDER + 'reddit_combined')
    print("Ace")
    evaluate_annotations(FOLDER + 'reddit_text.ann', REDDIT_TEXT, FOLDER + 'reddit_ace')
    print("Litbank")
    evaluate_annotations(FOLDER + 'reddit_text.ann', REDDIT_TEXT, FOLDER + 'reddit_litbank')
    # TODO: when reformatting glossary word sample, get rid of any duplicates


if __name__ == '__main__':
    main()
