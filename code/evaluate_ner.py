import spacy
import csv

ROOT = '/mnt/data0/lucy/manosphere/'
FOLDER = ROOT + 'logs/manual_annotations/'
REDDIT_SAMPLE = FOLDER + 'reddit_sample.txt'
REDDIT_TEXT = FOLDER + 'reddit_text.txt'
REDDIT_ANN = FOLDER + 'reddit_sample.ann'
FORUM_SAMPLE = FOLDER + 'forum_sample.txt'
FORUM_ANN = FOLDER + 'forum_sample.ann'
FORUM_TEXT = FOLDER + 'forum_text.txt'

def correct_indices(): 
    '''
    Adjust gold, manually annotated indices. 
    This is because each line in the annotations starts with 
    metadata such as category, month, subreddit 
    which we don't want to include when ner-tagging things. 
    '''
    nlp = spacy.load("en_core_web_sm")
    
    # output a file without metadata
    txt = []
    with open(REDDIT_SAMPLE, 'r') as infile: 
        reader = csv.reader(infile, delimiter='\t')
        for row in reader: 
            txt.append(row[4])
    with open(REDDIT_TEXT, 'w') as outfile: 
        writer = csv.writer(outfile, delimiter='\t')
        for comment in txt: 
            writer.writerow([comment])
    
    spans = []
    with open(FORUM_ANN, 'r') as infile: 
        for line in infile: 
            contents = line.split('\t') 
            entity = contents[1].split()
            spans.append((int(entity[1]), int(entity[2])))
            
    with open(FORUM_SAMPLE, 'r', newline='\n', encoding='utf-8') as infile:
        #for line in infile: 
        #    print("LINE:", line)
        doc = infile.read()
        for span in spans: 
            print(span, doc[span[0]:span[1]])
        

def main(): 
    correct_indices()


if __name__ == '__main__':
    main()