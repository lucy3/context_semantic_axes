import spacy
from common.pipelines import SpacyPipeline
from english.entity_tagger import LitBankEntityTagger
import json
import os

ROOT = '/mnt/data0/lucy/manosphere/'
POSTS = ROOT + 'data/submissions/'
LOGS = ROOT + 'logs/'
COMMENTS = ROOT + 'data/comments/'

def main(): 
    spacy_nlp = spacy.load("en_core_web_sm")
    tagger=SpacyPipeline(spacy_nlp)
    entityPath="/mnt/data0/dbamman/neural-booknlp/neural.booknlp.entities.model"
    tagsetPath="english/entity_cat.tagset"
    entityTagger=LitBankEntityTagger(entityPath, tagsetPath)
    month = '2016-03'
    outfile = open(LOGS + 'tagged_people/' + month, 'w') 
    with open(COMMENTS + 'RC_' + month + '/part-00000', 'r') as infile: 
         for line in infile: 
             d = json.loads(line)
             text = d['body']
             tokens = tagger.tag(text)
             entities = entityTagger.tag(tokens)
             for entity in entities: 
                 if entity[2] == 'NOM_PER': 
                     outfile.write(str(entity[0]) + ' ' + str(entity[1]) + ' ' + str(entity[3]) + '\t')
             outfile.write('\n')
             
    if os.path.exists(POSTS + 'RS_' + month + '/part-00000'): 
        post_path = POSTS + 'RS_' + month + '/part-00000'
    else: 
        post_path = POSTS + 'RS_v2_' + month + '/part-00000'
    with open(post_path, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            text = d['selftext']
            tokens = tagger.tag(text)
            entities = entityTagger.tag(tokens)
            for entity in entities: 
                if entity[2] == 'NOM_PER': 
                    outfile.write(str(entity[0]) + ' ' + str(entity[1]) + ' ' + str(entity[3]) + '\t')
            outfile.write('\n')
    outfile.close()

if __name__ == '__main__':
    main()
