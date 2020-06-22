"""
Various helper functions
"""
import csv

ROOT = '/mnt/data0/lucy/manosphere/'
MANUAL_PEOPLE = ROOT + 'data/manual_people.csv'
UD = ROOT + 'logs/urban_dict.csv'

def calculate_ud_coverage(): 
    """
    get the highest ranked definition for a word 
    """
    sings = {}
    plurals = {}
    sing_to_plural = {}
    with open(MANUAL_PEOPLE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['community'].strip() != 'generic': 
                word_sing = row['word (singular)']
                if word_sing.strip() == '': continue
                last_word = word_sing.split()[-1]
                other_words = word_sing.split()[:-1]
                if last_word == 'man': 
                    plural = ' '.join(other_words + ['men'])
                elif last_word == 'woman': 
                    plural = ' '.join(other_words + ['women'])
                else:
                    plural = word_sing + 's'
                word_sing = word_sing.lower()
                plural = plural.lower()
                sings[word_sing] = []
                plurals[plural] = []
                sing_to_plural[word_sing] = plural
    with open(UD, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('|')
            word = contents[0].lower()
            if word in sings: 
                sings[word].append(line)
            elif word in plurals: 
                plurals[word].append(line)
    missing_count = 0
    for w in sings: 
        if len(sings[w]) == 0: 
            if len(plurals[sing_to_plural[w]]) == 0: 
                print(w)
                missing_count += 1
    print(missing_count)

def main(): 
    calculate_ud_coverage()

if __name__ == '__main__':
    main()