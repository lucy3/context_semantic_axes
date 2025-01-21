# Discovering Differences in the Representation of People using Contextualized Semantic Axes

[Spreadsheets](https://docs.google.com/spreadsheets/d/11QGo0hjN-q-BDhdWX5BZpROa3OZ7lxZe80zNvKqv4TU/edit?usp=sharing) containing vocabulary and subreddits. 

## Code

### Meta
- `helpers.py`: helper functions

### Dataset
- `scrape_pushshift.py`: for downloading all of Reddit
- `filter_reddit.py`: creating the Reddit datasets
- `forum_helpers.py`: organize forum data 
- `gram_counting.py`: count all unigrams and bigrams in dataset 
- `count_viz.ipynb`: verifying that our dataset matches patterns from Ribeiro et al.

### Vocabulary

- `data_sampler.py`: reservoir sampling examples for NER evaluation, for context-level manosphere analyses
- `evaluate_ner.py`: evaluate based on human-annotated data
- Some scripts from booknlp multilingual for running NER model on entire dataset 
- `find_people.py`: to read in NER output, inspect glossary words, and create spreadsheet for manual annotation 
- `people_viz.ipynb`: for examining vocab
- `lexical_change.py`: for creating time series of words 
- `k_spectral_centroid.py`: for visualizing how words relate to waves of different communities 
- `time_series_plots.ipynb`: for examining time series for vocab
- `coref_forums.py`, `coref_reddit_control.py`, `coref_reddit.py`, `coref_dating.py`: running coref on different forum/Reddit datasets
- `coref_job_files.py`: creates job files for coref 
- `coref_helper.py`: analyzes coref output 
- `coref_viz.ipynb`: figuring out gender inference steps

### Building and validating semantic axes

- `setup_semantics.py`: finds occupation pages and creates WordNet axes
- `wikipedia_embeddings.py`: getting adjective and occupation embeddings from wikipedia 
- `axis_substitutes.py`: getting "good" contexts for adjectives in Wikipedia sentences.
- `validate_semantics.py`: functions for applying axes on occupation dataset (this contains functions for loading axes) 
- `axes_occupation_viz.ipynb`: evaluate axes on occupation data

`wikipedia/substitutes/bert-default` can be found [here](https://drive.google.com/file/d/1-EQ9V9xuuEJN09ju5qPysHbT_OzGPNHR/view?usp=sharing). 

`wikipedia/substitutes/bert-base-prob` can be found [here](https://drive.google.com/file/d/1XVmfWUy_EubnmAAf6OaRQpQn_n9IPxFd/view?usp=sharing). You will need both this and `bert-default` since we backoff to `bert-default` for cases where words are split into wordpieces. 

The z-scored versions of these vectors are much better than their original versions: 
```
from validate_semantics import load_wordnet_axes, get_poles_bert, get_good_axes

axes, axes_vocab = load_wordnet_axes()
adj_poles = get_poles_bert(axes, 'bert-base-prob-zscore')
good_axes = get_good_axes() # get axes that are self-consistent

for pole in tqdm(adj_poles): 
    if pole not in good_axes: continue
    left_vecs, right_vecs = adj_poles[pole]
    left_pole = left_vecs.mean(axis=0)
    right_pole = right_vecs.mean(axis=0)
    microframe = right_pole - left_pole
```

### Semantic differences and change 

- `prep_embedding_data.py`: prep data for getting embeddings 
- `reddit_forum_embeddings.py`: get term-level embeddings for Reddit/forums
- `apply_semantics.py`: apply axes to Reddit and forum embeddings 
- `semantics_viz.ipynb`: visualizing semantic axes' output 
