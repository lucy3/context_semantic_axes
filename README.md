# A Diachronic Typology of People in the Manosphere and Related Online Communities

## Code

Each script should be thoroughly commented.

### Meta
- `helpers.py`: helper functions

### Dataset
- `scrape_pushshift.py`: for downloading all of Reddit
- `filter_reddit.py`: creating the Reddit datasets
- `forum_helpers.py`: organize forum data 
- `gram_counting.py`: count all unigrams and bigrams in dataset 
- `count_viz.ipynb`: verifying that our dataset matches patterns from Ribeiro et al.

### Vocabulary

- `data_sampler.py`: sampling examples for NER evaluation
- `evaluate_ner.py`: evaluate based on human-annotated data
- Some scripts from booknlp multilingual for running NER model on entire dataset 
- `find_people.py`: to read in NER output, inspect glossary words, and create spreadsheet for manual annotation 
- `people_viz.ipynb`: for examining vocab
- `lexical_change.py`: for creating time series of words 
- `k_spectral_centroid.py`: for visualizing how words relate to waves of different communities 
- `time_series_plots.ipynb`: for examining time series for vocab
- `coref_forums.py`, `coref_reddit_control.py`, `coref_reddit.py`: running coref on different forum/Reddit datasets
- `coref_job_files.py`: creates job files for coref 
- `coref_helper.py`: analyzes coref output 
- `coref_viz.ipynb`: figuring out gender inference steps

### Building and validating semantic axes

- `setup_semantics.py`: finds occupation pages and creates WordNet axes
- `wikipedia_embeddings.py`: getting adjective and occupation embeddings from wikipedia 
- `axis_substitutes.py`: getting "good" contexts for adjectives in Wikipedia sentences.
- `validate_semantics.py`: functions for applying axes on occupation dataset
- `axes_occupation_viz.ipynb`: evaluate axes on occupation data

### Semantic differences and change 

- `prep_embedding_data.py`: prep data for getting embeddings 
- `reddit_forum_embeddings.py`: get embeddings for Reddit/forums
- `apply_semantics.py`: apply axes to Reddit and forum embeddings 
- `semantics_viz.ipynb`: visualizing semantics output 

### Deprecated
Some scripts were written to experiment with things but they will not be included in the paper. 
- `calc_npmi.py`: get embeddings using contexts with high NPMI (top contexts did not make sense)
- `community_users.py`: create user-based network among ideologies (resulted in ugly network)
- `singular_plural_matching.py`: singular plural matching, unused
- `explore_outliers.ipynb`: investigating a few long-post outliers in the dataset. This isn't needed because we are counting words once per comment/post
