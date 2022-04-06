# A Diachronic Typology of People in the Manosphere and Related Online Communities

## Code

Each script should be thoroughly commented.

### Dataset

### Vocabulary

- `coref_forums.py`: running coref on forum data

### Building semantic axes

- `axis_substitutes.py`: getting BERT substitutes for adjectives in Wikipedia sentences.

### Validating the axes 

- `axes_occupation_viz.ipynb`: evaluate axes on occupation data

### Semantic differences and change 

- `apply_semantics.py`: Apply axes to Reddit and forum embeddings 

### Deprecated
Some scripts were written to experiment with things but they will not be included in the paper. 
- `calc_npmi.py`: get embeddings using contexts with high NPMI (top contexts did not make sense)
- `community_users.py`: create user-based network among ideologies (resulted in ugly network)
