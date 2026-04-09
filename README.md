# Wikipedia Navigation AI

This project explores and compares three search algorithms on the Wikipedia navigation problem, commonly known as Wikiracing. The goal is to navigate from one Wikipedia article to another by following hyperlinks, finding the shortest path between two pages.

The three algorithms implemented are IDDFS (Iterative Deepening DFS), A* search with a Wikipedia2Vec embedding heuristic, and Bidirectional BFS.

The Wikipedia link graph is loaded from the SNAP enwiki-2013 dataset. A* uses pretrained Wikipedia2Vec embeddings combined with incoming link degree to estimate distance between pages. Neither require API calls during search.

## Setup

```
pip install wikipedia2vec numpy requests
```

Download the following files and place them in the `code/` directory:

**Wikipedia link graph (SNAP enwiki-2013):**
```
https://snap.stanford.edu/data/enwiki-2013.txt.gz
https://snap.stanford.edu/data/enwiki-2013-names.csv.gz
```

**Wikipedia2Vec model:**
```
http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2
```

Extract the `.bz2` file to get `enwiki_20180420_100d.pkl` and place it in `code/`.

On first run the graph is parsed and cached locally as `snap_cache.pkl`. Subsequent runs load from the cache and start in a few seconds instead of a minute.

## Usage

```
cd code
python main.py
```

## Course

CS4795 - Artificial Intelligence, University of New Brunswick
