import requests
import time
import gzip
import heapq
import csv
import numpy as np
import os
import pickle
from collections import defaultdict

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "WikiNavigationAI/1.0 (student project)"}

MAX_DEPTH = 6            # max path length before giving up

# SNAP dataset files - place in the same directory as this script
# download from: https://snap.stanford.edu/data/enwiki-2013.html
SNAP_GRAPH_FILE = "enwiki-2013.txt.gz"
SNAP_NAMES_FILE = "enwiki-2013-names.csv.gz"
SNAP_CACHE_FILE = "snap_cache.pkl"

# Wikipedia2Vec model file
# download from: http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2
WIKI2VEC_MODEL = "enwiki_20180420_100d.pkl"

# loaded once at startup
_graph = {}          # title -> list of neighbor titles
_reverse_graph = {}  # title -> list of incoming neighbour titles
_title_index = {}    # normalized title -> actual title in graph
_wiki2vec = None
_embedding_cache = {}
_max_degree = 1      # max incoming links across all pages, computed at load time


# --- data loading ---

def load_snap():
    """Load the SNAP Wikipedia graph into memory.
    Builds dicts for forward links, reverse links, and normalized title lookup.
    Saves a pickle cache so subsequent runs skip the ~1 min parse step.
    """
    global _graph, _reverse_graph, _title_index, _max_degree

    if os.path.exists(SNAP_CACHE_FILE):
        print(f"Loading cached SNAP graph from '{SNAP_CACHE_FILE}'...")
        try:
            with open(SNAP_CACHE_FILE, "rb") as f:
                data = pickle.load(f)
                _graph = data["graph"]
                _reverse_graph = data["reverse_graph"]
                _title_index = data["title_index"]
                _max_degree = max((len(v) for v in _reverse_graph.values()), default=1)
            return True
        except (OSError, pickle.PickleError, KeyError):
            print("  [warning] Cache load failed, rebuilding...")

    print("Loading SNAP Wikipedia graph...")
    print("  Reading page titles...")

    id_to_title = {}
    try:
        with gzip.open(SNAP_NAMES_FILE, "rt", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    node_id = int(row[0])
                except ValueError:
                    continue
                title = row[1].strip()
                if title:
                    id_to_title[node_id] = title
    except FileNotFoundError:
        print(f"  [error] '{SNAP_NAMES_FILE}' not found.")
        print(f"  Download from: https://snap.stanford.edu/data/enwiki-2013.html")
        return False

    print(f"  Loaded {len(id_to_title):,} page titles.")
    print("  Reading edges (this may take a minute)...")

    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    edge_count = 0
    try:
        with gzip.open(SNAP_GRAPH_FILE, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                try:
                    src = int(parts[0])
                    dst = int(parts[1])
                except ValueError:
                    continue

                if src in id_to_title and dst in id_to_title:
                    src_title = id_to_title[src]
                    dst_title = id_to_title[dst]

                    graph[src_title].append(dst_title)
                    reverse_graph[dst_title].append(src_title)
                    edge_count += 1
    except FileNotFoundError:
        print(f"  [error] '{SNAP_GRAPH_FILE}' not found.")
        print(f"  Download from: https://snap.stanford.edu/data/enwiki-2013.html")
        return False

    _graph = dict(graph)
    _reverse_graph = dict(reverse_graph)
    _max_degree = max((len(v) for v in _reverse_graph.values()), default=1)

    _title_index = {}
    for title in set(_graph.keys()) | set(_reverse_graph.keys()):
        key = normalize_title(title)
        if key not in _title_index:
            _title_index[key] = title

    print(f"  Loaded {edge_count:,} edges across {len(_graph):,} pages.")
    print(f"Saving graph cache to '{SNAP_CACHE_FILE}' (speeds up future runs)...")

    try:
        with open(SNAP_CACHE_FILE, "wb") as f:
            pickle.dump(
                {
                    "graph": _graph,
                    "reverse_graph": _reverse_graph,
                    "title_index": _title_index,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )
        print("  Cache saved.")
    except OSError:
        print("  [warning] Failed to save cache file.")

    return True


def load_wiki2vec():
    """Load the Wikipedia2Vec model. Falls back gracefully if not available."""
    global _wiki2vec, _embedding_cache
    _embedding_cache = {}
    try:
        from wikipedia2vec import Wikipedia2Vec
        print(f"Loading Wikipedia2Vec model from '{WIKI2VEC_MODEL}'...")
        _wiki2vec = Wikipedia2Vec.load(WIKI2VEC_MODEL)
        print("Model loaded.")
    except FileNotFoundError:
        print(f"  [warning] Wikipedia2Vec model not found at '{WIKI2VEC_MODEL}'.")
        print(f"  [warning] A* will use h=1 for all pages (uninformed).")
    except ImportError:
        print("  [warning] wikipedia2vec not installed. Run: pip install wikipedia2vec")
        print("  [warning] A* will use h=1 for all pages (uninformed).")


# --- graph access ---

def snap_title(title):
    # SNAP uses underscores instead of spaces
    return title.replace(" ", "_")


def normalize_title(title):
    # lowercase + underscores for case-insensitive lookup
    return snap_title(title.strip()).lower()


def snap_lookup(title):
    """Find the actual graph key for a given title, case-insensitive."""
    return _title_index.get(normalize_title(title))


def get_links(page_title):
    """Get outgoing links for a page from the SNAP graph."""
    return _graph.get(page_title, [])


def get_incoming_links(page_title):
    """Get incoming links for a page from the SNAP graph (used by bidirectional)."""
    return _reverse_graph.get(page_title, [])


def resolve_title(title):
    """Resolve user input to canonical Wikipedia title via the API.
    Only called twice at startup - not during search.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "redirects": 1,
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            if "title" in page and "-1" not in str(page.get("pageid", "")):
                return page["title"]
    except requests.RequestException:
        pass
    return title[:1].upper() + title[1:] if title else title


# --- heuristic ---

def _cosine_similarity(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return np.dot(v1, v2) / denom


def _get_embedding(page_title):
    """Look up a page's vector in the Wikipedia2Vec model.
    Tries both underscore and space formats since SNAP uses underscores
    but Wiki2Vec may use spaces.
    """
    if _wiki2vec is None:
        return None

    cached = _embedding_cache.get(page_title)
    if cached is not None or page_title in _embedding_cache:
        return cached

    vector = None
    try:
        vector = _wiki2vec.get_entity_vector(page_title)
    except KeyError:
        pass

    if vector is None:
        try:
            vector = _wiki2vec.get_entity_vector(page_title.replace("_", " "))
        except KeyError:
            pass

    if vector is None:
        try:
            vector = _wiki2vec.get_entity_vector(page_title.replace(" ", "_"))
        except KeyError:
            pass

    _embedding_cache[page_title] = vector
    return vector


def heuristic(page_title, goal_vector, h_cache):
    """Combined heuristic using Wikipedia2Vec cosine similarity and page degree.
    Semantic similarity guides toward topically related pages.
    Degree score biases toward well-connected hub pages that are reachable from anywhere.
    Not guaranteed admissible - semantic similarity doesn't reflect graph distance.
    Results are cached per A* run since the same page may be scored multiple times.
    """
    if page_title in h_cache:
        return h_cache[page_title]

    # semantic signal - how similar is this page to the goal
    if _wiki2vec is not None and goal_vector is not None:
        page_vector = _get_embedding(page_title)
        if page_vector is not None:
            similarity = _cosine_similarity(page_vector, goal_vector)
            semantic_score = 1.0 - ((similarity + 1) / 2)
        else:
            semantic_score = 1.0
    else:
        semantic_score = 1.0

    # degree signal - pages with high incoming links may act as navigational hubs
    degree = len(get_incoming_links(page_title))
    degree_score = 1.0 - (degree / _max_degree)

    score = 0.7 * semantic_score + 0.3 * degree_score
    h_cache[page_title] = score
    return score


def _reconstruct_path(parents, goal):
    # walk back through parent pointers to build the path
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parents[current]

    path.reverse()
    return path


# --- iddfs ---

def iddfs(start, goal):
    """Iterative Deepening DFS (Korf, 1985).
    Combines BFS optimality with DFS memory efficiency - only keeps current path in memory.
    Re-explores nodes on each iteration, but most nodes are at the deepest level anyway.
    """
    if start == goal:
        return [start], 0

    total_explored = 0

    for depth_limit in range(1, MAX_DEPTH + 1):
        print(f"  [IDDFS] Trying depth limit {depth_limit}...")
        path, explored, found = _dls(start, goal, depth_limit)
        total_explored += explored
        if found:
            return path, total_explored

    return None, total_explored


def _dls(start, goal, depth_limit):
    """Depth-limited DFS used internally by IDDFS.
    Uses an explicit stack to avoid Python's recursion limit.
    """
    stack = [(start, [start], 0)]
    pages_explored = 0

    while stack:
        current, path, depth = stack.pop()

        if current == goal:
            return path, pages_explored, True

        if depth >= depth_limit:
            continue

        pages_explored += 1
        links = get_links(current)

        for neighbor in reversed(links):  # reversed so we pop in original order
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor], depth + 1))

    return None, pages_explored, False


# --- a* ---

def astar(start, goal):
    """A* search using a combined Wikipedia2Vec + incoming degree heuristic h(n).
    Priority queue ordered by f = g + h(n).
    May not always find the shortest path since the heuristic is not admissible.
    """
    if start == goal:
        return [start], 0

    goal_vector = _get_embedding(goal) if _wiki2vec is not None else None
    if goal_vector is None:
        print(f"  [A*] '{goal}' not in Wiki2Vec model, using h=1 fallback.")
    else:
        print(f"  [A*] Goal vector loaded for '{goal}'.")

    h_cache = {}
    g_score = {start: 0}
    parents = {start: None}
    closed = set()
    progress_interval = 5000
    pages_explored = 0

    # heap entries: (f, g, page) - path reconstructed via parent pointers
    heap = [(heuristic(start, goal_vector, h_cache), 0, start)]

    while heap:
        f, g, current = heapq.heappop(heap)

        if current in closed:
            continue

        best_known_g = g_score.get(current)
        if best_known_g is None or g > best_known_g:
            continue

        closed.add(current)

        if current == goal:
            return _reconstruct_path(parents, goal), pages_explored

        if g >= MAX_DEPTH:
            continue

        links = get_links(current)
        pages_explored += 1

        if pages_explored % progress_interval == 0:
            print(f"  [A*] Explored {pages_explored} pages so far...")

        for neighbor in links:
            if neighbor in closed:
                continue

            new_g = g + 1
            if new_g > MAX_DEPTH:
                continue

            old_g = g_score.get(neighbor)
            if old_g is not None and new_g >= old_g:
                continue

            g_score[neighbor] = new_g
            parents[neighbor] = current
            new_f = new_g + heuristic(neighbor, goal_vector, h_cache)
            heapq.heappush(heap, (new_f, new_g, neighbor))

    return None, pages_explored


# --- bidirectional bfs ---

def bidirectional(start, goal):
    """Bidirectional BFS - searches from both start and goal simultaneously.
    Reduces search space from O(b^d) to O(b^(d/2)) by meeting in the middle.
    Always expands the smaller frontier.
    Backward pass uses the reverse graph to follow incoming links.
    """
    if start == goal:
        return [start], 0

    pages_explored = 0

    fwd_visited = {start: [start]}  # page -> path from start
    bwd_visited = {goal: [goal]}    # page -> path from goal

    fwd_frontier = {start}
    bwd_frontier = {goal}

    for depth in range(1, MAX_DEPTH + 1):
        if len(fwd_frontier) <= len(bwd_frontier):
            next_frontier = set()
            print(f"  [BiDir] Forward expanding {len(fwd_frontier)} pages at depth {depth}...")
            for page in fwd_frontier:
                links = get_links(page)
                pages_explored += 1
                for neighbor in links:
                    if neighbor not in fwd_visited:
                        fwd_visited[neighbor] = fwd_visited[page] + [neighbor]
                        next_frontier.add(neighbor)
                    if neighbor in bwd_visited:
                        # paths meet - stitch them together
                        full_path = fwd_visited[neighbor] + list(reversed(bwd_visited[neighbor]))[1:]
                        return full_path, pages_explored
            fwd_frontier = next_frontier
        else:
            next_frontier = set()
            print(f"  [BiDir] Backward expanding {len(bwd_frontier)} pages at depth {depth}...")
            for page in bwd_frontier:
                links = get_incoming_links(page)
                pages_explored += 1
                for neighbor in links:
                    if neighbor not in bwd_visited:
                        bwd_visited[neighbor] = bwd_visited[page] + [neighbor]
                        next_frontier.add(neighbor)
                    if neighbor in fwd_visited:
                        full_path = fwd_visited[neighbor] + list(reversed(bwd_visited[neighbor]))[1:]
                        return full_path, pages_explored
            bwd_frontier = next_frontier

        if not fwd_frontier and not bwd_frontier:
            break

    return None, pages_explored


# --- output ---

def print_section(title):
    print(f"\n{'=' * 52}")
    print(f"  {title}")
    print(f"{'=' * 52}")


def print_result(label, path, pages_explored, elapsed):
    print_section(label)
    if not path:
        print("  No path found within depth limit.")
        print(f"  Pages explored : {pages_explored}")
        print(f"  Time taken     : {elapsed:.2f}s")
        return

    print(f"  Path length : {len(path) - 1} link(s)")
    print(f"  Explored    : {pages_explored}")
    print(f"  Time        : {elapsed:.2f}s")
    print("\n  Path:")

    for i, page in enumerate(path):
        if i == 0:
            print(f"    START  {page}")
        elif i == len(path) - 1:
            print(f"    GOAL   {page}")
        else:
            print(f"      {i:<2}   {page}")


def print_comparison(results):
    # IDDFS is the baseline since it guarantees the shortest path
    print_section("Comparison Summary")

    col1 = 30
    col2 = 12
    col3 = 12
    col4 = 16

    print(f"{'Metric':<{col1}}{'IDDFS':>{col2}}{'A*':>{col3}}{'Bidirectional':>{col4}}")
    print("-" * (col1 + col2 + col3 + col4))

    lengths = [str(len(r[1]) - 1) if r[1] else "N/A" for r in results]
    explored = [str(r[2]) for r in results]

    print(f"{'Path length (links)':<{col1}}{lengths[0]:>{col2}}{lengths[1]:>{col3}}{lengths[2]:>{col4}}")
    print(f"{'Pages explored':<{col1}}{explored[0]:>{col2}}{explored[1]:>{col3}}{explored[2]:>{col4}}")

    baseline_path = results[0][1]
    baseline_explored = results[0][2]

    if baseline_explored and baseline_explored > 0:
        print("\n  vs IDDFS baseline:")
        for label, path, explored_n, _ in results[1:]:
            if explored_n is not None:
                diff = (1 - explored_n / baseline_explored) * 100
                direction = "fewer" if diff > 0 else "more"
                print(f"    {label:<16} {abs(diff):.1f}% {direction} pages explored")

    if baseline_path:
        print("\n  Optimality vs IDDFS:")
        for label, path, _, _ in results[1:]:
            if path:
                same = len(path) == len(baseline_path)
                text = "same length" if same else f"{len(path)-1} links vs {len(baseline_path)-1}"
                print(f"    {label:<16} {text}")
            else:
                print(f"    {label:<16} no path found")


# --- main ---

def run_search():
    start_input = input("\nStart page : ").strip()
    target_input = input("Target page: ").strip()

    if not start_input or not target_input:
        print("\nStart and target page cannot be empty.")
        return

    # try SNAP lookup on raw input first, only call API if that fails
    start_snap = snap_lookup(start_input)
    if start_snap is None:
        start_page = resolve_title(start_input)
        start_snap = snap_lookup(start_page)
    else:
        start_page = start_input

    target_snap = snap_lookup(target_input)
    if target_snap is None:
        target_page = resolve_title(target_input)
        target_snap = snap_lookup(target_page)
    else:
        target_page = target_input

    if start_snap is None:
        print(f"  [warning] '{start_page}' not found in SNAP graph.")
    else:
        print(f"  Resolved title: '{start_page}' -> '{start_snap}'")

    if target_snap is None:
        print(f"  [warning] '{target_page}' not found in SNAP graph.")
    else:
        print(f"  Resolved title: '{target_page}' -> '{target_snap}'")

    print(f"\nSearching: '{start_snap}' -> '{target_snap}'")

    if start_snap is None or target_snap is None:
        print("\nSearch cancelled. Both pages must resolve to valid SNAP graph titles before running algorithms.")
        return

    results = []

    print("\n--- Running IDDFS ---")
    t0 = time.time()
    iddfs_path, iddfs_explored = iddfs(start_snap, target_snap)
    iddfs_time = time.time() - t0
    results.append(("IDDFS", iddfs_path, iddfs_explored, iddfs_time))

    print("\n--- Running A* ---")
    t0 = time.time()
    astar_path, astar_explored = astar(start_snap, target_snap)
    astar_time = time.time() - t0
    results.append(("A*", astar_path, astar_explored, astar_time))

    print("\n--- Running Bidirectional BFS ---")
    t0 = time.time()
    bidir_path, bidir_explored = bidirectional(start_snap, target_snap)
    bidir_time = time.time() - t0
    results.append(("Bidirectional", bidir_path, bidir_explored, bidir_time))

    for label, path, pages_explored, elapsed in results:
        display_label = "Bidirectional BFS" if label == "Bidirectional" else label
        print_result(display_label, path, pages_explored, elapsed)

    print_comparison(results)


def main():
    load_wiki2vec()
    snap_ok = load_snap()
    if not snap_ok:
        print("\nCannot run without SNAP dataset. Exiting.")
        return

    print("\nWikipedia Navigation AI")
    print("-----------------------")
    print(f"Graph              : SNAP enwiki-2013 ({len(_graph):,} pages)")
    print(f"Max depth          : {MAX_DEPTH}")
    print(f"Heuristic          : {'combined (semantic + degree)' if _wiki2vec else 'degree only (no embeddings)'}")

    while True:
        run_search()
        again = input("\nRun another search? (y/n): ").strip().lower()
        if again not in {"y", "yes"}:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
