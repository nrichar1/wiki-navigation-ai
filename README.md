# Wikipedia Navigation AI

This project explores search algorithms for solving the Wikipedia Navigation Game, commonly known as Wikiracing.

In Wikiracing, a player starts on one Wikipedia article and attempts to reach another article by following hyperlinks between pages. Each article can be treated as a node in a graph and each hyperlink represents a connection between nodes.

The goal of this project is to model the problem as a graph search task and compare algorithms for navigating between pages.

The project will implement and compare:

- Breadth First Search (BFS)
- A* search with a heuristic

Wikipedia page links will be retrieved dynamically using the MediaWiki API.
