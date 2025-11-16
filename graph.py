from abc import abstractmethod, ABC
from collections import deque
import random
import numpy as np
import math
from union_find import UnionSet
import heapq

class Vertex:
    """
        Defines a Vertex of a Graph
    """
    def __init__(self, u):
        if not isinstance(u, Vertex):
            self.value = u
    
    def __eq__(self, vertex):
        return self.value == vertex.value

    def __ne__(self, vertex):
        return self.value != vertex.value

    def __hash__(self):
        return self.value

class Edge:
    """
        Defines an edge of a Graph
    """

    def __init__(self, u, v, w):
        if not isinstance(u, Vertex):
            u = Vertex(u)
        if not isinstance(v, Vertex):
            v = Vertex(v)
        
        self.point1, self.point2 = u, v
        self.weight = w

    def endpoints(self):
        return (self.point1, self.point2)
    
    def opposite(self, v: Vertex):
        """
            Returns the other vertex of an edge from one vertex
        """
        if not isinstance(v, Vertex):
            v = Vertex(v)
        
        if not self.contains(v):
            raise Exception("Vertex not connected by the edge")
        
        return self.point1 if v is self.point2 else self.point2
    
    def contains(self, v : Vertex):
        """
            Returns whether the edge contains the edge
        """

        if not isinstance(v, Vertex):
            v = Vertex(v)
        
        return v in (self.point1, self.point2)

class DirectedEdge(Edge):
    """
        To declare an edge as directed
    """
    pass

class UndirectedEdge(Edge):
    """
        To declare an edge as undirected
    """
    pass

class AdjacencyStructure(ABC):
    """
        Interface defining the layout of an adjacency structure of a graph
    """

    @abstractmethod
    def add_edge(self, e: Edge):
        pass
    
    @abstractmethod
    def remove_edge(self, e: Edge):
        pass
    
    @abstractmethod
    def add_vertex(self, v: Vertex):
        pass

    @abstractmethod
    def remove_vertex(self, v : Vertex):
        pass

    @abstractmethod
    def neighbors(self, v: Vertex):
        pass

    @abstractmethod
    def out_degree(self, v: Vertex):
        pass

    @abstractmethod
    def in_degree(self, v: Vertex):
        pass

class AdjacencyList(AdjacencyStructure):
    """
        Implements Adjacency List for a Graph
    """

    def __init__(self):
        self.map = dict()
    
    def add_edge(self, e: Edge):
        u, v = e.endpoints()
        if u not in self.map:   self.map[u] = [v]
        else:   self.map[u].append(v)

        if isinstance(e, UndirectedEdge):
            if v not in self.map:   self.map[v] = [u]
            else:   self.map[v].append(u)
    
    def remove_edge(self, e: Edge):
        u, v = e.endpoints()
        if u not in self.map:   raise ValueError("Edge doesn't exist")
        else:   self.map[u].remove(v)

        if isinstance(e, UndirectedEdge):
            if v not in self.map:   raise ValueError("Edge doesn't exist")
        else:   self.map[v].remove(u)
    
    def add_vertex(self, v: Vertex):
        if not isinstance(v, Vertex):   v = Vertex(v)
        self.map[v] = []
    
    def remove_vertex(self, v : Vertex):
        if v not in self.map:   raise ValueError("Vertex doesn't exist")
        del self.map[v]
    
    def neighbors(self, v: Vertex):
        if v not in self.map:   raise ValueError("Vertex doesn't exist")
        else:   return self.map[v]
    
    def out_degree(self, v):
        if v not in self.map:   raise ValueError("Vertex doesn't exist")
        else:   return len(self.map[v])

    def in_degree(self, v):
        if v not in self.map:   raise ValueError("Vertex doesn't exist")
        else:   return len(1 for u in self.map if v in self.map[u])

class AdjacencyMatrix(AdjacencyStructure):
    """
        Implements Adjacency Matrix for a Graph
    """

    def __init__(self):
        self.mat = np.array([])
        self.vertices = []
    
    def add_edge(self, e: Edge):
        u, v = e.endpoints()
        if u not in self.vertices:
            self.vertices.append(u)
            self.mat = np.append(self.mat, np.zeros((len(self.vertices), 1)), axis=1)
            self.mat = np.append(self.mat, np.zeros((len(self.vertices), 1)), axis=0)
        if v not in self.vertices:
            self.vertices.append(v)
            self.mat = np.append(self.mat, np.zeros((len(self.vertices), 1)), axis=1)
            self.mat = np.append(self.mat, np.zeros((len(self.vertices), 1)), axis=0)
        
        self.mat[self.vertices.index(u), self.vertices.index(v)] = 1

        if isinstance(e, UndirectedEdge):
            self.mat[self.vertices.index(v), self.vertices.index(u)] = 1
    
    def add_vertex(self, v: Vertex):
        self.vertices.append(v)
        self.mat = np.append(self.mat, np.zeros((len(self.vertices), 1)), axis=1)
        self.mat = np.append(self.mat, np.zeros((len(self.vertices), 1)), axis=0)
    
    def remove_edge(self, e : Edge):
        u, v = e.endpoints()
        if u not in self.vertices or v not in self.vertices: raise Exception("Edge not present")

        self.mat[self.vertices.index(u), self.vertices.index(v)] = 0

        if isinstance(e, UndirectedEdge):
            self.mat[self.vertices.index(v), self.vertices.index(u)] = 0
    
    def remove_vertex(self, v : Vertex):
        if v not in self.vertices: raise Exception("Vertex doesn't exist")

        self.mat = np.delete(self.mat, self.vertices.index(v), axis=0)
        self.mat = np.delete(self.mat, self.vertices.index(v), axis=1)

    def neighbors(self, v : Vertex):
        if v not in self.vertices: raise Exception("Vertex doesn't exist")
        return [self.vertices[i] for i in self.mat.where(self.mat[self.vertices.index(v), :] == 1)]
    
    def out_degree(self, v):
        if v not in self.vertices: raise Exception("Vertex doesn't exist")
        return len([self.vertices[i] for i in self.mat.where(self.mat[self.vertices.index(v), :] == 1)])

    def in_degree(self, v):
        if v not in self.vertices: raise Exception("Vertex doesn't exist")
        return len([self.vertices[i] for i in self.mat.where(self.mat[:, self.vertices.index(v)] == 1)])

class Graph:
    def __init__(self):
        self.adj_struct : AdjacencyStructure = AdjacencyList()
        self.vertices = []
        self.edges = []

    def vertex_count(self):
        return len(self.vertices)

    def weight(self, u, v):
        """
            Returns the edge weight of the edge between two vertices
        """

        for edge in self.edges:
            if (u, v) == edge.endpoints():
                return edge.weight

    def edge_count(self):
        return len(self.edges)

    def out_degree(self, v, out = True):
        return self.adj_struct.out_degree(Vertex(v))

    def in_degree(self, v, inc = True):
        return self.adj_struct.out_degree(Vertex(v))

    def insert_vertex(self, v):
        self.vertices.append(Vertex(v))
        self.adj_struct.add_vertex(Vertex(v))

    def insert_edge(self, u, v, w = None):
        e = Edge(u, v, w)
        self.edges.append(e)
        self.adj_struct.add_edge(e)

    def delete_vertex(self, v):
        self.vertices.remove(Vertex(v))
        self.adj_struct.remove_vertex(Vertex(v))

    def delete_edge(self, e: Edge):
        self.edges.remove(e)
        self.adj_struct.remove_edge(e)
    
    def neighbors(self, v):
        if not isinstance(v, Vertex):   return self.adj_struct.neighbors(Vertex(v))
        else:   return self.adj_struct.neighbors(v)

def bellman_ford(G : Graph, u, dist):
    """
        ASSP algo that handles detection of negative weight cycles
    """

    d = {Vertex(v): np.inf for v in G.vertices}  # approx dist
    d[Vertex(u)] = 0
    prec = {Vertex(v): None for v in G.vertices}  # predecessor of each node

    vertex_count = G.vertex_count()
    
    for _ in range(vertex_count - 1):  # limiting condition : iterate through all possible edges
        for e in G.edges:
            u, v, w = e.endpoints(), e.weight
            u, v = Vertex(u), Vertex(v)

            if d[u] + w < d[v]:  # relax / update the overestimation of the approx dist
                d[v] = d[u] + w
                prec[v] = u
    else:  # check for negative cycles
        for e in G.edges:
            u, v, w = e.endpoints(), e.weight
            u, v = Vertex(u), Vertex(v)

            if d[u] + w < d[v]:  # this is only possible if a negative edge exists
                prec[v] = u
                
                # check for a cycle
                visited = {Vertex(v):False for v in G.vertices()}
                visited[v] = True
                while not visited[u]:
                    visited[u] = True
                    u = prec[u]
                
                ncycle = [u]
                v = prec[u]
                while v != u:
                    ncycle.append(v)
                    v = prec[v]
                raise Exception("Graph has negative cycle")
    return d

def bfs(G : Graph, v, visit):
    """
        Graph traversal / ASSP algo with no edge weights
    """

    q = deque()
    visited = [False] * G.vertex_count()
    q.append(v)  # first  vertex is added to boundary

    while len(q):
        vertex = q.popleft()  # get the current vertex of the boundary
        visited[G.vertices.index(Vertex(vertex))] = True  # add it to the pack of visited
        visit(v, vertex)

        for u in G.neighbors(vertex):  # expand on the childern
            if not visited[G.vertices.index(u)]:  # add new nodes to the boundary
                q.append(u.value)  # ensures this node comes after all of the previous level nodes

def dfs(G : Graph, v,  visit):
    """
        Graph traversal / ASSP algo with no edge weights
    """

    q = deque()  # boundary
    visited = [False] * G.vertex_count()  # visited region
    q.append(v)

    while len(q):
        vertex = q.popleft()  # get a node from the boundary
        visited[G.vertices.index(Vertex(vertex))] = True  # add it to the visited region
        visit(v, vertex)

        for u in G.neighbors(vertex):  # for all unexplored childern of the node, add it to the boundary
            if not visited[G.vertices.index(u)]:
                q.appendleft(u.value)  # ensures this node is opened first

def forests(G : Graph):
    """
        Returns all connected components of a graph
        Connected component is a subgraph where you can reach a node from any other node in it
        A collection of connected components form a forest of trees
    """

    def visit(node, vertex):
        connected_components[G.vertices.index(Vertex(node))].append(vertex)

    connected_components = [[]] * G.vertex_count()

    for v in G.vertices:
        dfs(G, v.value, visit)
    return connected_components

def floyd_warshall(G : Graph):
    """
        APSP (All Pairs Shortest Path) algorithm
        works with graphs with both positive and negative edges
        not to be used with negative cycles
    """
    
    # initialise all pair shortest paths
    d = {(Vertex(u), Vertex(v)): np.inf for u in G.vertices for v in G.vertices}

    for e in G.edges:
        u, v, w = e.endpoints(), e.weight
        u, v = Vertex(u), Vertex(v)
        d[u, v] = w  # each edge vertex has its weight as shortest path lengths
    
    # Either do nesting of K - I - J or 3 times I K J / I J K
    for k in range(0, G.vertex_count()):
        for u in G.vertices:
            for v in G.vertices:  # check if adding an vertex in between helps
                d[u, v] = min(d[u, v], d[u, Vertex(k)] + d[Vertex(k), v])
    return d

def topological_ordering(G: Graph):
    """
        Returns a particular order of a DAG based on their in_order priority
        This order indicates an order / flow / sequence within a graph
    """

    order, in_count, ready = [], dict(), deque()

    for v in G.vertices:
        in_count[v] = G.in_degree(v)  # get the in_degree count of each vertex (unexplored)
        if in_count == 0:   ready.append(v)  # if a vertex has no parents, it is next in line (frontier)

    while len(ready) > 0:
        u = ready.popleft()  # get a vertex with no parents to add to the ordering (visited)
        order.append(u)

        for child in G.neighbors(u):  # prune the vertex from the graph and decrease its childern's in_degree
            in_count[child] -= 1
            if in_count[child] == 0:    ready.append(child)
    return order

def all_source_shortest_path(G : Graph, u, dist):
    d = {v: np.inf for v in G.vertices}  # approximated distance from the source
    d[Vertex(u)] = 0  # dist of vertex to itself is zero

    queue = [], visited = dict()  # frontier and explored
    heapq.heappush(queue, (d[Vertex(u)], Vertex(u)))

    while len(queue):
        d[Vertex(vertex)], vertex = heapq.heappop(queue)  # get a vertex out of the frontier
        visited[vertex] = True  # add it to explored
        
        for child in G.neighbors(vertex):  # iterate through its childern
            if d[vertex] + dist(vertex, child) < d[child]:  # the approx dist must overestimate the real dist
                d[child] = d[vertex] + dist(vertex, child)  # correct the approx dist to actual dist
                heapq.heappush((d[child], child))  # add the child to frontier
    return d

def djikstra(G : Graph, u):
    """
        ASSP algo as a simple BFS extension for weighted graphs
        Cannot handle negative weights
    """

    dist = lambda u, v: G.weight(u, v)
    return all_source_shortest_path(G, u, dist)

def a_star(G : Graph, u, v):
    heuristic = lambda u, v: np.abs(u - v)
    dist = lambda u, v: G.weight(u, v) + heuristic(u, v)
    return all_source_shortest_path(G, u, dist)

def prims_mst(G : Graph):
    """
        Greedy MST Construction Algorithm based on the cut property
    """

    mst = Graph()  # mst
    visited = {Vertex(v) : False for v in G.vertices}  # explored
    vertices = []
    cut_set = []  # frontier edges

    # initialise a random vertex
    start_vertex = G.vertices[np.random.choice(len(G.vertices), size = 1)]
    mst.insert_vertex(start_vertex)
    visited[start_vertex] = True
    vertices.append(start_vertex)
    cut_set += [e for e in G.edges if e.point1 == start_vertex]  # consider all out going edges from start
    
    while not all(visited.values()):
        min_edge = cut_set[np.argmin(np.array([e.weight for e in cut_set]))]
        u, v, weight = min_edge.endpoints(), min_edge.weight
        mst.insert_edge(u, v, weight)
        vertices.append(v)
        visited[vertex] = True
        other_edges = [e for e in G.edges if e.point2 == v]

        # update cut_set
        for t in other_edges:  # all internal edges in the explored
            cut_set.remove(t)
        cut_set += [e for e in G.edges if e.point1 == v]  # update new frontier
    
    return mst

def kruskal_mst(G : Graph):
    """
        Greedy MST construction algorithm
    """

    mst = Graph()
    us = UnionSet(len(G.vertices))  # union find for preventing cycles in the tree

    edges = [(e, e.weight) for e in G.edges]  # sort the edges according to weights
    edges.sort(key=lambda x: x[1])

    for e in edges:
        u, v, w = e.endpoints(), e.weight
        if us.find(u) != us.find(v):  # if the two vertices belong in different components
            mst.insert_edge(u, v, w)  # add them to the mst
            us.union(us.find(u), us.find(v))
    return mst

class Trie(Graph):
    pass
