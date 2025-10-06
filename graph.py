from abc import abstractmethod, ABC
from collections import deque
import random
import numpy as np
import math

class Vertex:
    def __init__(self, u):
        self.value = u

class Edge:
    def __init__(self, u : Vertex, v: Vertex):
        if not isinstance(u, Vertex):
            u = Vertex(u)
        if not isinstance(v, Vertex):
            v = Vertex(v)
        
        self.point1, self.point2 = u, v

    def endpoints(self):
        return (self.point1, self.point2)
    
    def opposite(self, v: Vertex):
        if not isinstance(v, Vertex):
            v = Vertex(v)
        
        return self.point1 if v is self.point2 else self.point2

class DirectedEdge(Edge):
    pass

class UndirectedEdge(Edge):
    pass

class AdjacencyStructure(ABC):
    @abstractmethod
    def add_edge(self, e: Edge):
        pass
    
    @abstractmethod
    def remove_edge(self, e: Edge):
        pass
    
    @abstractmethod
    def remove_vertex(self, v : Vertex):
        pass

    @abstractmethod
    def neighbors(self, v: Vertex):
        pass

    @abstractmethod
    def out_degree(self, v: Vertex, out = True):
        pass

    @abstractmethod
    def in_degree(self, v: Vertex, inc = True):
        pass

class AdjacencyList(AdjacencyStructure):
    def __init__(self):
        self.map = dict()
    
    def add_edge(self, e: Edge):
        u, v = e
        if u not in self.map:   self.map[u] = [v]
        else:   self.map[u].append(v)

        if isinstance(e, UndirectedEdge):
            if v not in self.map:   self.map[v] = [u]
            else:   self.map[v].append(u)
    
    def remove_edge(self, e: Edge):
        u, v = e
        if u not in self.map:   raise ValueError("Edge doesn't exist")
        else:   self.map[u].remove(v)

        if isinstance(e, UndirectedEdge):
            if v not in self.map:   raise ValueError("Edge doesn't exist")
        else:   self.map[v].remove(u)
    
    def remove_vertex(self, v : Vertex):
        if v not in self.map:   raise ValueError("Vertex doesn't exist")

        del self.map[v]
    
    def neighbors(self, v: Vertex):
        if v not in self.map:   raise ValueError("Vertex doesn't exist")
        else:   return self.map[v]
    
    def out_degree(self, v, out=True):
        if v not in self.map:   raise ValueError("Vertex doesn't exist")
        else:   return len(self.map[v])

    def in_degree(self, v, inc=True):
        if v not in self.map:   raise ValueError("Vertex doesn't exist")
        else:   return len(1 for u in self.map if v in self.map[u])

class AdjacencyMatrix(AdjacencyStructure):
    def __init__(self):
        self.mat = np.array([])
        self.vertices = []
    
    def add_edge(self, e: Edge):
        u, v = e
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
    
    def remove_edge(self, e : Edge):
        u, v = e
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
    
    def out_degree(self, v, out=True):
        if v not in self.vertices: raise Exception("Vertex doesn't exist")

        return len([self.vertices[i] for i in self.mat.where(self.mat[self.vertices.index(v), :] == 1)])

    def in_degree(self, v, inc=True):
        if v not in self.vertices: raise Exception("Vertex doesn't exist")

        return len([self.vertices[i] for i in self.mat.where(self.mat[:, self.vertices.index(v)] == 1)])

class Graph:
    def __init__(self):
        self.adj_struct : AdjacencyStructure = AdjacencyList()
        self.vertices = []
        self.edges = []

    def vertex_count(self):
        return len(self.vertices)

    def edge_count(self):
        return len(self.edges)

    def out_degree(self, v: Vertex, out = True):
        return self.adj_struct.out_degree(v)
        # return self.adj_mat.out_degree(v)

    def in_degree(self, v: Vertex, inc = True):
        return self.adj_struct.out_degree(v)
        # return self.adj_mat.out_degree(v)

    def insert_vertex(self, v : Vertex = None):
        self.vertices.append(v)

    def insert_edge(self, u : Vertex, v : Vertex, x = None):
        e = Edge(u, v)
        self.edges.append(e)
        self.adj_struct.add_edge(e)

    def delete_vertex(self, v: Vertex):
        self.vertices.remove(v)
        self.adj_struct.remove_vertex(v)

    def delete_edge(self, e: Edge):
        self.edges.remove(e)
        self.adj_struct.remove_edge(e)
    
    def neighbors(self, v : Vertex):
        return self.adj_struct.neighbors(v)

def bfs(G : Graph, u, visit):
    q = deque()
    visited = [False] * len(G.vertex_count())
    q.append(u)

    while len(q):
        vertex = q.popleft()
        visited[vertex] = True
        visit(vertex)

        for u in G.neighbors(vertex):
            if not visited[u]:
                q.append(u)

def dfs(G : Graph, u,  visit):
    q = deque()
    visited = [False] * len(G.vertex_count())
    q.append(u)

    while len(q):
        vertex = q.popleft()
        visited[vertex] = True
        visit(vertex)

        for u in G.neighbors(vertex):
            if not visited[u]:
                q.appendleft(u)

def forests(G : Graph):
    def visit(vertex):
        connected_components[G.vertices.index[vertex]].append(vertex)

    connected_components = [[] * len(G.vertices)]

    for v in G.vertices:
        dfs(G, v, visit)
    return connected_components

def floyd_warshall(G):
    pass

def topological_ordering(G):
    pass

def bellman_ford(G):
    pass
