from abc import abstractmethod, ABC
from collections import deque
import random
import numpy as np
import math
import heapq

class Vertex:
    def __init__(self, u):
        self.value = u
    
    def __eq__(self, vertex):
        return self.value == vertex.value

    def __hash__(self):
        return self.value

class Edge:
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
    def add_vertex(self, v: Vertex):
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

    def add_vertex(self, v: Vertex):
        pass
    
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

    def weight(self, u, v):
        for edge in self.edges:
            if (u, v) == edge.endpoints():
                return edge.weight

    def edge_count(self):
        return len(self.edges)

    def out_degree(self, v, out = True):
        return self.adj_struct.out_degree(Vertex(v))
        # return self.adj_mat.out_degree(v)

    def in_degree(self, v, inc = True):
        return self.adj_struct.out_degree(Vertex(v))
        # return self.adj_mat.out_degree(v)

    def insert_vertex(self, v):
        self.vertices.append(Vertex(v))
        self.adj_struct.add_vertex(Vertex(v))

    def insert_edge(self, u, v, x = None):
        e = Edge(u, v, x)
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

def bfs(G : Graph, v, visit):
    q = deque()
    visited = [False] * G.vertex_count()
    q.append(v)

    while len(q):
        vertex = q.popleft()
        visited[G.vertices.index(Vertex(vertex))] = True
        visit(v, vertex)

        for u in G.neighbors(vertex):
            if not visited[G.vertices.index(u)]:
                q.append(u.value)

def dfs(G : Graph, v,  visit):
    q = deque()
    visited = [False] * G.vertex_count()
    q.append(v)

    while len(q):
        vertex = q.popleft()
        visited[G.vertices.index(Vertex(vertex))] = True
        visit(v, vertex)

        for u in G.neighbors(vertex):
            if not visited[G.vertices.index(u)]:
                q.appendleft(u.value)

def forests(G : Graph):
    def visit(node, vertex):
        connected_components[G.vertices.index(Vertex(node))].append(vertex)

    connected_components = [[]] * G.vertex_count()

    for v in G.vertices:
        dfs(G, v.value, visit)
    return connected_components

def floyd_warshall(G):
    pass

def topological_ordering(G: Graph):
    order, in_count, ready = [], dict(), deque()

    for v in G.vertices:
        in_count[v] = G.in_degree(v)
        if in_count == 0:   ready.append(v)

    while len(ready) > 0:
        u = ready.popleft()
        order.append(u)

        for child in G.neighbors(u):
            in_count[child] -= 1
            if in_count[child] == 0:    ready.append(child)
    return order

def bellman_ford(G):
    pass

def all_source_shortest_path(G : Graph, u, dist):
    d = {v: np.inf for v in G.vertices}
    d[Vertex(u)] = 0

    queue = [], visited = dict()
    heapq.heappush(queue, (d[Vertex(u)], Vertex(u)))

    while len(queue):
        d[Vertex(vertex)], vertex = heapq.heappop(queue)
        visited[vertex] = True
        
        for child in G.neighbors(vertex):
            if d[vertex] + dist(vertex, child) < d[child]:
                d[child] = d[vertex] + dist(vertex, child)
                heapq.heappush((d[child], child))
    return d

def djikstra(G : Graph, u):
    dist = lambda u, v: G.weight(u, v)
    return all_source_shortest_path(G, u, dist)

def a_star(G, u, v):
    heuristic = lambda u, v: np.abs(u - v)
    dist = lambda u, v: G.weight(u, v) + heuristic(u, v)
    return all_source_shortest_path(G, u, dist)
