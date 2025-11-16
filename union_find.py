class UnionSet:
    def __init__(self, size):
        self.reprs = list(range(size))
        self.rank = [0] * size
    
    def find(self, i):
        if self.reprs[i] == i:
            return i
        return self.find(self.reprs[i])
    
    def union(self, i, j):
        irep = self.find(i)
        jrep = self.find(j)

        if irep != jrep:
            if self.rank[irep] < self.rank[jrep]:
                self.reprs[irep] = jrep
            elif self.rank[irep] > self.rank[jrep]:
                self.reprs[jrep] = irep
            else:
                self.reprs[jrep] = irep
                self.rank[irep] += 1

    def connected(self, i, j):
        irep = self.find(i)
        jrep = self.find(j)
        return irep == jrep
