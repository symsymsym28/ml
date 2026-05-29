class UnionFind:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, i):
        if self.parent[i] == i:
            return i
        return self.find(self.parent[i])

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

def get_mst(nodes, edges):
    sorted_edges = sorted(edges, key=lambda x: x[2])
    uf = UnionFind(nodes)
    mst_edges = []
    for u, v, weight in sorted_edges:
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
    return mst_edges

def identify_clusters(nodes, mst_edges):
    adj = {node: [] for node in nodes}
    for u, v, w in mst_edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    final_edges = []
    for u, v, weight in mst_edges:
        adjacent_weights = [w for neighbor, w in adj[u] if neighbor != v] + \
                           [w for neighbor, w in adj[v] if neighbor != u]
        
        if adjacent_weights:
            avg_adjacent = sum(adjacent_weights) / len(adjacent_weights)
            if weight > avg_adjacent:
                continue 
        final_edges.append((u, v))

    remaining_adj = {node: [] for node in nodes}
    for u, v in final_edges:
        remaining_adj[u].append(v)
        remaining_adj[v].append(u)

    visited = set()
    clusters = []
    for node in nodes:
        if node not in visited:
            component = []
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    component.append(curr)
                    stack.extend(remaining_adj[curr])
            clusters.append(component)
    return clusters

nodes_list = ['a', 'b', 'c', 'd', 'e', 'f']
edges_list = [('a', 'b', 1), ('b', 'c', 2), ('c', 'd', 6), ('d', 'e', 3), ('e', 'f', 1), ('a', 'c', 4)]

mst = get_mst(nodes_list, edges_list)
result_clusters = identify_clusters(nodes_list, mst)

for i, clus in enumerate(result_clusters):
    print(f"Cluster {i+1}: {clus}")