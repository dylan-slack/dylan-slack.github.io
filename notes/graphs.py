# binary search

a = list(range(-321, 12345))
outside = list(range(-1000, -700)) + list(range(12500, 13000))


def search(value, a=a):
	i, j = 0, len(a) - 1
	while i <= j: # if equivilent, we'll find and return this one
		mid = int((i + j) / 2)
		if a[mid] == value:
			return True
		elif a[mid] > value: # if mid is greater, this is now the top
			j = mid - 1 # decrement cause we already checked mid
		else:
			i = mid + 1 # same as above
	return False


assert all(map(search, a))
assert not any(map(search, outside))

to_remove = [-100, 20, 0, -1, 2, 10, 3333]
for v in to_remove:
	a.remove(v)

assert not any(map(search, to_remove))

# stronly connected components

# let's say we have some graph, and we want to find all the strongly connected components
# vertexes are strongly connected if we can reach a vertex from any other vertex


#  [ 2 1 6 5 3 ]

#  [1 -> 2 -> 3 -> 1] [3 -> 5 -> 6]
# [3, 5, 6, 1, 2]
# []


# adjacency graph
graph = {
	1: [2],
	2: [3],
	3: [5, 4],
	4: [1]
}

all_nodes = set()
for g in graph:
	all_nodes.add(g)
	for v in graph[g]:
		all_nodes.add(v)

transposed_graph = {a: [] for a in all_nodes}
roots = {a: [] for a in all_nodes}

for g in graph:
	for v in graph[g]:
		transposed_graph[v].append(g)

print(transposed_graph)

# brute force, if two nodes can be reached both ways, combine them 


# kasaraju's

stack = []
visited = set()


class Search:

	def __init__(self, graph):
		self.stack = []
		self.visited = set()
		self.graph = graph

	def search(self, c):
		if c in self.visited:
			return
		self.visited.add(c)
		if c in graph:
			for child in self.graph[c]:
				self.search(child)
		self.stack.append(c)


ss = Search(graph)

for a in all_nodes:
	ss.search(a)

stack = ss.stack

print(stack)

# 

visited = set()

def backsearch(c, root):
	if c in visited:
		return
	roots[root].append(c)
	visited.add(c)
	for child in transposed_graph[c]:
		backsearch(child, root)

while stack:
	s = stack.pop()
	backsearch(s, s)

print(roots)




