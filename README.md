# sloan

# DFS

def DFS(graph,start,goal):
    visited = set()
    stack = [(start,[])]

    while stack:
        node, path = stack.pop()
        print('Node Removed From Stack is: '+node)
        visited.add(node)

        if node == goal:
            return path + [node]

        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append((neighbor,path+[node]))
    return None

graph = {
    'S': ['A','B','C'],
    'A': ['D'],
    'B': ['E'],
    'C': ['F'],
    'E': [],
    'G': [],
    'D': [],
    'F': ['G']

}

start = input('Enter Start Node: ')
goal = input('Enter Goal Node: ')

path = DFS(graph,start,goal)


if path:
    print(f"Path From {start} to {goal} is: {path}")
else:
    print(f"Path Not Found from {start} to {goal}.")



_____________________________________________________________________

# BFS
def BFS(graph,start,goal):
    visited = set()
    queue = [(start,[])]

    while queue:
        node, path = queue.pop(0)
        print('Node Removed From Queue is: '+node)
        visited.add(node)

        if node == goal:
            return path + [node]

        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append((neighbor,path + [node]))

    return None


graph = {

    'S': ['A','B','C'],
    'A': ['D'],
    'B': ['E'],
    'C': ['F'],
    'E': ['G'],
    'G': [],
    'D': [],
    'F': []

}


start = input('Enter Start Node: ')
goal = input('Enter Goal Node: ')

path = BFS(graph,start,goal)

if path:
    print(f"Path from {start} to {goal} is: {path}")
else:
    print(f"No Path found from {start} to {goal}.")

_____________________________________________________________________

#A*

import heapq

def AOstar(graph, start, goal, heuristic, alpha=0.5):
    visited = set()
    open_list = [(0, start, [], 0)]  # Priority queue with (f-cost, node, path, visits) tuples

    while open_list:
        f_cost, node, path, visits = heapq.heappop(open_list)
        print('Node Removed From Open List is:', node)
        visited.add(node)

        if node == goal:
            return path + [node]

        for neighbor, edge_length in graph[node]:
            if neighbor not in visited:
                g_cost = len(path) + edge_length  # g-cost is the actual cost from start to the current node
                h_cost = (1 - alpha) * heuristic[neighbor] + alpha * (visits + 1)
                f_cost = g_cost + h_cost

                heapq.heappush(open_list, (f_cost, neighbor, path + [node], visits + 1))

    return None

# Define the graph and heuristic values
graph = {
    'S': [('A', 1), ('B', 5), ('C', 8)],
    'A': [('S', 1), ('D', 3), ('E', 7), ('G', 9)],
    'B': [('S', 5), ('G', 4)],
    'C': [('S', 8), ('G', 5)],
    'D': [('A', 3)],
    'E': [('A', 7)],
}

heuristic = {'S': 8, 'A': 9, 'B': 4, 'C': 3, 'D': 5000, 'E': 5000, 'G': 0}

start = input('Enter Start Node: ')
goal = input('Enter Goal Node: ')

path = AOstar(graph, start, goal, heuristic)

if path:
    print(f"Path from {start} to {goal} is: {path}")
else:
    print(f"Path not found from {start} to {goal}.")

_____________________________________________________________________

# AO*
def Cost(H, condition, weight = 1):
	cost = {}
	if 'AND' in condition:
		AND_nodes = condition['AND']
		Path_A = ' AND '.join(AND_nodes)
		PathA = sum(H[node]+weight for node in AND_nodes)
		cost[Path_A] = PathA

	if 'OR' in condition:
		OR_nodes = condition['OR']
		Path_B =' OR '.join(OR_nodes)
		PathB = min(H[node]+weight for node in OR_nodes)
		cost[Path_B] = PathB
	return cost

# Update the cost
def update_cost(H, Conditions, weight=1):
	Main_nodes = list(Conditions.keys())
	Main_nodes.reverse()
	least_cost= {}
	for key in Main_nodes:
		condition = Conditions[key]
		print(key,':', Conditions[key],'>>>', Cost(H, condition, weight))
		c = Cost(H, condition, weight) 
		H[key] = min(c.values())
		least_cost[key] = Cost(H, condition, weight)		 
	return least_cost

# Print the shortest path
def shortest_path(Start,Updated_cost, H):
	Path = Start
	if Start in Updated_cost.keys():
		Min_cost = min(Updated_cost[Start].values())
		key = list(Updated_cost[Start].keys())
		values = list(Updated_cost[Start].values())
		Index = values.index(Min_cost)
		
		# FIND MINIMIMUM PATH KEY
		Next = key[Index].split()
		# ADD TO PATH FOR OR PATH
		if len(Next) == 1:

			Start =Next[0]
			Path += '<--' +shortest_path(Start, Updated_cost, H)
		# ADD TO PATH FOR AND PATH
		else:
			Path +='<--('+key[Index]+') '

			Start = Next[0]
			Path += '[' +shortest_path(Start, Updated_cost, H) + ' + '

			Start = Next[-1]
			Path += shortest_path(Start, Updated_cost, H) + ']'

	return Path
		
		

H = {'A': -1, 'B': 5, 'C': 2, 'D': 4, 'E': 7, 'F': 9, 'G': 3, 'H': 0, 'I':0, 'J':0}

Conditions = {
'A': {'OR': ['B'], 'AND': ['C', 'D']},
'B': {'OR': ['E', 'F']},
'C': {'OR': ['G'], 'AND': ['H', 'I']},
'D': {'OR': ['J']}
}
# weight
weight = 1
# Updated cost
print('Updated Cost :')
Updated_cost = update_cost(H, Conditions, weight=1)
print('*'*75)
print('Shortest Path :\n',shortest_path('A', Updated_cost,H))

_____________________________________________________________________

#### Jug A: 4 lit max cap
#### Jug B: 3 lit max cap
#### Goal: 2 lit
#### start (0,0)  goal (a,2) or goal(2,b)

# Prompt the user to enter X, Y, and Z values
X = int(input("Enter the capacity of Jug A: "))
Y = int(input("Enter the capacity of Jug B: "))
Z = int(input("Enter the target amount to measure: "))

# Initialize the initial state
initial_state = (0, 0)

# Initialize the `visited` set
visited = set()

# Initialize the `path` list
path = []

# Define the `dfs()` function
def dfs(jug_state, path):

    # Get the current state of the jugs
    jug_a, jug_b = jug_state

    # Check if the goal state has been reached
    if jug_a == Z or jug_b == Z:
        path.append(jug_state)
        return True

    # Add the current state to the visited set
    visited.add(jug_state)

    # Define the possible operations that can be performed
    operations = [
        (X, jug_b),  # Fill A
        (jug_a, Y),  # Fill B
        (0, jug_b),  # Empty A
        (jug_a, 0),  # Empty B
        (max(0, jug_a - (Y - jug_b)), min(jug_a + jug_b, Y)),  # Pour A to B
        (min(jug_a + jug_b, X), max(0, jug_b - (X - jug_a)))  # Pour B to A
    ]

    # Iterate over the possible operations
    for op in operations:

        # Check if the operation has already been performed
        if op not in visited:

            # Add the new state to the path
            path.append(jug_state)

            # Recursively call the `dfs()` function on the new state
            if dfs(op, path):
                return True

            # Remove the new state from the path
            path.pop()

    # Return False if none of the operations lead to the goal state
    return False

# Call the `dfs()` function on the initial state
if dfs(initial_state, path):
    print("Solution found:")
    for step in path:
        print(f"Jug A: {step[0]} liters, Jug B: {step[1]} liters")
else:
    print("No solution found.")

_____________________________________________________________________

# N Queens Problem:

class Solution:
    def solveNQueens(self, n: int):
        col = set()
        posDiag = set() # (r + c)
        negDiag = set() # (r - c)

        res = []
        board = [["."]* n for i in range(n)]

        def backtrack(r):
            if r == n:
                copy = ["".join(row) for row in board]
                res.append(copy)
                return

            for c in range(n):
                if c in col or (r + c) in posDiag or (r - c) in negDiag:
                    continue

                col.add(c)
                posDiag.add(r + c)
                negDiag.add(r - c)
                board[r][c] = "Q"

                backtrack(r + 1)

                col.remove(c)
                posDiag.remove(r + c)
                negDiag.remove(r - c)
                board[r][c] = "."

        backtrack(0)
        return res


q = Solution()

# Get user input for the number of queens
n = int(input("Number of Queens: "))

# Call the solveNQueens method
res = q.solveNQueens(n)

if res:
    last_solution = res[1]
    print(f"Solution for {n} Queens:")
    for row in last_solution:
        print(" ".join(row))
else:
    print("No solutions found for this configuration.")
