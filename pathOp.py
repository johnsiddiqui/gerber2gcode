import pulp
import numpy as np

# Positions of features
positions = {
    1: (2, 3),
    2: (8, 4),
    3: (5, 8),
    4: (1, 1),
    5: (6, 1)
}

# Compute distance matrix
N = len(positions)
distances = np.zeros((N, N))
for i in range(1, N+1):
    for j in range(1, N+1):
        xi, yi = positions[i]
        xj, yj = positions[j]
        distances[i-1, j-1] = np.hypot(xi - xj, yi - yj)

# Initialize MILP problem
prob = pulp.LpProblem("CoatingPathOptimization", pulp.LpMinimize)

# Create binary variables x[i][j] for moving from i to j
x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(N)] for i in range(N)]

# Objective: Minimize total distance
prob += pulp.lpSum(distances[i][j] * x[i][j] for i in range(N) for j in range(N))

# Constraints
for i in range(N):
    prob += pulp.lpSum(x[i][j] for j in range(N) if j != i) == 1   # Leave each node once
    prob += pulp.lpSum(x[j][i] for j in range(N) if j != i) == 1   # Enter each node once

# Subtour elimination: Miller-Tucker-Zemlin (MTZ) formulation
u = [pulp.LpVariable(f"u_{i}", lowBound=0, upBound=N-1, cat="Continuous") for i in range(N)]
for i in range(1, N):
    for j in range(1, N):
        if i != j:
            prob += u[i] - u[j] + (N-1) * x[i][j] <= N-2

# Solve the problem
prob.solve()

# Print the solution
print("Status:", pulp.LpStatus[prob.status])
print("Optimal Path:")
for i in range(N):
    for j in range(N):
        if pulp.value(x[i][j]) == 1:
            print(f"From Feature {i+1} to Feature {j+1}")
