import numpy as np
import timeit

# **Vectors and Scalars**

a = np.arange(1_000)
a_magnitude = np.sqrt(np.sum(a**2))
print(a_magnitude)
# 18243.72

a_unit = a / a_magnitude
norm = np.linalg.norm(a_unit, ord=2)
assert norm == 1.0

# Broadcasting

a = np.random.rand(100, 42, 3)
b = np.random.rand(42, 3)
c = a + b
print(c.shape) # (100, 42, 3)
assert c[0].sum() == (a[0] + b).sum() # True

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) # (3, 4)
arranged = np.arange(a.shape[1]).reshape(1, a.shape[1]) # (1, 4)
inds = np.array([1, 3, 2]).reshape(a.shape[0], 1) # (3, 1)
mask = arranged <= inds # (4, 3)
print(mask)
# [[ True  True False False]
#  [ True  True  True  True]
#  [ True  True  True False]]
a[~mask] = 0
print(a)
# [[ 1  2  0  0]
#  [ 5  6  7  8]
#  [ 9 10 11  0]]

# Dot Product

a = np.random.rand(10)
b = np.random.rand(10)
c = a.dot(b)
c_d = (a * b).sum()
assert np.allclose(c, c_d)

# Matrix multiplication

a = np.random.rand(100, 10)
b = np.random.rand(100, 10)
c = a @ b.T # (500, 100), simple
b_t = b.T # (100, 500)

def mat(a, b):
    c_manual = []
    for i in range(a.shape[0]):
        col = []
        for j in range(b.shape[1]):
            c_v = 0
            for q in range(a.shape[1]):
                c_v += a[i, q] * b[q, j]
            col.append(c_v)
        c_manual.append(col)
    return c_manual

assert np.allclose(c, mat(a, b_t))

def mat_swap(a, b):
    sol = np.zeros((a.shape[0], b.shape[1]))
    for q in range(a.shape[1]):
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                sol[i, j] += a[i, q] * b[q, j]
    return sol

assert np.allclose(mat_swap(a, b_t), mat(a, b_t))


def mat_tiled(a, b, T):
    sol = np.zeros((a.shape[0], b.shape[1]))
    for i in range(0, a.shape[0], T):
        for j in range(0, b.shape[1], T):
            for q in range(0, a.shape[1], T):

                # break up each matmul into chunks
                for i_t in range(i, min(i+T, a.shape[0])): # -1 for 0 index
                    for j_t in range(j, min(j+T, b.shape[1])):
                        for q_t in range(q, min(q+T, a.shape[1])):
                            sol[i_t, j_t] += a[i_t, q_t] * b[q_t, j_t]
    return sol

T = int(a.shape[1] ** 0.5)
assert np.allclose(mat_tiled(a, b_t, T), mat(a, b_t))

# Gaussian Elimination

def gaussian_elimination(a):
    cur_j = 0
    n_swaps = 0
    for i in range(a.shape[1]):
        col_sort = np.argsort(np.abs(a[:, i]))
        col_sort = col_sort[col_sort >= cur_j]
        if a[col_sort[-1], i] == 0:
            continue # solved
        else:
            if cur_j != col_sort[-1]:
                largest_row = a[col_sort[-1]].copy()
                a[col_sort[-1]] = a[cur_j]
                a[cur_j] = largest_row
                # swap
                # d *= -1
                n_swaps += 1
            for q in range(cur_j+1, a.shape[0]):
                if a[q, i] == 0:
                    continue
                a[q] = a[cur_j] - a[q] * a[cur_j, i] / a[q, i]
            cur_j += 1
    determinant = a[np.arange(a.shape[0]), np.arange(a.shape[1])].prod()
    rank = (np.sum(np.abs(a), axis=1) > 0).sum()
    return determinant, rank

a = np.random.randint(3, size=(4, 4)) * 1.
det, rank = gaussian_elimination(a)
det_np = np.linalg.det(a)
assert np.allclose(det, det_np), f"{det} | {det_np}" # True
assert np.linalg.matrix_rank(a) == rank, f"{rank}"


# Relationship of trace and Frobenius inner product

import itertools

# lets say we have
a = np.random.rand(20, 10)
b = np.random.rand(20, 10)

# the Frobenius inner product is given as 
frob = 0
iterator = itertools.product(range(len(a)), range(len(a[0])))
for i, j in iterator:
    frob += a[i][j] * b[i][j]

# examine matmul

# transponse
new_a = [[None for _ in range(len(a))] for _ in range(len(a[0]))]
iterator = itertools.product(range(len(a)), range(len(a[0])))
for i, j in iterator:
    new_a[j][i] = a[i][j]
new_a = np.array(new_a)

# matmul
trace_frob = 0
sol = np.zeros((new_a.shape[0], b.shape[1]))
for i in range(new_a.shape[0]):
    for j in range(b.shape[1]):
        for q in range(new_a.shape[1]):
            sol[i, j] += new_a[i, q] * b[q, j]
        if i == j:
            trace_frob += sol[i, j]

assert np.allclose(trace_frob, frob), f"{trace_frob} {frob}"

# QR Decomposition

n = 4
v = np.random.rand(n, n)
for i in range(n):
    for j in range(i, n):
        v[j, i] = v[i, j]

ev, _ = np.linalg.eig(v)

def graham_schmidt(a):
    a = a.T
    # applying graham schmidt to columns, transposing for ease
    proj_v = np.zeros(a.shape[1])
    sol = []
    u = []
    for i in range(a.shape[0]):
        # get projection
        proj_v = np.zeros(a.shape[1])
        for j in range(i):
            proj_v += (a[i].dot(u[j]) / u[j].dot(u[j])) * u[j]
        # get u_i by subtracting all projections from a
        u_i = a[i] - proj_v
        u.append(u_i)
        e_i = u_i / np.linalg.norm(u_i, ord=2)
        sol.append(e_i)
    return np.vstack(sol).T

for i in range(50):
    Q = graham_schmidt(v)
    v = Q.T @ v @ Q
    s = np.arange(v.shape[0])
    qr_e_v = v[s, s]

print("Eigenvalues:")
print(ev)
print(qr_e_v)
assert np.allclose(np.sort(ev), np.sort(qr_e_v), rtol=1e-3)
print()

