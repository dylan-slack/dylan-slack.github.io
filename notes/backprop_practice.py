import multiprocessing
import numpy as np
import time
from functools import partial

def experiment():
	x = np.array([3, 2, 8, 8, 4, 1, 0]) * 1.
	y = np.array([3, 2, 8, 8, 4, 1, 0]) * 1.
	sort = np.argsort(x)
	resort = np.argsort(sort)

	def f(x):
		return np.sort(x, axis=0) / np.median(x)


	def b_f(inputs, gradients):
		answer = (1./np.median(x)) - (x/(np.median(x)**2))
		return answer * gradients[resort]

	y += 1e-8
	print(((f(y) - f(x)) / 1e-8)[resort])
	print(b_f(x, np.ones_like(x)))


def parallel_inner(i, A, B):
	results = np.zeros(B.shape[1])
	for j in range(B.shape[1]):
		s_i_j = 0
		for q in range(A.shape[1]):
			s_i_j += A[i, q] * B[q, j]
		results[j] = s_i_j
	return results

def parallel_block(i: tuple, j: tuple, q: tuple):
	


def run():

	# matrix multiplication

	A = np.random.rand(100, 100)
	B = np.random.rand(100, 100)

	# naive

	start = time.time()
	s = np.zeros((A.shape[0], B.shape[1]))
	for i in range(A.shape[0]):
		for j in range(B.shape[1]):
			for q in range(A.shape[1]):
				s[i,j] += A[i, q] * B[q, j]
	end = time.time()
	print(end - start)

	assert np.allclose(A @ B, s)

	start = time.time()
	pool = multiprocessing.Pool(processes=8)
	pi_f = partial(parallel_inner, A=A, B=B)
	out = pool.map(pi_f, range(A.shape[0]))
	pool.close()
	end = time.time()
	print(end - start)
	s = np.vstack(out)

	assert np.allclose(A @ B, s)



if __name__ == "__main__":
	run()
