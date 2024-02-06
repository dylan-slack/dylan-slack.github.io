import numpy as np


a = np.random.rand(10, 10)
b = np.random.rand(10, 10)
c = a @ b


sol = np.zeros((a.shape[0], a.shape[1]))

for i in range(a.shape[0]):
	for j in range(b.shape[1]):
		for q in range(a.shape[1]):
			sol[i, j] += a[i, q] * b[q, j]

np.allclose(c, sol)

sol = np.zeros((a.shape[0], a.shape[1]))
batch = int(a.shape[0] ** 0.5)

for i in range(0, a.shape[0], batch):
	for j in range(0, b.shape[1], batch):
		for q in range(0, a.shape[1], batch):

			for i_t in range(i, min(i+batch, a.shape[0])):
				for j_t in range(j, min(j+batch, b.shape[0])):
					for q_t in range(q, min(q+batch, a.shape[1])):
						sol[i_t, j_t] += a[i_t, q_t] * b[q_t, j_t]

np.allclose(c, sol)


sol = np.zeros((a.shape[0], b.shape[1]))
for i in range(a.shape[0]):
	for j in range(b.shape[1]):
		for q in range(a.shape[1]):
			sol[i, j] = a[i, q] * b[q, j]

# parallelism

# one layer dnn

data = np.random.rand(100, 3)
y = ((data[:, 1] > data[:, 0]**7) * 1.)[:, None]


w1 = np.random.rand(3, 10)
w2 = np.random.rand(10, 1)


def sig(x):
	"""Sigmoid function."""
	return 1 / (1 + np.exp(-1 * x))


def forward(data, y, w1, w2):
	l1_o = data @ w1 # -> (B, 10)
	l2_o = l1_o @ w2 # (B, 10) (10, 1) -> (B, 1)
	preds = sig(l2_o)
	cost = ((preds - y) ** 2).mean()
	return cost, preds, l1_o, l2_o


def backward(preds, y, l1_o, l2_o, data, w1, w2):

	# d_c_d_preds_i = (2 * (preds - y) / preds.shape[0] )

	d_cost_d_preds = ( 2 * (preds - y) / preds.shape[0] ) # (B, 1)
	d_preds_d_l2_o = d_cost_d_preds * sig(l2_o) * (1 - sig(l2_o)) # (B, 1)

	d_l2_o_d_l1_o = d_preds_d_l2_o @ w2.T # w2: (10, 1) (B, 1) 
	d_l2_o_d_w2 = l1_o.T @ d_preds_d_l2_o # (B, 10).T (B, 1)  -> (10, 1)
	d_l1_o_d_w1 = data.T @ d_l2_o_d_l1_o # (100, 3), (3, 10)

	w2 = w2 - 1e-2 * d_l2_o_d_w2
	w1 = w1 - 1e-2 * d_l1_o_d_w1

	return w1, w2


for j in range(10000):
	cost, preds, l1_o, l2_o = forward(data, y, w1, w2)
	w1, w2 = backward(preds, y, l1_o, l2_o, data, w1, w2)
	if j % 100 == 0:
		print(cost)
