import numpy as np

# Basic idea

def linear(X, W):

    assert W.shape == (1, X.shape[1])
    assert len(X.shape) == 2

    sol = X @ W.T

    def linear_back(a, w):
        # (x_1_1 * w_1) + (x_1_2 * w_1)... + x_n * w_n
        # d out / d_w_i  = x_i
        # w.r.t linear layers
        # (3, 1), W=(1,2) => (3, 3)
        # print(a)
        out = a
        return out

    return sol, linear_back

def s(x):
    return 1. / (1. + np.exp(-1. * x))


def sigmoid(v):

    sol = s(v)

    def sig_back(c):

        return s(c) * (1 - s(c))

    return sol, sig_back


def subtract(a, b):

    sol = a - b

    def sub_back(c):
        # a - b
        # d/da = 1

        return 1 # just for first element

    return sol, sub_back


def square(a):

    sol = a ** 2

    def s_back(c):
        # 2 * c_i
        return 2.0 * c

    return sol, s_back


def mean(a):

    sol = np.mean(a)

    def mean_back(c):
        # c_1 / n + ... + c_n / n
        # d_out / d_c_n = 1 / n

        return (1 / a.size)

    return sol, mean_back


a = np.array([[1.0, 1.0], [1.0, 5.0], [1.0, 4.0], [1.0, -10.0]]) # (3, 2)
w = np.array([[0.0, 0.0]]) # (1, 2)
t = np.array([[0., 1., 1., 0.]]).T # (3, 1)


def avg_cost(a, w, trip = False):
    if trip:
        w[0, 1] += 1e-10
    linear_out, linear_back = linear(a, w)
    sig_out, sig_back = sigmoid(linear_out)
    diff, diff_out = subtract(sig_out, t)
    cost, cost_out = square(diff)
    ac, avg_cost_out = mean(cost)
    outs = (linear_back, sig_back, diff_out, cost_out, avg_cost_out)
    intermediates = (linear_out, sig_out, diff, cost)
    return ac, outs, intermediates


def grad_descent(a, w):

    for i in range(100):

        ac, outs, ints = avg_cost(a, w)
        linear_back, sig_back, diff_out, cost_out, avg_cost_out = outs
        linear_out, sig_out, diff, cost = ints

        w_c = w.copy()
        ac_e, _, _ = avg_cost(a, w_c, trip=True)
        grad_approx = (ac_e - ac) / 1e-10

        previous = 1.0
        d_avg_cost_d_cost = previous * avg_cost_out(cost)
        d_avg_cost_d_diff = d_avg_cost_d_cost * cost_out(diff)
        d_avg_cost_d_sub_out = d_avg_cost_d_diff * diff_out(sig_out)
        d_avg_cost_d_sig_out = d_avg_cost_d_sub_out * sig_back(linear_out)
        d_avg_cost_d_linear_out = d_avg_cost_d_sig_out * linear_back(a, w)
        grad = np.sum(d_avg_cost_d_linear_out, axis=0)
        w = w - 0.5 * grad

        assert np.allclose(grad[1], grad_approx, atol=1e-3), f"{grad[1]} | {grad_approx}"

    return w, sigmoid(linear(a, w)[0])[0]

# grad_descent(a, w)


# Backprop through strange functions


a = np.array([100, 20, 30, 5, 15])

# np.sort(a) / np.median(a)
# x / x_i => x * (x_i ^ -1) => (x_i ^ -1) - x * (x_i ^ -2)

def forward(x):

    asort = np.argsort(x) # [4, 0, 2, 1, 3] => [1, 3, 2, 4, 0]
    re_sort = np.argsort(asort) # => [4, 0, 2, 1, 3]
    med = np.median(x)

    def backward(x_in, grad):
        resorted_grad = grad[re_sort]
        med_in = np.median(x_in)
        out = (med_in ** -1) - ( x_in * med_in ** -2 )
        return out * resorted_grad

    return x[asort] / med, backward

out, back = forward(a)
print(out)
print(back(a, np.ones_like(out)))
# d = (forward(a + 1e-8)[0]  - forward(a)[0]) / 1e-8
# print(d)

# chain rule



def f_a(x):
    return x ** 2

def f_b(x, y):
    return x - y

def f_c(x):
    return 0.5 * x


def b_f_a(x, grads):
    o = 2 * x
    grad_out = grads * o
    return grad_out

def b_f_b(x, y, grads):
    o_1, o_2 = 1, -1
    return o_1 * grads, o_2 * grads

def b_f_c(x, grads):
    return 0.5 * grads

a, b = 5, 10

b_out = f_b(a, b)
c_out = f_c(b_out)
a_out = f_a(c_out)
print(a_out) # 6.25

# dO / d_b_in
d_a_d_c = b_f_a(c_out, 1.) # initial gradients are 1, dO/dO = 1.
d_c_d_b = b_f_c(b_out, d_a_d_c)
d_b_d_b_in = b_f_b(a, b, d_c_d_b)
print(d_b_d_b_in) # (-2.5, 2.5)

# ( 0.5 (a - b) ) ** 2 
# d/da = 2 * (0.5 (a - b )) (0.5) => 0.5 (a - b)
assert d_b_d_b_in[0] == (0.5 * (a - b)), f"{0.5 * (a - b)}"


print()

# matmul

def matmul_f(A, B):

    return A @ B

def matmul_b(A, B, grads):

    # A : (a, b)
    # B : (b, c)
    # grads : (a, c)

    # d_A_i_j / d_A_@_B_i_j

    # print(A.T.shape)
    # print(grads.shape)
    return grads @ B.T, A.T @ grads

a = np.random.randint(3, size=(10, 20)) * 1.
b = np.random.randint(3, size=(20, 5)) * 1.
mat_out = matmul_b(a, b, 1. * np.ones((10, 5)))


# Gradients for mean of largest top_k indices

def sort_forward(a):
    # assuming a is shape (B, N) and sorting along first dimension
    argsort = np.argsort(a, axis=1)
    resort = np.argsort(argsort, axis=1)
    def sort_backward(inputs, gradients):
        # gradients will be sorted, so map back
        gradients = np.take_along_axis(gradients, resort, axis=1)
        return gradients
    return np.take_along_axis(a, argsort, axis=1), sort_backward


def slice_last_k(a, k: int = 4):
    def slice_backward(inputs, gradients):
        grads = np.zeros_like(inputs) * 1.
        grads[:, -k:] = gradients
        return grads
    return a[:, -k:], slice_backward

def mean(a):
    def mean_backward(inputs, gradients):
        # (a_0 + a_1 + ... + a_n) / n  => d_i/dx = 1/n
        return gradients[:, None] * (np.ones_like(a) / a.shape[1])
    return np.mean(a, axis=1), mean_backward

inputs = np.array(
    [[3, 1, 9, 10, 3, 1],
     [3, 3, 3, 3, 3, 4]]
)

sort_out, sort_b = sort_forward(inputs)
slice_out, slic_b = slice_last_k(sort_out)
mean_out, mean_b = mean(slice_out)

s = np.ones_like(mean_out)
b = mean_b(slice_out, s)
sl = slic_b(sort_out, b)
f = sort_b(inputs, sl)
print(inputs)
# [[ 3  1  9 10  3  1]
#  [ 3  3  3  3  3  4]]
print(f)
# [[0.25 0.   0.25 0.25 0.25 0.  ]
#  [0.   0.   0.25 0.25 0.25 0.25]]

print()

# Neural Network


def relu(a):
    mask = a < 0
    def relu_backward(inputs, gradients):
        # weight gradients are (W, 1)
        # a is (B, W)
        return gradients * (mask * 1.)
    a[mask] = 0 # nonlinearity
    return a, relu_backward


def sig(x):
    return 1. / (1. + np.exp(-1. * x))


def d_sig(x):
    return sig(x) * (1 - sig(x))


def sigmoid(a):
    def sig_back(inputs, gradients):
        # inputs: (B, D) 
        # gradients: (B, D)
        sig_grads = d_sig(inputs)
        # print(sig_grads.shape)
        # print(gradients.shape)
        return sig_grads * gradients
    return sig(a), sig_back


def cost(a, t):
    def cost_back(a, t):
        # assuming gradients is always 1
        d = (t / (a + 1e-32)) - ((1 - t) / (1 - a + 1e-32))
        d *= -1. / a.size
        return d
    return -1 * (t * np.log(a + 1e-32) + (1 - t) * np.log(1 - a + 1e-32)).mean(), cost_back


data = np.random.normal(size=(4000, 5))

y = np.logical_xor(np.logical_and(data[:, 0] > 0, data[:, 1] > 0), np.logical_and(data[:, 0] < 0, data[:, 1] < 0)) * 1.
y = y[:, None]

def forward_backward(d, y, w1, w2, final_layer, lr=1e-2, i=0, split='train'):

    # forward
    l1_o = matmul_f(d, w1)
    r_l1_o, b_l1_o = relu(l1_o)
    l2_o = matmul_f(r_l1_o, w2)
    # r_l2_o, b_l2_o = relu(l2_o)
    f_o = matmul_f(l2_o, final_layer)
    sig_o, b_sig_o = sigmoid(f_o)
    cost_o, b_cost_o = cost(sig_o, y)

    # backward
    d_cost_d_sig = b_cost_o(sig_o, y)
    d_sig_d_f = b_sig_o(f_o, d_cost_d_sig)
    d_f_d_r_l2_o, d_f_d_wf = matmul_b(l2_o, final_layer, d_sig_d_f)
    final_layer = final_layer - lr * d_f_d_wf
    d_l2_o_d_l1_o, d_w2_d_l1_o = matmul_b(l1_o, w2, d_f_d_r_l2_o)
    w2 = w2 - lr * d_w2_d_l1_o
    d_relu_o_d_l1_0 = b_l1_o(l1_o, d_l2_o_d_l1_o)
    _, d_l1_o_d_w1 = matmul_b(d, w1, d_relu_o_d_l1_0)
    w1 = w1 - lr * d_l1_o_d_w1

    if i % 100 == 0 and split == "test":
        print(cost_o)
        acc = ((sig_o > 0.5) * 1. == y).mean()
        print(acc)

    return w1, w2, final_layer


w1 = np.random.normal(size=(5, 50))
w2 = np.random.normal(size=(50, 50))
final_layer = np.random.normal(size=(50, 1))

train_x, train_y = data[:3000], y[:3000]
test_x, test_y = data[3000:], y[3000:]
for i in range(10_000):
    perm = np.random.permutation(train_x.shape[0])
    train_x, train_y = train_x[perm], train_y[perm]
    for b in range(0, train_x.shape[0], 512):
        w1, w2, final_layer = forward_backward(train_x[b:b+512], train_y[b:b+512], w1, w2, final_layer, 3e-4, i)
    if i % 100 == 0:
        forward_backward(test_x, test_y, w1, w2, final_layer, 1e-2, i, 'test')
