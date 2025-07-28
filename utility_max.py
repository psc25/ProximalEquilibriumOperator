import numpy as np
import tensorflow as tf
import itertools as itt
import scipy.special as ssp
import matplotlib.pyplot as plt
import time

import seaborn as sns
sns.set_style("whitegrid")
sns.color_palette("deep")

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(0)
print("utility_max")

E = 3
sumE = 3
ind = np.array([e for e in itt.product(np.arange(E+1), repeat = 2) if sum(e) <= sumE])
R = len(ind)
print(R)

S0 = 1.0
r = 0.005
mu = 0.01
sigma = 0.05
y0 = S0

N = 500
n = 100
T = 1.0

dt = T/n
tt = np.linspace(0.0, T, n).astype(np.float32)
t1 = np.reshape(tt, [1, -1])
dW = np.random.normal(size = [N, n-1], scale = np.sqrt(dt)).astype(np.float32)
W = np.concatenate([np.zeros([N, 1]), np.cumsum(dW, axis = 1)], axis = 1)
S = S0*np.exp((mu-0.5*sigma**2)*t1 + sigma*W)
B = np.exp(r*np.reshape(tt, [1, -1]))
lam = (mu-r)/sigma
Z = np.exp(-lam*W - 0.5*lam**2*t1)
H = Z/B

# Proximal operator of f
t2 = np.reshape(tt[:-1], [1, 1, 1, n-1])
W2 = np.reshape(W[:, :-1], [1, 1, N, n-1])
tW2 = np.power(t2, np.reshape(ind[:, 0], [1, -1, 1, 1]))*np.power(W2, np.reshape(ind[:, 1], [1, -1, 1, 1]))/np.sqrt(ssp.factorial(np.reshape(ind[:, 1], [1, -1, 1, 1])))

compute_Ginv = False
if compute_Ginv:
    G = np.zeros([R, R]).astype(np.float32)
    for i in range(R):
        for j in range(R):
            m = ind[i, 1] + ind[j, 1]
            if m % 2 == 0:
                a = ind[i, 0] + ind[j, 0] + 0.5*m + 1.0
                G[i, j] = ssp.factorial2(m-1)/np.sqrt(ssp.factorial(ind[i, 1])*ssp.factorial(ind[j, 1]))*np.power(T, a)/a
    
    Ginv = np.linalg.inv(G)
    np.savetxt("utility_max/Ginv.csv", Ginv)
else:
    Ginv = np.expand_dims(np.loadtxt("utility_max/Ginv.csv", dtype = np.float32), axis = 0)
    
def prox_f(x):
    pi = tf.reduce_sum(tf.expand_dims(x, axis = -1)*tW2, axis = 1, keepdims = True)
    y = T*tf.reduce_mean(tf.math.maximum(pi, 0.0)*tW2, axis = (2, 3), keepdims = True)
    return tf.matmul(Ginv, y[:, :, 0])

# Functions g
K = 500
val_split = 0.2
Ktrain = int(K*(1.0-val_split))
Ktest = K-Ktrain
def u(y, eta, xrf):
    return tf.pow(tf.maximum(y-xrf, 0.0) + 1e-5, 1.0-eta)/(1.0-eta)

t3 = np.reshape(tt[:-1], [1, 1, 1, 1, n-1])
W3 = np.reshape(W[:, :-1], [1, 1, 1, N, n-1])
tW3 = np.power(t3, np.reshape(ind[:, 0], [1, -1, 1, 1, 1]))*np.power(W3, np.reshape(ind[:, 1], [1, -1, 1, 1, 1]))/np.sqrt(ssp.factorial(np.reshape(ind[:, 1], [1, -1, 1, 1, 1])))
B3 = np.reshape(B[:, :-1], [1, 1, 1, n-1])
dW3 = np.reshape(dW, [1, 1, N, n-1])
def g(x, eta, xrf):
    pi = tf.reduce_sum(tf.expand_dims(tf.expand_dims(x, axis = -1), axis = -1)*tW3, axis = 1)
    y = B[0, -1]*(y0 + tf.reduce_sum(pi/B3*((mu-r)*dt + sigma*dW3), axis = -1))
    return -tf.reduce_mean(u(y, eta, xrf), axis = -1)

eta = np.random.uniform(size = [K, 1, 1], low = 0.25, high = 0.75)
print(np.sum(eta == 1.0))
xrf = np.concatenate([np.zeros([Ktrain, 1, 1]), np.random.exponential(size = [Ktest, 1, 1], scale = 1.0)], axis = 0)

L = 10
M = 40
gamma = 0.5

ep = 5000
eval_every = 125
lr = 2e-6
batch_size = 100
nr_batch = int(Ktrain/batch_size)
print_details = True

ind_train = np.arange(Ktrain)
ind_test = np.arange(Ktrain, K)
init = tf.random_normal_initializer(stddev = 0.01)

xl = tf.random.normal(shape = [1, R, 1], stddev = 0.01, dtype = tf.float32)
train_tf = tf.placeholder(shape = (), dtype = tf.bool)
eta_tf = tf.placeholder(shape = (None, 1, 1), dtype = tf.float32)
xrf_tf = tf.placeholder(shape = (None, 1, 1), dtype = tf.float32)
for l in range(1, L+1):
    Al = tf.Variable(initial_value = init(shape = (1, R, R)), dtype = tf.float32)
    Bl = tf.Variable(initial_value = init(shape = (1, R, M)), dtype = tf.float32)
    xlm = tf.Variable(initial_value = init(shape = (1, R, M)), dtype = tf.float32)
    bl = tf.Variable(initial_value = init(shape = (1, R, 1)), dtype = tf.float32)
    gout = g(xl + xlm, eta_tf, xrf_tf)
    hd = tf.matmul(Al, xl) + tf.matmul(Bl, tf.expand_dims(gout, axis = -1)) + bl
    xl = gamma*xl + (1.0-gamma)*prox_f(hd)    
    
AL1 = tf.Variable(initial_value = init(shape = (1, R, R)), dtype = tf.float32)
xL1 = prox_f(tf.matmul(AL1, xl))

pi = tf.reduce_sum(tf.expand_dims(xL1, axis = -1)*tW2, axis = 1)
xTrue_tf = tf.placeholder(shape = (None, N, n-1), dtype = tf.float32)
loss = tf.reduce_mean(tf.square(pi - xTrue_tf))

global_step = tf.Variable(0, trainable = False)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

xTrue = np.zeros([K, N, n-1])
for k in range(K):
    xTrue[k] = np.fmax((mu-r)/(sigma**2*eta[k]*H[:, :-1])*(y0-xrf[k]/B[:, -2:-1])*np.power(Z[:, :-1], 1.0-1.0/eta[k])/np.exp(0.5*lam**2*(1-eta[k])/eta[k]**2*t1[:, :-1]), 0.0)
    
res_loss = np.nan*np.ones([ep, 2])
for i in range(ep):
    begin = time.time()
    np.random.shuffle(ind_train)
    loss1 = np.zeros(nr_batch)
    for r in range(nr_batch):
        ind_batch = ind_train[(r*batch_size):((r+1)*batch_size)]
        feed_dict = {train_tf: True, eta_tf: eta[ind_batch], xrf_tf: xrf[ind_batch], xTrue_tf: xTrue[ind_batch]}
        _, loss1[r], pi2 = sess.run([train_op, loss, pi], feed_dict)
        
    res_loss[i, 0] = np.mean(loss1)
    end = time.time()
    if print_details:
        print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 0]))
        
    if i == 0 or (i+1) % eval_every == 0:
        begin = time.time()
        feed_dict = {train_tf: False, eta_tf: eta[ind_test], xrf_tf: xrf[ind_test], xTrue_tf: xTrue[ind_test]}
        res_loss[i, 1], pi1 = sess.run([loss, pi], feed_dict)
        end = time.time()
        if print_details:
            print("\nEvaluation on test data:")
            print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 1]))
            print("")
            plt.plot(res_loss[:, 0], "-b", label = "Train")
            plt.plot(res_loss[:, 1], "og", label = "Test", markersize = 5)
            plt.ylim([0.0, 1.02*np.nanmax(res_loss)])
            plt.xlabel("Epochs")
            plt.ylabel("MSE")
            plt.legend(loc = "upper right")
            plt.show()
            
            plt.plot(tt[:-1], xTrue[Ktrain, 0], "-k", label = "True")
            plt.plot(tt[:-1], pi1[0, 0], ":r", label = "Predict")
            plt.plot(tt[:-1], xTrue[Ktrain+2, 0], "-k")
            plt.plot(tt[:-1], pi1[2, 0], ":r")
            plt.xlabel("$t$")
            plt.ylabel("$x_t(\\omega)$")
            plt.legend(loc = "upper right")
            plt.show()
            
        if np.isnan(res_loss[i, 0]) or np.isnan(res_loss[i, 1]):
            break
            
fig = plt.figure()
plt.plot(res_loss[:, 0], "-b", label = "Train")
plt.plot(res_loss[:, 1], "og", label = "Test", markersize = 5)
plt.ylim([0.0, 1.02*np.nanmax(res_loss)])
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(loc = "upper right")
plt.savefig("utility_max_loss.png", bbox_inches = 'tight', dpi = 500) 
plt.close(fig)

ind = np.argsort(np.mean(np.square(xTrue[Ktrain:, 0] - pi1[:, 0]), axis = -1))
for i in range(Ktest):
    if np.max(xTrue[ind[i], 0]) > 0.0:
        ind1 = ind[i]
        
for j in range(i+1, Ktest):
    if np.max(xTrue[ind[j], 0]) > 0.0:
        ind2 = ind[j]

fig = plt.figure()
plt.plot(tt[:-1], xTrue[Ktrain+ind1, 0], "-k", label = "True")
plt.plot(tt[:-1], pi1[ind1, 0], ":r", label = "Predict")
plt.plot(tt[:-1], xTrue[Ktrain+ind2, 0], "-k")
plt.plot(tt[:-1], pi1[ind2, 0], ":r")
plt.xlabel("$t$")
plt.ylabel("$x_t(\\omega)$")
plt.legend(loc = "upper right")
plt.savefig("utility_max_test.png", bbox_inches = 'tight', dpi = 500) 
plt.close(fig)