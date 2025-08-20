import numpy as np
import tensorflow as tf
import itertools as itt
import matplotlib.pyplot as plt
import time

import seaborn as sns
sns.set_style("whitegrid")
sns.color_palette("deep")

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(0)
print("quad_hedg")

new_par = False

E = 2
sumE = 2
ind = np.array([e for e in itt.product(np.arange(E+1), repeat = 3) if sum(e) <= sumE])
R = 1+len(ind)
print(R)

T = 1.0
S0 = [10.0, 0.05]
mu = 0.02
kappa = 4.0
theta = 0.05
sigma = 0.3
rho = -0.7

N = 500
n = 100
dt = T/n
tt = np.linspace(0.0, T, n).astype(np.float32)

if new_par:
    dW = np.random.normal(size = [N, n, 2], scale = np.sqrt(dt)).astype(np.float32)
    S = np.zeros([N, n, 2], dtype = np.float32)
    S[:, 0, 0] = S0[0]
    S[:, 0, 1] = S0[1]
    for t in range(n-1):
        S[:, t+1, 0] = S[:, t, 0] + mu*S[:, t, 0]*dt + np.sqrt(S[:, t, 1])*S[:, t, 0]*dW[:, t, 0]
        S[:, t+1, 1] = S[:, t, 1] + kappa*(theta - S[:, t, 1])*dt + sigma*np.sqrt(S[:, t, 1])*(rho*dW[:, t, 0] + np.sqrt(1.0-rho**2)*dW[:, t, 1])
        
    np.savetxt("quad_hedg/S.csv", np.reshape(S, [N, -1]))
else:
    S = np.reshape(np.loadtxt("quad_hedg/S.csv", dtype = np.float32), [N, n, 2])

# Proximal operator of f: R \oplus L^2(S) -> (-\inf,\inf]
C = 10.0
def prox_f(x):
    x0 = tf.minimum(tf.maximum(x[:, 0:1], 0.0), C)
    return tf.concat([x0, x[:, 1:]], axis = 1)

# Functions g: R \oplus L^2(S) -> (-inf,inf)
K = 250
val_split = 0.2
Ktrain = int(K*(1.0-val_split))
Ktest = K-Ktrain
t2 = np.reshape(tt[:-1], [1, 1, 1, 1, -1])
Y2 = np.reshape(np.log(S[:, :-1, 0]/S0[0]), [1, 1, 1, N, n-1])
V2 = np.reshape(S[:, :-1, 1]/S0[1], [1, 1, 1, N, n-1])
ind2 = np.reshape(ind, [1, R-1, 1, 1, 1, 3])
tYV2 = np.power(t2, ind2[:, :, :, :, :, 0])*np.power(Y2, ind2[:, :, :, :, :, 1])*np.power(V2, ind2[:, :, :, :, :, 2])
diffS = np.reshape(S[:, 1:, 0] - S[:, :-1, 0], [1, 1, N, -1])
termS = np.reshape(S[:, -1, 0], [1, 1, N])
def g(x, l, train, Kstr_call_tf, Kstr_put_tf):
    x = tf.expand_dims(tf.expand_dims(x, axis = -1), axis = -1)
    theta = tf.reduce_sum(x[:, 1:]*tYV2, axis = 1)
    v = S0[0]*x[:, 0, :, 0] + tf.reduce_sum(theta*diffS, axis = -1)
    
    def hedg_err_train(v, l, Kstr_call_tf, Kstr_put_tf):
        v1 = tf.cond(tf.cast(l < 1.5, tf.bool), lambda: [v, v], lambda: [v[:batch_size], v[batch_size:]])
        call = tf.reduce_mean(tf.square(tf.nn.relu(termS - Kstr_call_tf) - v1[0]), axis = -1)
        put = tf.reduce_mean(tf.square(tf.nn.relu(Kstr_put_tf - termS) - v1[1]), axis = -1)
        return tf.concat([call, put], axis = 0)
    
    def hedg_err_test(v, l, Kstr_call_tf, Kstr_put_tf):
        v1 = tf.cond(tf.cast(l < 1.5, tf.bool), lambda: [v, v], lambda: [v[:Ktest], v[Ktest:]])
        gap_call = tf.reduce_mean(tf.square((termS - Kstr_call_tf)*tf.cast(tf.greater_equal(termS, Kstr_put_tf), dtype = tf.float32) - v1[0]), axis = -1)
        gap_put = tf.reduce_mean(tf.square((Kstr_put_tf - termS)*tf.cast(tf.less_equal(termS, Kstr_call_tf), dtype = tf.float32) - v1[1]), axis = -1)
        return tf.concat([gap_call, gap_put], axis = 0)
    
    return tf.cond(train, lambda: hedg_err_train(v, l, Kstr_call_tf, Kstr_put_tf), lambda: hedg_err_test(v, l, Kstr_call_tf, Kstr_put_tf))

if new_par:
    Kstr_call = np.random.uniform(size = K, low = 9.0, high = 10.0)
    Kstr_put = np.random.uniform(size = K, low = 10.0, high = 11.0)
    np.savetxt("quad_hedg/Kstr_call.csv", Kstr_call)
    np.savetxt("quad_hedg/Kstr_put.csv", Kstr_put)
    Kstr_call = np.reshape(Kstr_call, [-1, 1, 1])
    Kstr_put = np.reshape(Kstr_put, [-1, 1, 1])
else:
    Kstr_call = np.loadtxt("quad_hedg/Kstr_call.csv", dtype = np.float32)
    Kstr_put = np.loadtxt("quad_hedg/Kstr_put.csv", dtype = np.float32)
    Kstr_call = np.reshape(Kstr_call, [-1, 1, 1])
    Kstr_put = np.reshape(Kstr_put, [-1, 1, 1])

L = 10
M = 40
gamma = 0.5

ep = 5000
eval_every = 125
lr = 5e-5
batch_size = 50
nr_batch = int(Ktrain/batch_size)
print_details = True

ind_train = np.arange(Ktrain)
ind_train2 = np.union1d(np.arange(Ktrain), np.arange(K, K+Ktrain))
ind_test = np.arange(Ktrain, K)
ind_test2 = np.union1d(np.arange(Ktrain, K), np.arange(K+Ktrain, 2*K))
init = tf.random_normal_initializer(stddev = 0.001)

xl = tf.random.normal(shape = [1, R, 1], stddev = 0.001)
train_tf = tf.placeholder(shape = (), dtype = tf.bool)
Kstr_call_tf = tf.placeholder(shape = (None, 1, 1), dtype = tf.float32)
Kstr_put_tf = tf.placeholder(shape = (None, 1, 1), dtype = tf.float32)
for l in range(1, L+1):
    Al = tf.Variable(initial_value = init(shape = (1, R, R)), dtype = tf.float32)
    Bl = tf.Variable(initial_value = init(shape = (1, R, M)), dtype = tf.float32)
    xlm = tf.Variable(initial_value = init(shape = (1, R, M)), dtype = tf.float32)
    bl = tf.Variable(initial_value = init(shape = (1, R, 1)), dtype = tf.float32)
    gout = g(xl + xlm, l, train_tf, Kstr_call_tf, Kstr_put_tf)
    hid = tf.matmul(Al, xl) + tf.matmul(Bl, tf.expand_dims(gout, axis = -1)) + bl
    xl = gamma*xl + (1.0-gamma)*prox_f(hid)
    
AL1 = tf.Variable(initial_value = init(shape = (1, R, R)), dtype = tf.float32)
xL1 = prox_f(tf.matmul(AL1, xl))

price = S0[0]*xL1[:, 0, 0]
x2 = tf.expand_dims(tf.expand_dims(xL1, axis = -1), axis = -1)
theta = tf.reduce_sum(x2[:, 1:]*tYV2, axis = 1)[:, 0]

S2 = np.reshape(S[:, :-1, 0], [1, N, -1])
V2 = np.reshape(S[:, :-1, 1], [1, N, -1])
xTrue_tf = tf.placeholder(shape = (None, N, n), dtype = tf.float32)
loss1 = tf.reduce_mean(tf.square(price - xTrue_tf[:, 0, 0]))
loss2 = T*tf.reduce_mean(tf.square(tf.reduce_mean(tf.abs((theta - xTrue_tf[:, :, 1:])*mu*S2), axis = -1)))
loss3 = T*tf.reduce_mean(tf.square(theta - xTrue_tf[:, :, 1:])*tf.square(S2)*V2)
loss = loss1 + loss2 + loss3

global_step = tf.Variable(0, trainable = False)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def proj(x):
    return np.fmax(np.fmin(x, C), 0.0)

xTrue = np.zeros([2*K, N, n])
xTrue[:Ktrain, :, 0] = proj(np.tile(np.reshape(np.loadtxt("quad_hedg/price_call.csv", dtype = np.float32), [-1, 1]), [1, N]))
xTrue[:Ktrain, :, 1:] = np.reshape(np.loadtxt("quad_hedg/hedg_call.csv", dtype = np.float32), [Ktrain, N, n-1])
xTrue[Ktrain:K, :, 0] = proj(np.tile(np.reshape(np.loadtxt("quad_hedg/price_gap_call.csv", dtype = np.float32), [-1, 1]), [1, N]))
xTrue[Ktrain:K, :, 1:] = np.reshape(np.loadtxt("quad_hedg/hedg_gap_call.csv", dtype = np.float32), [Ktest, N, n-1])
xTrue[K:(K+Ktrain), :, 0] = proj(np.tile(np.reshape(np.loadtxt("quad_hedg/price_put.csv", dtype = np.float32), [-1, 1]), [1, N]))
xTrue[K:(K+Ktrain), :, 1:] = np.reshape(np.loadtxt("quad_hedg/hedg_put.csv", dtype = np.float32), [Ktrain, N, n-1])
xTrue[(K+Ktrain):, :, 0] = proj(np.tile(np.reshape(np.loadtxt("quad_hedg/price_gap_put.csv", dtype = np.float32), [-1, 1]), [1, N]))
xTrue[(K+Ktrain):, :, 1:] = np.reshape(np.loadtxt("quad_hedg/hedg_gap_put.csv", dtype = np.float32), [Ktest, N, n-1])

xTrue[:Ktrain, :, 1:] = np.fmax(np.fmin(xTrue[:Ktrain, :, 1:], 1.0), 0.0)
xTrue[K:(K+Ktrain), :, 1:] = np.fmax(np.fmin(xTrue[K:(K+Ktrain):, :, 1:], 0.0), -1.0)
xTrue[Ktrain:(K+Ktrain), :, 1:] = np.fmax(np.fmin(xTrue[Ktrain:(K+Ktrain), :, 1:], 2.0), -1.0)
xTrue[(K+Ktrain):, :, 1:] = np.fmax(np.fmin(xTrue[(K+Ktrain):, :, 1:], 1.0), -2.0)

res_loss = np.nan*np.ones([ep, 2])
ind_plot = np.random.choice(np.arange(Ktest), 2)
for i in range(ep):
    begin = time.time()
    np.random.shuffle(ind_train)
    loss1 = np.zeros(nr_batch)
    for r in range(nr_batch):
        ind_batch = ind_train[(r*batch_size):((r+1)*batch_size)]
        ind_batch2 = np.union1d(ind_batch, K+ind_batch)
        feed_dict = {train_tf: True, Kstr_call_tf: Kstr_call[ind_batch], Kstr_put_tf: Kstr_put[ind_batch], xTrue_tf: xTrue[ind_batch2]}
        _, loss1[r], pr, th = sess.run([train_op, loss, price, theta], feed_dict)
    
    res_loss[i, 0] = np.mean(loss1)
    end = time.time()
    if print_details:
        print("Step {}, time {}s, loss {:g}, call_p {:5.2f} & {:5.2f}, call_h {:5.2f} & {:5.2f}, put_p {:5.2f} & {:5.2f}, put_h {:5.2f} & {:5.2f}".format(i+1, round(end-begin, 1), res_loss[i, 0], xTrue[0, 0, 0], pr[0], xTrue[0, 0, 1], th[0, 0, 0], xTrue[K, 0, 0], pr[batch_size], xTrue[K, 0, 1], th[batch_size, 0, 0]))
        
    if i == 0 or (i+1) % eval_every == 0:
        begin = time.time()
        feed_dict = {train_tf: False, Kstr_call_tf: Kstr_call[ind_test], Kstr_put_tf: Kstr_put[ind_test], xTrue_tf: xTrue[ind_test2]}
        res_loss[i, 1], pr, th = sess.run([loss, price, theta], feed_dict)
        end = time.time()
        if print_details:
            print("\nEvaluation on test data:")
            print("Step {}, time {}s, loss {:g}, gcall_p {:5.2f} & {:5.2f}, gcall_h {:5.2f} & {:5.2f}, gput_p {:5.2f} & {:5.2f}, gput_h {:5.2f} & {:5.2f}".format(i+1, round(end-begin, 1), res_loss[i, 1], xTrue[Ktrain, 0, 0], pr[0], xTrue[Ktrain, 0, 1], th[0, 0, 0], xTrue[K+Ktrain, 0, 0], pr[Ktest], xTrue[K+Ktrain, 0, 1], th[Ktest, 0, 0]))
            print("")
            plt.plot(res_loss[:, 0], "-b", label = "Train")
            plt.plot(res_loss[:, 1], "og", label = "Test", markersize = 5)
            plt.ylim([0.0, 1.02*np.nanmax(res_loss)])
            plt.xlabel("Epochs")
            plt.ylabel("MSE")
            plt.legend(loc = "upper right")
            plt.show()
            
            plt.plot(tt[:-1], xTrue[Ktrain+ind_plot[0], 0, 1:], linestyle = "solid", color = "black")
            plt.plot(tt[:-1], th[ind_plot[0], 0], linestyle = "dotted", color = "red")
            plt.plot(np.nan, np.nan, linestyle = None, alpha = 0.0)
            plt.plot(tt[:-1], xTrue[K+Ktrain+ind_plot[1], 0, 1:], linestyle = "solid", color = "dimgray")
            plt.plot(tt[:-1], th[Ktest+ind_plot[1], 0], linestyle = "dotted", color = "darkorange")
            plt.plot(np.nan, np.nan, linestyle = None, alpha = 0.0)
            lines = plt.gca().get_lines()
            plt.ylim([-1.8, 1.8])
            plt.xlabel("$t$")
            plt.ylabel("$\\theta_t(\\omega)$")
            legend1 = plt.legend([lines[k] for k in range(0, 3)], ["True,    price $x = " + "{:4.3f}".format(xTrue[Ktrain+ind_plot[0], 0, 0]) + "$",
                                                                   "Predict, price $x = " + "{:4.3f}".format(pr[ind_plot[0]]) + "$", "Gap Call"], 
                                 ncol = 2, loc = "upper left", prop = {'family': 'DejaVu Sans Mono'}, fontsize = 8)
            legend2 = plt.legend([lines[k] for k in range(3, 6)], ["True,    price $x = " + "{:4.3f}".format(xTrue[K+Ktrain+ind_plot[1], 0, 0]) + "$",
                                                                   "Predict, price $x = " + "{:4.3f}".format(pr[Ktest+ind_plot[1]]) + "$", "Gap Put"], 
                                 ncol = 2, loc = "lower left", prop = {'family': 'DejaVu Sans Mono'}, fontsize = 8)
            plt.gca().add_artist(legend1)
            plt.gca().add_artist(legend2)
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
plt.savefig("quad_hedg_loss.png", bbox_inches = 'tight', dpi = 500) 
plt.close(fig)

fig = plt.figure()
plt.plot(tt[:-1], xTrue[Ktrain+ind_plot[0], 0, 1:], linestyle = "solid", color = "black")
plt.plot(tt[:-1], th[ind_plot[0], 0], linestyle = "dotted", color = "red")
plt.plot(np.nan, np.nan, linestyle = None, alpha = 0.0)
plt.plot(tt[:-1], xTrue[K+Ktrain+ind_plot[1], 0, 1:], linestyle = "solid", color = "dimgray")
plt.plot(tt[:-1], th[Ktest+ind_plot[1], 0], linestyle = "dotted", color = "darkorange")
plt.plot(np.nan, np.nan, linestyle = None, alpha = 0.0)
lines = plt.gca().get_lines()
plt.ylim([-1.8, 1.8])
plt.xlabel("$t$")
plt.ylabel("$\\theta_t(\\omega)$")
legend1 = plt.legend([lines[k] for k in range(0, 3)], ["True,    price $x = " + "{:4.3f}".format(xTrue[Ktrain+ind_plot[0], 0, 0]) + "$",
                                                       "Predict, price $x = " + "{:4.3f}".format(pr[ind_plot[0]]) + "$", "Gap Call"], 
                     ncol = 2, loc = "upper left", prop = {'family': 'DejaVu Sans Mono'}, fontsize = 8)
legend2 = plt.legend([lines[k] for k in range(3, 6)], ["True,    price $x = " + "{:4.3f}".format(xTrue[K+Ktrain+ind_plot[1], 0, 0]) + "$",
                                                       "Predict, price $x = " + "{:4.3f}".format(pr[Ktest+ind_plot[1]]) + "$", "Gap Put"], 
                     ncol = 2, loc = "lower left", prop = {'family': 'DejaVu Sans Mono'}, fontsize = 8)
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)
plt.savefig("quad_hedg_test.png", bbox_inches = 'tight', dpi = 500)
plt.close(fig)