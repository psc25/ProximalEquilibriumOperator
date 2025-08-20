import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

import seaborn as sns
sns.set_style("whitegrid")
sns.color_palette("deep")

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(0)
print("nonlinear_pde")

new_par = False

N = 500
Z = np.linspace(-5.0, 5.0, N).astype(np.float32)
np.savetxt("nonlinear_pde/Z.csv", Z)

yT = np.loadtxt("nonlinear_pde/sol.csv", dtype = np.float32)

R = 8
T = 1.0

Z1 = tf.reshape(Z, [1, 1, -1])
ej = [tf.pow(np.pi, -0.25)*tf.exp(-0.5*tf.square(Z1))]
ej.append(np.sqrt(2.0)*Z1*ej[0])
for j in range(2, R+1):
    ej.append(np.sqrt(2.0/j)*Z1*ej[-1] - np.sqrt((j-1)/j)*ej[-2])
    
ej1 = [-Z1*ej[0]]
for j in range(1, R):
    ej1.append(np.sqrt(0.5*j)*ej[j-1] - np.sqrt(0.5*(j+1))*ej[j+1])
    
ej = tf.stack(ej[:-1], axis = 1)
ej1 = tf.stack(ej1, axis = 1)

xi = 0.5
def prox_f(x):
    xu = tf.reduce_sum(tf.expand_dims(x, axis = -1)*ej, axis = 1, keepdims = True)
    pr = xu - 0.5*xi*tf.minimum(xu, 0.0)
    return tf.reduce_mean(pr*ej, axis = -1)
    
# Functions g
K = 500
val_split = 0.2
Ktrain = int(K*(1.0-val_split))
Ktest = K-Ktrain
def g(x, nu):
    xj1 = tf.reduce_sum(tf.expand_dims(x, axis = -1)*ej1, axis = 1)
    return 0.5*nu*tf.reduce_mean(tf.square(xj1), axis = -1)

if new_par:
    nu1 = 0.01 + 0.39*np.random.beta(a = 1.0, b = 2.0, size = K)
    np.savetxt("nonlinear_pde/nu.csv", nu1)
    nu = np.reshape(nu1, [-1, 1])
else:
    nu1 = np.loadtxt("nonlinear_pde/nu.csv", dtype = np.float32)
    nu = np.reshape(nu1, [-1, 1])

L = 10
M = 20
gamma = 0.5

ep = 20000
eval_every = 500
lr = 1e-4
print_details = True

ind_train = np.arange(Ktrain)
ind_test = np.arange(Ktrain, K)
init = tf.random_normal_initializer(stddev = 0.01)

xl = tf.random.normal(shape = [1, R, 1], stddev = 0.01)
nu_tf = tf.placeholder(shape = (None, 1), dtype = tf.float32)
yT_tf = tf.placeholder(shape = (None, N), dtype = tf.float32)
for l in range(1, L+1):
    Al = tf.Variable(initial_value = init(shape = (1, R, R)), dtype = tf.float32)
    Bl = tf.Variable(initial_value = init(shape = (1, R, M)), dtype = tf.float32)
    xlm = tf.Variable(initial_value = init(shape = (1, R, M)), dtype = tf.float32)
    bl = tf.Variable(initial_value = init(shape = (1, R, 1)), dtype = tf.float32)
    gout = g(xl + xlm, nu_tf)
    hd = tf.matmul(Al, xl) + tf.matmul(Bl, tf.expand_dims(gout, axis = -1)) + bl
    xl = gamma*xl + (1.0-gamma)*prox_f(hd)
    
AL1 = tf.Variable(initial_value = init(shape = (1, R, R)), dtype = tf.float32)
xL1 = prox_f(tf.matmul(AL1, xl))

yP = tf.reduce_sum(tf.expand_dims(xL1, axis = -1)*ej, axis = 1)[:, 0]
loss = tf.reduce_mean(tf.square(yP - yT_tf))

global_step = tf.Variable(0, trainable = False)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

res_loss = np.nan*np.ones([ep, 2])
ind_plot = np.random.choice(np.arange(Ktest), 2)
for i in range(ep):
    begin = time.time()
    feed_dict = {nu_tf: nu[ind_train], yT_tf: yT[ind_train]}
    _, res_loss[i, 0] = sess.run([train_op, loss], feed_dict)
    end = time.time()
    if print_details:
        print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 0]))
        
    if i == 0 or (i+1) % eval_every == 0:
        begin = time.time()
        feed_dict = {nu_tf: nu[ind_test], yT_tf: yT[ind_test]}
        res_loss[i, 1], y1, g1 = sess.run([loss, yP, gout], feed_dict)
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
            
            plt.plot(Z, yT[Ktest+ind_plot[0]], "-k", label = "True")
            plt.plot(Z, yT[Ktest+ind_plot[1]], "-k")
            plt.plot(Z, y1[ind_plot[0]], ":r", label = "Predict")
            plt.plot(Z, y1[ind_plot[1]], ":r")
            plt.xlabel("$u$")
            plt.ylabel("$y(T,u)$")
            plt.legend(loc = "upper right")
            plt.show()
            
fig = plt.figure()
plt.plot(res_loss[:, 0], "-b", label = "Train")
plt.plot(res_loss[:, 1], "og", label = "Test", markersize = 5)
plt.ylim([0.0, 1.02*np.nanmax(res_loss)])
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(loc = "upper right")
plt.savefig("nonlinear_pde_loss.png", bbox_inches = 'tight', dpi = 500) 
plt.close(fig)

fig = plt.figure()
plt.plot(Z, yT[Ktest+ind_plot[0]], "-k", label = "True")
plt.plot(Z, yT[Ktest+ind_plot[1]], "-k")
plt.plot(Z, y1[ind_plot[0]], ":r", label = "Predict")
plt.plot(Z, y1[ind_plot[1]], ":r")
plt.xlabel("$u$")
plt.ylabel("$y(T,u)$")
plt.legend(loc = "upper right")
plt.savefig("nonlinear_pde_test.png", bbox_inches = 'tight', dpi = 500) 
plt.close(fig)