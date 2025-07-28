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
print("min_op")

d = 2
R = d

def prox_f(x):
    return tf.math.maximum(tf.math.minimum(x, 1.0), -1.0)

# function g: R^d -> (-inf,inf)
K = 10000
val_split = 0.1
Ktrain = int(K*(1.0-val_split))
Ktest = K-Ktrain
def g(x, A, b, c, logexpb, logexpc):
    y1 = 0.5*tf.reduce_sum(tf.matmul(A, x)*x, axis = 1)
    y2 = tf.reduce_sum(b*x, axis = 1)
    y3 = c[:, 0]
    z1 = tf.log(tf.reduce_sum(tf.exp(logexpb*x), axis = 1) + logexpc[:, 0])
    return y1 + y2 + y3 + z1

gA1 = np.random.normal(size = [Ktrain, d, d], scale = 1.0)
gA = np.matmul(np.transpose(gA1, [0, 2, 1]), gA1).astype(np.float32)
gA = np.concatenate([gA, np.zeros([Ktest, d, d], dtype = np.float32)], axis = 0)
gb = np.random.normal(size = [Ktrain, d, 1], scale = 1.0).astype(np.float32)
gb = np.concatenate([gb, np.zeros([Ktest, d, 1], dtype = np.float32)], axis = 0)
gc = np.random.normal(size = [Ktrain, 1, 1], scale = 1.0).astype(np.float32)
gc = np.concatenate([gc, np.zeros([Ktest, 1, 1], dtype = np.float32)], axis = 0)
glogexpb = np.random.exponential(size = [Ktest, d, 1], scale = 1.0).astype(np.float32)
glogexpb = np.concatenate([np.zeros([Ktrain, d, 1], dtype = np.float32), -glogexpb[:int(Ktest/2)], glogexpb[int(Ktest/2):]], axis = 0)
glogexpc = np.random.exponential(size = [Ktest, 1, 1], scale = 1.0).astype(np.float32)
glogexpc = np.concatenate([np.ones([Ktrain, 1, 1], dtype = np.float32), glogexpc], axis = 0)

L = 20
M = 20
gamma = 0.5

ep = 10000
eval_every = 250
lr = 2e-4
print_details = True

ind_train = np.arange(Ktrain)
ind_test = np.arange(Ktrain, K)
init = tf.random_normal_initializer(stddev = 0.01)

xl = tf.random.normal(shape = [1, d, 1], stddev = 0.01)
gA_tf = tf.placeholder(shape = (None, d, d), dtype = tf.float32)
gb_tf = tf.placeholder(shape = (None, d, 1), dtype = tf.float32)
gc_tf = tf.placeholder(shape = (None, 1, 1), dtype = tf.float32)
glogexpb_tf = tf.placeholder(shape = (None, d, 1), dtype = tf.float32)
glogexpc_tf = tf.placeholder(shape = (None, 1, 1), dtype = tf.float32)
for t in range(1, L+1):
    Al = tf.Variable(initial_value = init(shape = (1, R, R)), dtype = tf.float32)
    Bl = tf.Variable(initial_value = init(shape = (1, R, M)), dtype = tf.float32)
    xlm = tf.Variable(initial_value = init(shape = (1, d, M)), dtype = tf.float32)
    bl = tf.Variable(initial_value = init(shape = (1, R, 1)), dtype = tf.float32)
    gout = g(xl + xlm, gA_tf, gb_tf, gc_tf, glogexpb_tf, glogexpc_tf)
    hd = tf.matmul(Al, xl) + tf.matmul(Bl, tf.expand_dims(gout, axis = -1)) + bl
    xl = gamma*xl + (1.0-gamma)*prox_f(hd)
    
AL1 = tf.Variable(initial_value = init(shape = (1, R, R)), dtype = tf.float32)
xL1 = prox_f(tf.matmul(AL1, xl))[:, :, 0]

xTrue_tf = tf.placeholder(shape = (None, d), dtype = tf.float32)
loss = d*tf.reduce_mean(tf.square(xL1 - xTrue_tf))

global_step = tf.Variable(0, trainable = False)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

xTrue = np.zeros([K, d])
for k in range(Ktrain):
    xTrue[k] = np.fmax(np.fmin(-np.matmul(np.linalg.inv(gA[k]), gb[k])[:, 0], 1.0), -1.0)
    
for k in range(Ktrain, Ktrain + int(Ktest/2)):
    xTrue[k] = np.ones(d)
    
for k in range(Ktrain + int(Ktest/2), K):
    xTrue[k] = -np.ones(d)

res_loss = np.nan*np.ones([ep, 2])
for i in range(ep):
    begin = time.time()
    feed_dict = {gA_tf: gA[ind_train], gb_tf: gb[ind_train], gc_tf: gc[ind_train], 
                 glogexpb_tf: glogexpb[ind_train], glogexpc_tf: glogexpc[ind_train],         
                 xTrue_tf: xTrue[ind_train]}
    _, loss1 = sess.run([train_op, loss], feed_dict)
    res_loss[i, 0] = loss1
    end = time.time()
    if print_details:
        print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 0]))
        
    if i == 0 or (i+1) % eval_every == 0:
        begin = time.time()
        feed_dict = {gA_tf: gA[ind_test], gb_tf: gb[ind_test], gc_tf: gc[ind_test], 
                     glogexpb_tf: glogexpb[ind_test], glogexpc_tf: glogexpc[ind_test],
                     xTrue_tf: xTrue[ind_test]}
        res_loss[i, 1] = sess.run(loss, feed_dict)
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
            
fig = plt.figure()
plt.plot(res_loss[:, 0], "-b", label = "Train")
plt.plot(res_loss[:, 1], "og", label = "Test", markersize = 5)
plt.ylim([0.0, 1.02*np.nanmax(res_loss)])
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(loc = "upper right")
plt.savefig("min_op_loss.png", bbox_inches = 'tight', dpi = 500) 
plt.close(fig)

txt = ""
ind_plot = [0, 1, int(Ktest/2), int(Ktest/2)+1]
feed_dict = {gA_tf: gA[ind_test], gb_tf: gb[ind_test], gc_tf: gc[ind_test],
             glogexpb_tf: glogexpb[ind_test], glogexpc_tf: glogexpc[ind_test]}
xPred = sess.run(xL1, feed_dict)
for k in range(len(ind_plot)):
    txt = txt + str(Ktrain+ind_plot[k]+1) + " & " + "$\\begin{pmatrix} " + '{:.3f}'.format(xTrue[Ktrain+ind_plot[k], 0]) + " \\\ " + '{:.3f}'.format(xTrue[Ktrain+ind_plot[k], 1]) + " \end{pmatrix}$"
    txt = txt + " & " + "$\\begin{pmatrix} " + '{:.3f}'.format(xPred[ind_plot[k], 0]) + " \\\ " + '{:.3f}'.format(xPred[ind_plot[k], 1]) + " \end{pmatrix}$"
    if k+1 < len(ind_plot):
        txt = txt + " \\\[10pt] "
        
text_file = open("min_op_test.txt", "w")
text_file.write(txt)
text_file.close()