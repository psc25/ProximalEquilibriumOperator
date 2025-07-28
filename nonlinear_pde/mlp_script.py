import numpy as np
from MultiLevelPicard import MLP_model
import time

np.random.seed(0)

T = 1.0
M = 5

Z = np.reshape(np.loadtxt("Z.csv"), [-1, 1])
L = Z.shape[0]

nu = np.loadtxt("nu.csv")
K = len(nu)

xi = 0.5

def q(y):
    return xi*np.fmin(y, 0.0)

print("======================================================================")
sol = np.zeros([K, L])
for k in range(K):
    y0 = lambda u: 5.0*u*np.exp(-np.square(u))
    b = time.time()
    mlp = MLP_model(M, T, nu[k], q, y0)
    sol[k] = mlp.compute(0.0, Z, M)
    e = time.time()
    print("MLP performed in " + str(np.round(e-b, 1)) + "s for k = " + str(k+1) + "/" + str(K) + ", sol[k, 0] = " + str(sol[k, 0]))
    
print("======================================================================")
np.savetxt("sol.csv", sol)
print("MLP solutions saved")