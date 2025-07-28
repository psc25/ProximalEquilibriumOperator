import numpy as np
import time

K = 250
N = 500
n = 100
T = 1.0
S0 = [10.0, 0.1]
dt = T/n
tt = np.linspace(0, T, n)

kappa = 4.0
theta = 0.1
sigma = 0.5
rho = -0.7
S = np.loadtxt("S.csv")
S = np.reshape(S, (N, n, 2))
Kstr_call = np.reshape(np.loadtxt("Kstr_call.csv"), (-1, 1, 1))
Kstr_put = np.reshape(np.loadtxt("Kstr_put.csv"), (-1, 1, 1))

M = 1024
umin = 1e-12
umax = 100.0
u = np.reshape(np.linspace(umin, umax, M), [1, 1, -1])
val_split = 0.2
Ktrain = int(K*(1.0-val_split))

def char_heston(u, t, vt):
    xi = kappa - sigma*rho*u*1.0j
    d = np.sqrt(xi**2 + sigma**2*(np.square(u) + 1.0j*u))
    g1 = (xi + d)/(xi - d)
    g2 = 1.0/g1
    cf = np.exp((kappa*theta)/(sigma**2)*((xi - d)*t - 2.0*np.log((1.0 - g2*np.exp(-d*t))/(1.0 - g2))) + (vt/sigma**2)*(xi - d)*(1.0 - np.exp(-d*t))/(1.0 - g2*np.exp(-d*t)))
    dcf = cf/sigma**2*(xi - d)*(1.0 - np.exp(-d*t))/(1.0 - g2*np.exp(-d*t))
    return cf, dcf

# Training set - European call option
charfct1, _ = char_heston(u[0]-1.0j, tt[0], np.reshape(S0[1], [1, 1]))
charfct2, _ = char_heston(u[0], tt[0], np.reshape(S0[1], [1, 1]))
theta1 = np.real(np.exp(-1.0j*u[0]*np.log(Kstr_call[:Ktrain, 0]/np.reshape(S0[0], [1, 1])))*charfct1/(1.0j*u[0]*np.pi))
theta2 = np.real(np.exp(-1.0j*u[0]*np.log(Kstr_call[:Ktrain, 0]/np.reshape(S0[0], [1, 1])))*charfct2/(1.0j*u[0]*np.pi))
out1 = np.trapz(theta1, x = u[0], axis = -1)
out2 = np.trapz(theta2, x = u[0], axis = -1)
price_call = S0[0]*(0.5 + out1) - Kstr_call[:Ktrain, 0, 0]*(0.5 + out2)
np.savetxt("price_call.csv", price_call)

hedg_call = np.zeros([Ktrain, N, n-1])
for k in range(n-1):
    begin = time.time()
    
    St = np.reshape(S[:, k+1, 0], [1, -1, 1])
    Vt = np.reshape(S[:, k+1, 1], [1, -1, 1])
    
    charfct1, dercharfct1 = char_heston(u-1.0j, tt[k+1], Vt)
    charfct2, dercharfct2 = char_heston(u, tt[k+1], Vt)
    
    theta1 = np.real(np.exp(-1.0j*u*np.log(Kstr_call[:Ktrain]/St))*(1.0j*u+1.0)*charfct1/(1.0j*u*np.pi))
    theta2 = np.real(np.exp(-1.0j*u*np.log(Kstr_call[:Ktrain]/St))*(1.0j*u)*charfct2/(1.0j*u*np.pi))
    theta3 = np.real(np.exp(-1.0j*u*np.log(Kstr_call[:Ktrain]/St))*dercharfct1/(1.0j*u*np.pi))
    theta4 = np.real(np.exp(-1.0j*u*np.log(Kstr_call[:Ktrain]/St))*dercharfct2/(1.0j*u*np.pi))
    
    out1 = np.trapz(theta1, x = u, axis = -1)
    out2 = np.trapz(theta2, x = u, axis = -1)
    out3 = np.trapz(theta3, x = u, axis = -1)
    out4 = np.trapz(theta4, x = u, axis = -1)
    
    dS_contr = (0.5 + out1) - Kstr_call[:Ktrain, :, 0]/St[:, :, 0]*out2
    dV_contr = (St[:, :, 0]*out3 - Kstr_call[:Ktrain, :, 0]*out4)*rho*sigma/St[:, :, 0]
    hedg_call[:, :, k] = dS_contr + dV_contr
    
    end = time.time()
    print("Step " + str(k+1) + ", time " + str(round(end-begin, 1)) + "s, hedg " + str(np.round(hedg_call[0, 0, k], 4)) + ", min " + str(np.round(np.min(hedg_call[:, :, k]), 4)) + ", max " + str(np.round(np.max(hedg_call[:, :, k]), 4)))
    
np.savetxt("hedg_call.csv", np.reshape(hedg_call, (Ktrain, -1)))
print("")

# Training set - European put option
charfct1, _ = char_heston(u[0]-1.0j, tt[0], np.reshape(S0[1], [1, 1]))
charfct2, _ = char_heston(u[0], tt[0], np.reshape(S0[1], [1, 1]))
theta1 = np.real(np.exp(-1.0j*u[0]*np.log(Kstr_put[:Ktrain, 0]/np.reshape(S0[0], [1, 1])))*charfct1/(1.0j*u[0]*np.pi))
theta2 = np.real(np.exp(-1.0j*u[0]*np.log(Kstr_put[:Ktrain, 0]/np.reshape(S0[0], [1, 1])))*charfct2/(1.0j*u[0]*np.pi))
out1 = np.trapz(theta1, x = u[0], axis = -1)
out2 = np.trapz(theta2, x = u[0], axis = -1)
price_put = Kstr_put[:Ktrain, 0, 0]*(0.5 - out2) - S0[0]*(0.5 - out1)
np.savetxt("price_put.csv", price_put)

hedg_put = np.zeros([Ktrain, N, n-1])
for k in range(n-1):
    begin = time.time()
    
    St = np.reshape(S[:, k+1, 0], [1, -1, 1])
    Vt = np.reshape(S[:, k+1, 1], [1, -1, 1])
    
    charfct1, dercharfct1 = char_heston(u-1.0j, tt[k+1], Vt)
    charfct2, dercharfct2 = char_heston(u, tt[k+1], Vt)
    
    theta1 = np.real(np.exp(-1.0j*u*np.log(Kstr_put[:Ktrain]/St))*(1.0j*u+1.0)*charfct1/(1.0j*u*np.pi))
    theta2 = np.real(np.exp(-1.0j*u*np.log(Kstr_put[:Ktrain]/St))*(1.0j*u)*charfct2/(1.0j*u*np.pi))
    theta3 = np.real(np.exp(-1.0j*u*np.log(Kstr_put[:Ktrain]/St))*dercharfct1/(1.0j*u*np.pi))
    theta4 = np.real(np.exp(-1.0j*u*np.log(Kstr_put[:Ktrain]/St))*dercharfct2/(1.0j*u*np.pi))
    
    out1 = np.trapz(theta1, x = u, axis = -1)
    out2 = np.trapz(theta2, x = u, axis = -1)
    out3 = np.trapz(theta3, x = u, axis = -1)
    out4 = np.trapz(theta4, x = u, axis = -1)
    
    dS_contr = -Kstr_put[:Ktrain, :, 0]/St[:, :, 0]*out2 - (0.5 - out1)
    dV_contr = (-Kstr_put[:Ktrain, :, 0]*out4 + St[:, :, 0]*out3)*rho*sigma/St[:, :, 0]
    hedg_put[:, :, k] = dS_contr + dV_contr
    
    end = time.time()
    print("Step " + str(k+1) + ", time " + str(round(end-begin, 1)) + "s, hedg " + str(np.round(hedg_put[0, 0, k], 4)) + ", min " + str(np.round(np.min(hedg_put[:, :, k]), 4)) + ", max " + str(np.round(np.max(hedg_put[:, :, k]), 4)))
    
np.savetxt("hedg_put.csv", np.reshape(hedg_put, (Ktrain, -1)))
print("")

# Test set - European gap call option
charfct1, _ = char_heston(u[0]-1.0j, tt[0], np.reshape(S0[1], [1, 1]))
charfct2, _ = char_heston(u[0], tt[0], np.reshape(S0[1], [1, 1]))
theta1 = np.real(np.exp(-1.0j*u[0]*np.log(Kstr_put[Ktrain:, 0]/np.reshape(S0[0], [1, 1])))*charfct1/(1.0j*u[0]*np.pi))
theta2 = np.real(np.exp(-1.0j*u[0]*np.log(Kstr_put[Ktrain:, 0]/np.reshape(S0[0], [1, 1])))*charfct2/(1.0j*u[0]*np.pi))
out1 = np.trapz(theta1, x = u[0], axis = -1)
out2 = np.trapz(theta2, x = u[0], axis = -1)
price_gap_call = S0[0]*(0.5 + out1) - Kstr_call[Ktrain:, 0, 0]*(0.5 + out2)
np.savetxt("price_gap_call.csv", price_gap_call)

hedg_gap_call = np.zeros([K-Ktrain, N, n-1])
for k in range(n-1):
    begin = time.time()
    
    St = np.reshape(S[:, k+1, 0], [1, -1, 1])
    Vt = np.reshape(S[:, k+1, 1], [1, -1, 1])
    
    charfct1, dercharfct1 = char_heston(u-1.0j, tt[k+1], Vt)
    charfct2, dercharfct2 = char_heston(u, tt[k+1], Vt)
    
    theta1 = np.real(np.exp(-1.0j*u*np.log(Kstr_put[Ktrain:]/St))*(1.0j*u+1.0)*charfct1/(1.0j*u*np.pi))
    theta2 = np.real(np.exp(-1.0j*u*np.log(Kstr_put[Ktrain:]/St))*(1.0j*u)*charfct2/(1.0j*u*np.pi))
    theta3 = np.real(np.exp(-1.0j*u*np.log(Kstr_put[Ktrain:]/St))*dercharfct1/(1.0j*u*np.pi))
    theta4 = np.real(np.exp(-1.0j*u*np.log(Kstr_put[Ktrain:]/St))*dercharfct2/(1.0j*u*np.pi))
    
    out1 = np.trapz(theta1, x = u, axis = -1)
    out2 = np.trapz(theta2, x = u, axis = -1)
    out3 = np.trapz(theta3, x = u, axis = -1)
    out4 = np.trapz(theta4, x = u, axis = -1)
    
    dS_contr = (0.5 + out1) - Kstr_call[Ktrain:, :, 0]/St[:, :, 0]*out2
    dV_contr = (St[:, :, 0]*out3 - Kstr_call[Ktrain:, :, 0]*out4)*rho*sigma/St[:, :, 0]
    hedg_gap_call[:, :, k] = dS_contr + dV_contr
    
    end = time.time()
    print("Step " + str(k+1) + ", time " + str(round(end-begin, 1)) + "s, hedg " + str(np.round(hedg_gap_call[0, 0, k], 4)) + ", min " + str(np.round(np.min(hedg_gap_call[:, :, k]), 4)) + ", max " + str(np.round(np.max(hedg_gap_call[:, :, k]), 4)))
    
np.savetxt("hedg_gap_call.csv", np.reshape(hedg_gap_call, (K-Ktrain, -1)))
print("")

# Test set - European gap put option
charfct1, _ = char_heston(u[0]-1.0j, tt[0], np.reshape(S0[1], [1, 1]))
charfct2, _ = char_heston(u[0], tt[0], np.reshape(S0[1], [1, 1]))
theta1 = np.real(np.exp(-1.0j*u[0]*np.log(Kstr_call[Ktrain:, 0]/np.reshape(S0[0], [1, 1])))*charfct1/(1.0j*u[0]*np.pi))
theta2 = np.real(np.exp(-1.0j*u[0]*np.log(Kstr_call[Ktrain:, 0]/np.reshape(S0[0], [1, 1])))*charfct2/(1.0j*u[0]*np.pi))
out1 = np.trapz(theta1, x = u[0], axis = -1)
out2 = np.trapz(theta2, x = u[0], axis = -1)
price_gap_put = Kstr_put[Ktrain:, 0, 0]*(0.5 - out2) - S0[0]*(0.5 - out1)
np.savetxt("price_gap_put.csv", price_gap_put)

hedg_gap_put = np.zeros([K-Ktrain, N, n-1])
for k in range(n-1):
    begin = time.time()
    
    St = np.reshape(S[:, k+1, 0], [1, -1, 1])
    Vt = np.reshape(S[:, k+1, 1], [1, -1, 1])
    
    charfct1, dercharfct1 = char_heston(u-1.0j, tt[k+1], Vt)
    charfct2, dercharfct2 = char_heston(u, tt[k+1], Vt)
    
    theta1 = np.real(np.exp(-1.0j*u*np.log(Kstr_call[Ktrain:]/St))*(1.0j*u+1.0)*charfct1/(1.0j*u*np.pi))
    theta2 = np.real(np.exp(-1.0j*u*np.log(Kstr_call[Ktrain:]/St))*(1.0j*u)*charfct2/(1.0j*u*np.pi))
    theta3 = np.real(np.exp(-1.0j*u*np.log(Kstr_call[Ktrain:]/St))*dercharfct1/(1.0j*u*np.pi))
    theta4 = np.real(np.exp(-1.0j*u*np.log(Kstr_call[Ktrain:]/St))*dercharfct2/(1.0j*u*np.pi))
    
    out1 = np.trapz(theta1, x = u, axis = -1)
    out2 = np.trapz(theta2, x = u, axis = -1)
    out3 = np.trapz(theta3, x = u, axis = -1)
    out4 = np.trapz(theta4, x = u, axis = -1)
    
    dS_contr = -Kstr_put[Ktrain:, :, 0]/St[:, :, 0]*out2 - (0.5 - out1)
    dV_contr = (-Kstr_put[Ktrain:, :, 0]*out4 + St[:, :, 0]*out3)*rho*sigma/St[:, :, 0]
    hedg_gap_put[:, :, k] = dS_contr + dV_contr
    
    end = time.time()
    print("Step " + str(k+1) + ", time " + str(round(end-begin, 1)) + "s, hedg " + str(np.round(hedg_gap_put[0, 0, k], 4)) + ", min " + str(np.round(np.min(hedg_gap_put[:, :, k]), 4)) + ", max " + str(np.round(np.max(hedg_gap_put[:, :, k]), 4)))
    
np.savetxt("hedg_gap_put.csv", np.reshape(hedg_gap_put, (K-Ktrain, -1)))