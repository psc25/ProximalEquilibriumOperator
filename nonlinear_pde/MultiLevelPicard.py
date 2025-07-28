import numpy as np

class MLP_model():
    def __init__(self, M, T, nu, q, y0, dtype = np.float32):
        self.M = M
        self.T = T
        self.nu = nu
        self.q = q
        self.y0 = y0
        self.dtype = dtype
    
    def compute(self, t, x, n):
        if n == 0:
            return 0.0
        else:
            Mn = np.power(self.M, n)
            W = np.random.normal(size = [1, Mn], scale = np.sqrt(self.T-t)).astype(self.dtype)
            X = x + np.sqrt(self.nu)*W
            u = np.mean(self.y0(X), axis = -1)
            
            for l in range(n):
                Mnl = np.power(self.M, n-l)
                b = 0.0
                for i in range(Mnl):
                    R = np.random.uniform(size = 1, low = t, high = self.T).astype(self.dtype)
                    W = np.random.normal(size = [1, 1], scale = np.sqrt(R-t)).astype(self.dtype)
                    Y = x + np.sqrt(self.nu)*W
                    b = b + self.q(self.compute(R, Y, l))
                    if l > 0:
                        b = b - self.q(self.compute(R, Y, l-1)) 
                        
                u = u + (self.T-t)*b/Mnl
                     
            return u