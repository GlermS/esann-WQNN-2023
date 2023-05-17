import numpy as np

class DistributiveEncoder:
    def __init__(self, bins=120):
        self.bins = bins
        self.quantiles = [0 for i in range(self.bins)]
        
    def fit(self, vals):
        k = 1/self.bins
        self.quantiles = [np.quantile(vals, k*(i + 1)) for i in range(self.bins)]
        
        return self
        
    def encode(self, var):
        return np.array([1 if var >= self.quantiles[i] else 0 for i in range(self.bins)])
        

class LinearEncoder:
    def __init__(self, vmin = None, vmax = None,  bins=120):
        self.min = vmin
        self.max = vmax
        self.bins = bins
        
    def fit(self, vals):
        self.min = np.min(vals)
        self.max = np.max(vals)
        return self
        
    def encode(self, var):
        k = (self.max - self.min)/self.bins
        a = var - self.min
        return np.array([1 if i <= a//k -1 else 0 for i in range(self.bins)])
        
class PowerEncoder:
    def __init__(self, bins=120, power=2):
        self.bins = bins
        self.power = power
        self.quantiles = [0 for i in range(self.bins)]
        
    def fit(self, vals):
        self.min = np.min(vals)
        self.max = np.max(vals)
        return self
        
    def encode(self, var):
        k = (self.max**2 - self.min**2)/self.bins
        a = var**2 - self.min**2
        return np.array([1 if i <= a//k -1 else 0 for i in range(self.bins)])

class CircularEncoder:
    def __init__(self, bins=120, marker_size=3):
        self.bins = bins
        self.msize = marker_size
        self.k = None
        
    def fit(self, vals):
        self.min = np.min(vals)
        self.max = np.max(vals)
        self.k = (self.max - self.min)/self.bins
        return self
        
    def encode(self, var):
        delta = var - self.min
        i = delta//self.k
        j = i + self.msize
        
        if j <= self.bins:
            return np.array([1 if (n >= i and n < j) else 0 for n in range(self.bins)])
        else:
            j = j - self.bins
            return np.array([1 if (n >= i or n < j) else 0 for n in range(self.bins)])