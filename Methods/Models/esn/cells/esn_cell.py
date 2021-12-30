import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import copy

class ESNCell():
    def __init__(self, reservoir_size, radius, sparsity, sigma_input, W_scaling=1, flip_sign=False):
        
        self.reservoir_size = reservoir_size
        self.W_scaling = W_scaling
        self.flip_sign = flip_sign
  
        self.resample(reservoir_size, radius, sparsity, sigma_input)
        self.h = np.zeros((self.reservoir_size, 1))


    def _getHiddenMatrix(self, reservoir_size, radius, sparsity):
        
        success = False
        counter = 5
        
        while not success and counter>=0:
            
            try:
                W = sparse.random(reservoir_size, reservoir_size, density=sparsity) # TODO Maybe Make the matrix symmetric?
                W *= self.W_scaling 

                eigenvalues, eigvectors = splinalg.eigs(W)
                eigenvalues = np.abs(eigenvalues)

                W = (W / np.max(eigenvalues)) * radius
                success = True
                if self.flip_sign:

                    W *= (np.random.binomial(1, 0.5, (sizex, sizey)) - 0.5)*2

            except:
                sucess = False
                counter -= 1
        if counter < 0:
            
            print("-------------------- BIG CONVERGENCE PROBLEMS -----------------")
            
        return W

    def _getInputMatrx(self, sigma_input):
        # Initializing the input weights
        W_in = np.zeros((self.reservoir_size, 1))
        q = int(self.reservoir_size / 1)
        for i in range(0, 1):
            W_in[i * q:(i + 1) * q, i] = sigma_input * (-1 + 2 * np.random.rand(q))
        return W_in
    
    def resample(self, reservoir_size, radius, sparsity, sigma_input):
        
        self.W_h = self._getHiddenMatrix(reservoir_size, radius, sparsity)
        self.W_in = self._getInputMatrx(sigma_input)
        
    def fix_weights(W_in, W_h):
        
        self.W_h = copy.deepcopy(W_h)
        self.W_in = copy.deepcopy(W_in)
        
    def forward(self, input):
        i = np.reshape(input, (-1, 1))
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ i)
        return self.h

    def reset(self):
        self.h = np.zeros((self.reservoir_size, 1))
