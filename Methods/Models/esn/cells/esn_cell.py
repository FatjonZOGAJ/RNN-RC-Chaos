import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


class ESNCell():
    def __init__(self, reservoir_size, radius, sparsity, sigma_input):
        self.reservoir_size = reservoir_size
        self.W_h = self._getHiddenMatrix(reservoir_size, radius, sparsity)
        self.W_in = self._getInputMatrx(sigma_input)
        self.h = np.zeros((self.reservoir_size, 1))

    def _getHiddenMatrix(self, reservoir_size, radius, sparsity):
        W = sparse.random(reservoir_size, reservoir_size, density=sparsity)
        eigenvalues, _ = splinalg.eigs(W)
        eigenvalues = np.abs(eigenvalues)
        W = (W / np.max(eigenvalues)) * radius
        return W

    def _getInputMatrx(self, sigma_input):
        # Initializing the input weights
        W_in = np.zeros((self.reservoir_size, 1))
        q = int(self.reservoir_size / 1)
        for i in range(0, 1):
            W_in[i * q:(i + 1) * q, i] = sigma_input * (-1 + 2 * np.random.rand(q))
        return W_in

    def forward(self, input):
        i = np.reshape(input, (-1, 1))
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ i)
        return self.h

    def reset(self):
        self.h = np.zeros((self.reservoir_size, 1))
