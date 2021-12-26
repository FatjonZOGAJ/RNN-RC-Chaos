#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by:  Jaideep Pathak, University of Maryland
				Vlachas Pantelis, CSE-lab, ETH Zurich
"""
# !/usr/bin/env python
import numpy as np
import torch
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
import os
import sys


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

module_paths = [
    os.path.abspath(os.getcwd()),
    os.path.abspath(os.getcwd() + '//rc_chaos//Methods'),
]

for module_path in module_paths:
    print(module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)

from models.utils import eval_simple, eval_all_dyn_syst, eval_single_dyn_syst
from rc_chaos.Methods.RUN import new_args_dict

from rc_chaos.Methods.Models.Utils.global_utils import *
from rc_chaos.Methods.Models.Utils.plotting_utils import *
import pandas as pd
from functools import partial

print = partial(print, flush=True)

from sklearn.linear_model import Ridge
from typing import Union, Sequence, Optional
from darts import TimeSeries


class esn(GlobalForecastingModel):
    def delete(self):
        return 0

    def __init__(self, reservoir_size=1000, sparsity=0.01, radius=0.6, sigma_input=1, dynamics_fit_ratio=2 / 7,
                 regularization=0.0,
                 scaler_tt='Standard', solver='auto'):
        self.reservoir_size = reservoir_size
        self.sparsity = sparsity
        self.radius = radius
        self.sigma_input = sigma_input
        self.dynamics_fit_ratio = dynamics_fit_ratio
        self.regularization = regularization
        self.scaler_tt = scaler_tt
        self.solver = solver
        ##########################################
        self.scaler = scaler(self.scaler_tt)

    def getWeights(self, sizex, sizey, radius, sparsity):
        W = np.random.random((sizex, sizey))
        # a = np.random.choice([True, False], (sizex, sizey), True, [0.9, 0.1])
        # W[a] = 0
        eigenvalues, _ = splinalg.eigs(W)
        eigenvalues = np.abs(eigenvalues)
        W = (W / np.max(eigenvalues)) * radius
        return W

    def augmentHidden(self, h):
        h_aug = h.copy()
        h_aug[::2] = pow(h_aug[::2], 2.0)
        return h_aug

    def getAugmentedStateSize(self):
        return self.reservoir_size

    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None
            ) -> None:
        super().fit(series)

        data = np.array(series.all_values())
        train_input_sequence = data.squeeze(1)
        dynamics_length = int(len(data) * self.dynamics_fit_ratio)
        N, input_dim = np.shape(train_input_sequence)

        train_input_sequence = self.scaler.scaleData(train_input_sequence)

        W_h = self.getWeights(self.reservoir_size, self.reservoir_size, self.radius, self.sparsity)

        # Input weights
        W_in = np.zeros((self.reservoir_size, input_dim))
        q = int(self.reservoir_size / input_dim)
        for i in range(0, input_dim):
            W_in[i * q:(i + 1) * q, i] = self.sigma_input * (-1 + 2 * np.random.rand(q))

        # Training length
        tl = N - dynamics_length

        W_h = torch.tensor(W_h, requires_grad=True)
        W_in = torch.tensor(W_in, requires_grad=True)
        learning_rate = 0.01
        a0 = learning_rate
        decay = 0.95
        iterations = 1
        for iter in range(iterations):
            h = torch.zeros((self.reservoir_size, 1), dtype=torch.float64)
            # Washout phase
            for t in range(dynamics_length):
                i = np.reshape(train_input_sequence[t], (-1, 1))
                i = torch.tensor(i)
                h = torch.tanh(W_h @ h + W_in @ i)

            H = []
            Y = []

            # Training
            for t in range(tl - 1):
                i = np.reshape(train_input_sequence[t + dynamics_length], (-1, 1))
                i = torch.tensor(i, dtype=torch.float64)
                h = torch.tanh(W_h @ h + W_in @ i)
                copy = torch.clone(h)
                copy[1::2] = 1
                h_aug = h * copy
                H.append(h_aug[:, 0])
                target = np.reshape(train_input_sequence[t + dynamics_length + 1], (-1, 1))
                Y.append(target[:, 0])

            split = int(tl * 0.8)
            H_train = [torch.clone(x).detach().numpy() for x in H[:split]]
            Y_train = Y[:split]
            H_test = H[split:]
            Y_test = [torch.tensor(x) for x in Y[split:]]
            H = [torch.clone(x).detach().numpy() for x in H]

            # Learn mapping H -> Y with Ridge Regression
            if self.solver in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]:
                ridge = Ridge(alpha=self.regularization, fit_intercept=False, copy_X=True,
                              solver=self.solver)
                if iter == iterations - 1:
                    ridge.fit(H, Y)
                else:
                    ridge.fit(H_train, Y_train)
                W_out = torch.tensor(ridge.coef_)
            else:
                raise ValueError("Undefined solver.")

            loss = torch.tensor([0], dtype=torch.float64)
            for sample in range(len(H_test)):
                loss += torch.pow(W_out @ H_test[sample] - Y_test[sample], 2)

            loss.backward()

            with torch.no_grad():
                print('Loss: ' + str(loss))
                if iter == iterations - 1:
                    break
                W_in -= learning_rate * W_in.grad
                W_h -= learning_rate * W_h.grad
                W_in.grad.zero_()
                W_h.grad.zero_()
                learning_rate = decay ** iter * a0

        self.W_in = W_in.detach().numpy()
        self.W_h = W_h.detach().numpy()
        self.W_out = W_out.detach().numpy()

        self.n_trainable_parameters = np.size(self.W_out)
        self.n_model_parameters = np.size(self.W_in) + np.size(self.W_h) + np.size(self.W_out)

    def predictSequence(self, input_sequence, n):
        W_h = self.W_h
        W_out = self.W_out
        W_in = self.W_in
        N = np.shape(input_sequence)[0]
        # HAS TO BE LENGTH OF INPUT SEQUENCE TO PREDICT THE FOLLOWING STEPS N + 1, N + 2, ...
        dynamics_length = N
        iterative_prediction_length = n  # until N + n

        self.reservoir_size, _ = np.shape(W_h)

        prediction_warm_up = []
        h = np.zeros((self.reservoir_size, 1))
        for t in range(dynamics_length):
            i = np.reshape(input_sequence[t], (-1, 1))
            h = np.tanh(W_h @ h + W_in @ i)
            out = W_out @ self.augmentHidden(h)
            prediction_warm_up.append(out)

        prediction = []
        for t in range(iterative_prediction_length):
            out = W_out @ self.augmentHidden(h)
            prediction.append(out)
            i = out
            h = np.tanh(W_h @ h + W_in @ i)

        prediction = np.array(prediction)[:, :, 0]
        prediction_warm_up = np.array(prediction_warm_up)[:, :, 0]

        prediction_augment = np.concatenate((prediction_warm_up, prediction), axis=0)

        return prediction, prediction_augment

    def predict(self,
                n: int,
                series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                num_samples: int = 1,
                ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if series is None:
            series = self.training_series
        input_sequence = series.all_values().squeeze(1)  # (1000, 1)

        # TODO maybe we can average results of multiple predictions with various dynamic_fit_ratios?
        num_test_ICS = 1  # TODO self.num_test_ICS
        input_sequence = self.scaler.scaleData(input_sequence, reuse=1)
        for ic_num in range(num_test_ICS):
            # TODO: try out only once with full length
            # ic_idx = random.choice(range(self.dynamics_length, len(input_sequence) - self.iterative_prediction_length))[0]# random indexes within input_sequence_len
            # input_sequence_ic = input_sequence[ic_idx-self.dynamics_length:ic_idx+self.iterative_prediction_length]
            ic_idx = 0
            input_sequence_ic = input_sequence
            prediction, prediction_augment = self.predictSequence(input_sequence_ic, n)
            prediction = self.scaler.descaleData(prediction)
            df = pd.DataFrame(np.squeeze(prediction))
            df.index = range(len(input_sequence_ic), len(input_sequence_ic) + n)
            return TimeSeries.from_dataframe(df)


def main():
    # eval_simple(esn(**new_args_dict()))
    # np.random.seed(114)
    # value = eval_single_dyn_syst(esn(**new_args_dict()), 'Chua')
    # best_seed = -1
    # best_score = np.inf
    # for i in range(100, 200):
    #     np.random.seed(i)
    #     value = eval_single_dyn_syst(esn(**new_args_dict()), 'Chua')
    #     if value < best_score:
    #         best_seed = i
    #         best_score = value
    # print(best_seed, best_score)

    eval_all_dyn_syst(esn(**new_args_dict()))


if __name__ == '__main__':
    main()
