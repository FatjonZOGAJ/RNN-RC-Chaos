#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by:  Jaideep Pathak, University of Maryland
				Vlachas Pantelis, CSE-lab, ETH Zurich
"""
# !/usr/bin/env python
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
import os
import sys

# TODO: fix this so that it works from command line and PyCharm Debugger
module_paths = [
    os.path.abspath(os.getcwd()),
    os.path.abspath(os.getcwd() + '//rc_chaos//Methods'),
]

for module_path in module_paths:
    print(module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)

from models.utils import eval_simple, eval_all_dyn_syst
from rc_chaos.Methods.RUN import new_args_dict

from rc_chaos.Methods.Models.Utils.global_utils import *
from rc_chaos.Methods.Models.Utils.plotting_utils import *
import pandas as pd
from functools import partial

print = partial(print, flush=True)

from sklearn.linear_model import Ridge
from typing import Union, Sequence, Optional
from darts import TimeSeries

# TODO: RegressorModel or GlobalForecastingModel ?
class esn(GlobalForecastingModel):
    def delete(self):
        return 0

    # -
    # display_output
    # worker_id (seed)
    # input_dim
    # N_USED
    # self.iterative_prediction_length = kwargs["iterative_prediction_length"]
    # self.num_test_ICS = kwargs["num_test_ICS"]
    # self.train_data_path = kwargs["train_data_path"]
    # self.test_data_path = kwargs["test_data_path"]
    # self.fig_dir = kwargs["fig_dir"]
    # self.model_dir = kwargs["model_dir"]
    # self.logfile_dir = kwargs["logfile_dir"]
    # self.write_to_log = kwargs["write_to_log"]

    # +
    # approx_reservoir_size -> reservoir_size (1000)
    # degree -> sparsity (0.01)
    # radius 0.6
    # sigma_input 1
    # dynamics_length -> dynamics_fit_ratio 2/7
    def __init__(self, reservoir_size=1000, sparsity=0.01, radius=0.6, sigma_input=1, dynamics_fit_ratio=2/7, regularization=0.0,
                 scaler_tt='Standard', solver='auto'):
        # TODO: global random seed
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

    def getSparseWeights(self, sizex, sizey, radius, sparsity):
        # W = np.zeros((sizex, sizey))
        # Sparse matrix with elements between 0 and 1
        # WEIGHT INIT
        W = sparse.random(sizex, sizey, density=sparsity)
        # W = sparse.random(sizex, sizey, density=sparsity, random_state=worker_id, data_rvs=np.random.randn)
        # Sparse matrix with elements between -1 and 1
        # W.data *=2
        # W.data -=1
        # to print the values do W.A
        # EIGENVALUE DECOMPOSITION
        eigenvalues, eigvectors = splinalg.eigs(W)
        eigenvalues = np.abs(eigenvalues)
        W = (W / np.max(eigenvalues)) * radius
        return W

    def augmentHidden(self, h):
        h_aug = h.copy()
        # h_aug = pow(h_aug, 2.0)
        # h_aug = np.concatenate((h,h_aug), axis=0)
        h_aug[::2] = pow(h_aug[::2], 2.0)
        return h_aug

    def getAugmentedStateSize(self):
        return self.reservoir_size

    # def augmentHidden(self, h):
    #     h_aug = h.copy()
    #     h_aug = pow(h_aug, 2.0)
    #     h_aug = np.concatenate((h,h_aug), axis=0)
    #     return h_aug
    # def getAugmentedStateSize(self):
    #     return 2*self.reservoir_size

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

        W_h = self.getSparseWeights(self.reservoir_size, self.reservoir_size, self.radius, self.sparsity)

        # Initializing the input weights
        W_in = np.zeros((self.reservoir_size, input_dim))
        q = int(self.reservoir_size / input_dim)
        for i in range(0, input_dim):
            W_in[i * q:(i + 1) * q, i] = self.sigma_input * (-1 + 2 * np.random.rand(q))

        # TRAINING LENGTH
        tl = N - dynamics_length

        # H_dyn = np.zeros((dynamics_length, self.getAugmentedStateSize(), 1))
        h = np.zeros((self.reservoir_size, 1))
        for t in range(dynamics_length):
            i = np.reshape(train_input_sequence[t], (-1, 1))
            h = np.tanh(W_h @ h + W_in @ i)
        # H_dyn[t] = self.augmentHidden(h)

        if self.solver == "pinv":
            NORMEVERY = 10
            HTH = np.zeros((self.getAugmentedStateSize(), self.getAugmentedStateSize()))
            YTH = np.zeros((input_dim, self.getAugmentedStateSize()))
        H = []
        Y = []

        # TRAINING: Teacher forcing...
        for t in range(tl - 1):
            i = np.reshape(train_input_sequence[t + dynamics_length], (-1, 1))
            h = np.tanh(W_h @ h + W_in @ i)
            # AUGMENT THE HIDDEN STATE
            h_aug = self.augmentHidden(h)
            H.append(h_aug[:, 0])
            target = np.reshape(train_input_sequence[t + dynamics_length + 1], (-1, 1))
            Y.append(target[:, 0])
            if self.solver == "pinv" and (t % NORMEVERY == 0):
                # Batched approach used in the pinv case
                H = np.array(H)
                Y = np.array(Y)
                HTH += H.T @ H
                YTH += Y.T @ H
                H = []
                Y = []

        if self.solver == "pinv" and (len(H) != 0):
            # ADDING THE REMAINING BATCH
            H = np.array(H)
            Y = np.array(Y)
            HTH += H.T @ H
            YTH += Y.T @ H

        if self.solver == "pinv":
            """
            Learns mapping H -> Y with Penrose Pseudo-Inverse
            """
            I = np.identity(np.shape(HTH)[1])
            pinv_ = scipypinv2(HTH + self.regularization * I)
            W_out = YTH @ pinv_

        elif self.solver in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]:
            """
            Learns mapping H -> Y with Ridge Regression
            """
            ridge = Ridge(alpha=self.regularization, fit_intercept=False, normalize=False, copy_X=True,
                          solver=self.solver)
            ridge.fit(H, Y)
            W_out = ridge.coef_
        else:
            raise ValueError("Undefined solver.")

        # FINALISING WEIGHTS...
        self.W_in = W_in
        self.W_h = W_h
        self.W_out = W_out

        # COMPUTING PARAMETERS...
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
    eval_simple(esn(**new_args_dict()))
    eval_all_dyn_syst(esn(**new_args_dict()))


if __name__ == '__main__':
    main()
