import time
import random
import darts
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from scipy.linalg import pinv as scipypinv
import os
import sys

module_paths = [
    os.path.abspath(os.getcwd()),
    os.path.abspath(os.getcwd() + '//rc_chaos//Methods'),
]

for module_path in module_paths:
    print(module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)

# TODO: fix this so that it works from command line and PyCharm Debugger
from rc_chaos.Methods.Models.esn.cells import get_cell
from rc_chaos.Methods.Models.esn.cells.rnn_cell import RNNCell
from sklearn.base import BaseEstimator


from models.utils import eval_simple, eval_all_dyn_syst, set_seed, eval_single_dyn_syst
from models.utils_filip import eval_all_dyn_syst_filip

from rc_chaos.Methods.RUN import new_args_dict

from rc_chaos.Methods.Models.Utils.global_utils import *
from rc_chaos.Methods.Models.Utils.plotting_utils import *
import pandas as pd
from functools import partial

print = partial(print, flush=True)

from sklearn.linear_model import Ridge
from typing import Union, Sequence, Optional
from darts import TimeSeries

DYNAMICS_FIT_RATIOS = [2/7, 3/7, 4/7]

# TODO: RegressorModel or GlobalForecastingModel ?
class esn(GlobalForecastingModel, BaseEstimator):
    def delete(self):
        return 0

    def __init__(self, cell_type="ESN", reservoir_size=1000, sparsity=0.1, radius=0.5, sigma_input=1,
                 dynamics_fit_ratio=2/7, regularization=0.1, scaler_tt='Standard', solver="auto", model_name='RC-CHAOS-ESN',
                 seed=1, W_scaling=1, flip_sign=False, ensemble_base_model=False):
        
        self.ensemble_base_model = ensemble_base_model      # return array instead of TimeSeries
        self.dynamics_fit_ratio = dynamics_fit_ratio
        self.regularization = regularization
        self.scaler_tt = scaler_tt
        self.solver = solver
        self.seed = seed
        self.model_name = model_name

        # need to store for copying objects
        self.W_scaling = W_scaling
        self.flip_sign = flip_sign
        self.radius = radius
        self.reservoir_size = reservoir_size
        self.radius = radius
        self.sparsity = sparsity
        self.sigma_input = sigma_input
        
        self.cell_type = cell_type
        self._cell = get_cell(cell_type, reservoir_size, radius, sparsity, sigma_input, W_scaling, flip_sign)
        
        self.scaler = scaler(self.scaler_tt)
        #set_seed(seed)
        self._estimator_type = 'regressor' # for VotingRegressor

    
    def augmentHidden(self, h):
        h_aug = h.copy()
        h_aug[::2] = pow(h_aug[::2], 2.0)
        return h_aug

    def getAugmentedStateSize(self):
        return self._cell.reservoir_size

    # TODO: also implement with separate validation train data
    def find_best_initial_weights(self, y_train, y_val, n_tries=10):
        
        min_smape = 1000000
        min_smape_model = None

        y_train_ts = TimeSeries.from_dataframe(pd.DataFrame(y_train))
        y_val_ts = TimeSeries.from_dataframe(pd.DataFrame(y_val))
                
        for i in range(50):
            
            hyperparams = self._cell.sample_set_hyperparams()
            
            if hyperparams['reservoir_size'] == 100:
                
                self.regularization = 0.01
            
            elif hyperparams['reservoir_size'] == 500: 
                
                self.regularization = 0.1
                
            else:
                
                self.regularization = 1.0
                
            self._cell.resample()       
            self.dynamics_fit_ratio = random.choice(DYNAMICS_FIT_RATIOS)
            
            self.fit(y_train_ts)

            y_val_pred = self.predict(len(y_val))
            y_val_pred = np.squeeze(y_val_pred.values())

            pred_y = TimeSeries.from_dataframe(pd.DataFrame(y_val_pred))
            true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)))

            metric_func = getattr(darts.metrics.metrics, 'smape')
            score = metric_func(true_y, pred_y)
            if score < min_smape:
                
                min_dynamics_fit_ratio = self.dynamics_fit_ratio
                min_hyperparams = hyperparams
                min_smape = score
                min_W_in = self._cell.W_in
                min_W_h  = self._cell.W_h

        print(min_hyperparams, "dynamics_fit_ratio", min_dynamics_fit_ratio)
        self.dynamics_fit_ratio = min_dynamics_fit_ratio
        self._cell.set_hyperparams(min_hyperparams)
        self._cell.fix_weights(min_W_in, min_W_h)
        self.regularization = 0.01
        
        
    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None
            ) -> None:
        
        super().fit(series)
        data = np.array(series.all_values())
        train_data = data.squeeze(1)
        dynamics_length = int(len(train_data) * self.dynamics_fit_ratio)
        N, input_dim = np.shape(train_data)

        train_data = self.scaler.scaleData(train_data)
                    
        # TRAINING LENGTH
        tl = N - dynamics_length
        
        self._cell.reset()
        for t in range(dynamics_length):
            self._cell.forward(train_data[t])

        if self.solver == "pinv":
            NORMEVERY = 10
            HTH = np.zeros((self.getAugmentedStateSize(), self.getAugmentedStateSize()))
            YTH = np.zeros((input_dim, self.getAugmentedStateSize()))
        H = []
        Y = []
        
        

        for t in range(tl - 1):
            h = self._cell.forward(train_data[t + dynamics_length])

            # AUGMENT THE HIDDEN STATE
            h_aug = self.augmentHidden(h)
            H.append(h_aug[:, 0])
            target = np.reshape(train_data[t + dynamics_length + 1], (-1, 1))
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
            pinv_ = scipypinv(HTH + self.regularization * I)
            self.W_out = YTH @ pinv_
        
        elif self.solver in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]:
            """
            Learns mapping H -> Y with Ridge Regression
            """
            ridge = Ridge(alpha=self.regularization, fit_intercept=False, copy_X=True,
                          solver=self.solver)
            ridge.fit(H, Y)
            self.W_out = ridge.coef_
        else:
            raise ValueError("Undefined solver.")
           

    def predictSequence(self, input_sequence, n):
        N = np.shape(input_sequence)[0]
        dynamics_length = N
        iterative_prediction_length = n

        prediction_warm_up = []
        self._cell.reset()
        for t in range(dynamics_length):
            h = self._cell.forward(input_sequence[t])
            out = self.W_out @ self.augmentHidden(h)
            prediction_warm_up.append(out)

        prediction = []
        for t in range(iterative_prediction_length):
            out = self.W_out @ self.augmentHidden(h)
            prediction.append(out)
            h = self._cell.forward(out)

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
            if self.ensemble_base_model:
                return np.squeeze(prediction)
            df = pd.DataFrame(np.squeeze(prediction))
            df.index = range(len(input_sequence_ic), len(input_sequence_ic) + n)
            return TimeSeries.from_dataframe(df)


# TODO: evaluate different seeds and calculate average rank
# TODO: calculate eigenvalues for those matrices and then plot in the landscape
# for which dyn systs those perform best, are there any characteristics
def main():
    model_name = 'RC-CHAOS-ESN'
    eval_all_dyn_syst_filip(esn())

    #kwargs = new_args_dict()
    #kwargs['model_name'] = model_name
    #eval_simple(esn(**kwargs, resample=True))
    #eval_simple(esn(**kwargs, resample=False))

    #start_time = time.time()
    # best Rank 5.5 with torch
    #kwargs['cell_type'] = 'ESN_torch'
    #eval_all_dyn_syst(esn(**kwargs, resample=False))
    #print(f'Eval all took {time.time() - start_time} seconds')
    #return


    # Resampling can be a lot better
    #eval_single_dyn_syst(esn(**kwargs, resample=True), 'Chua')
    #eval_single_dyn_syst(esn(**kwargs, resample=False), 'Chua')

    # or slightly worse
    #eval_single_dyn_syst(esn(**kwargs, resample=True), 'CellCycle')
    #eval_single_dyn_syst(esn(**kwargs, resample=False), 'CellCycle')

    # but is actually on average worse...
    #eval_all_dyn_syst(esn(**kwargs, resample=True))
    #eval_all_dyn_syst(esn(**kwargs, resample=False))
    



if __name__ == '__main__':
    main()
