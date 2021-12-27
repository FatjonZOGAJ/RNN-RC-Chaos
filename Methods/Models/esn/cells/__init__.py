from rc_chaos.Methods.Models.esn.cells.esn_cell import ESNCell as ESNCell_numpy
from rc_chaos.Methods.Models.esn.cells.esn_cell_torch import ESNCell as ESNCell_torch
from rc_chaos.Methods.Models.esn.cells.gru_cell import GRUCell
from rc_chaos.Methods.Models.esn.cells.lstm_cell import LSTMCell
from rc_chaos.Methods.Models.esn.cells.rnn_cell import RNNCell


def get_cell(type, reservoir_size, radius, sparsity, sigma_input):
    if type == 'GRU':
        return GRUCell(reservoir_size)
    elif type == 'LSTM':
        return LSTMCell(reservoir_size)
    elif type == 'RNN':
        return RNNCell(reservoir_size)
    elif type == 'ESN':
        return ESNCell_numpy(reservoir_size, radius, sparsity, sigma_input)
    elif type == 'ESN_torch':
        from esn_cell_torch import ESNCell
        return ESNCell_torch(reservoir_size, radius, sparsity, sigma_input)
    else:
        raise RuntimeError('Unknown cell type.')
