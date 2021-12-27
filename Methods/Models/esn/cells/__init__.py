def get_cell(type, reservoir_size, radius, sparsity, sigma_input):
    if type == 'GRU':
        from cells.gru_cell import GRUCell
        return GRUCell(reservoir_size)
    elif type == 'LSTM':
        from cells.lstm_cell import LSTMCell
        return LSTMCell(reservoir_size)
    elif type == 'RNN':
        from cells.rnn_cell import RNNCell
        return RNNCell(reservoir_size)
    elif type == 'ESN':
        from cells.esn_cell import ESNCell
        return ESNCell(reservoir_size, radius, sparsity, sigma_input)
