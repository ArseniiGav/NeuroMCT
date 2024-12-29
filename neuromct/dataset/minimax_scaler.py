from sklearn.preprocessing import MinMaxScaler
import pickle

def fit_minimax_scaler(path_to_models, data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    pickle.dump(scaler, open(f'{path_to_models}/minmax_scaler.pkl', 'wb'))

def load_minimax_scaler(path_to_models):
    return pickle.load(open(f"{path_to_models}/minmax_scaler.pkl", "rb"))
