import pickle

def load_minimax_scaler(base_path_to_models):
    return pickle.load(open(f"{base_path_to_models}/models/minmax_scaler.pkl", "rb"))
