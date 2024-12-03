import uproot
import numpy as np
from tqdm import tqdm


def load_raw_data(path_to_raw_data, source, dataset_type, n_points, bins):
    NPEs_counts_list = []
    for i in tqdm(range(n_points)):
        reco_file = uproot.open(f"{path_to_raw_data}/{source}/{dataset_type}/reco/reco-{i}.root")
        NPEs = 1.07 * np.array(reco_file['TRec']['m_NPE'].array(), dtype=np.float64) / 1000.
        NPEs_counts, _ = np.histogram(NPEs, bins=bins)
        NPEs_counts = NPEs_counts.reshape(-1, 1)        
        NPEs_counts_list.append(NPEs_counts)

    NPEs_counts_array = np.concatenate(NPEs_counts_list, axis=1, dtype=np.float64)
    return NPEs_counts_array.T
