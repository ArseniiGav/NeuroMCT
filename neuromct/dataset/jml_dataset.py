from torch.utils.data import Dataset

from .load_data import load_processed_data


class JMLDataset(Dataset):
    def __init__(
            self,
            dataset_type: str,
            path_to_processed_data: str,
            approach_type: str,
            transform=False,
            val2_rates=False
        ):
        """
        Args:
            dataset_type (str): Type of the dataset ('training', 'val1', 'val2_1', 'val2_2', or 'val2_3').
            path_to_processed_data (str): Path to the directory containing the processed data files.
            approach_type (str): Type of the approach ('tede' or 'nfde').
            transform (callable, optional): A function/transform to apply to the spectra.
        """
        self.spectra, self.params, self.source_types = load_processed_data(
            dataset_type, path_to_processed_data, approach_type, val2_rates) 
        self.transform = transform

        # Verify data consistency
        if len(self.spectra) != len(self.params) or len(self.spectra) != len(self.source_types):
            raise ValueError("Mismatch in the lengths of spectra, params, and source_types tensors.")

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectra = self.spectra[idx]
        params = self.params[idx]
        source_types = self.source_types[idx]

        if self.transform:
            spectra = self.transform(spectra)

        return spectra, params, source_types
