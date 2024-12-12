import torch
from torch.utils.data import Dataset
from neuromct.configs import data_configs
from .val2_data_rates_processing import get_val2_data_rates


class JMLDataset(Dataset):
    def __init__(
            self,
            dataset_type: str,
            path_to_processed_data: str,
            transform=None,
            val2_rates=None
        ):
        """
        Args:
            dataset_type (str): Type of the dataset ('training', 'val1', 'val2_1', 'val2_2', or 'val2_3').
            path_to_processed_data (str): Path to the directory containing the processed data files.
            transform (callable, optional): A function/transform to apply to the spectra.
        """
        if dataset_type in ["training", "val1", "val2_1", "val2_2", "val2_3"]:
            spectra_path = f"{path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_spectra.pt"
            params_path = f"{path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_params.pt"
            source_types_path = f"{path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_source_types.pt"
        else:
            raise ValueError("Invalid dataset_type! Choose between 'training', 'val1', 'val2_1', 'val2_2', and 'val2_3'.")

        self.spectra = torch.load(spectra_path, weights_only=True)
        self.params = torch.load(params_path, weights_only=True)
        self.source_types = torch.load(source_types_path, weights_only=True)
        self.transform = transform

        if dataset_type in ["val2_1", "val2_2", "val2_3"]:
            if val2_rates:
                val2_data = (self.spectra, self.params, self.source_types)
                val2_data_rates = get_val2_data_rates(val2_data)
                self.spectra, self.params, self.source_types = val2_data_rates

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
