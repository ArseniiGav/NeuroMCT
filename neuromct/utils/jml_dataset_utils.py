from torchvision import transforms

from ..dataset import JMLDataset
from ..dataset import PoissonNoise
from ..dataset import BuildPDF

def create_dataset(dataset_type, path_to_processed_data, transform=False, val2_rates=False):
    return JMLDataset(
        dataset_type=dataset_type,
        path_to_processed_data=path_to_processed_data,
        transform=transform, 
        val2_rates=val2_rates
    )

def define_transformations(dataset_type, Δx):
    if dataset_type == 'training':
        data_transform = transforms.Compose([
            PoissonNoise(), # Apply Poisson resampling
            BuildPDF(Δx)  # Build a PDF
        ])
    elif dataset_type == 'val':
        data_transform = transforms.Compose([
            BuildPDF(Δx)  # Build a PDF
        ])
    return data_transform
