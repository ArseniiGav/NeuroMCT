from neuromct.configs import data_configs
from neuromct.dataset import PoissonNoise, NormalizeToUnity, JMLDataset
from torchvision import transforms


def create_dataset(dataset_type, transform=None, val2_rates=None):
    return JMLDataset(
        dataset_type=dataset_type,
        path_to_processed_data=data_configs['path_to_processed_data'],
        transform=transform, 
        val2_rates=val2_rates
    )


def define_transformations(dataset_type):
    if dataset_type == 'training':
        data_transform = transforms.Compose([
            PoissonNoise(),     # Apply Poisson resampling
            NormalizeToUnity()  # Normalize to unity
        ])
    elif dataset_type == 'val':
        data_transform = transforms.Compose([
            NormalizeToUnity()  # Normalize to unity
        ])
    return data_transform
