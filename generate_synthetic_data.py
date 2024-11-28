import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic data for crop genetics, environmental factors, and yields.

    This function creates a synthetic dataset simulating the relationship 
    between genetic traits, environmental conditions, and crop yields. 
    It generates genetic and environmental features, calculates the 
    corresponding yields, and normalizes the values for better 
    performance in machine learning models.

    Parameters:
    ----------
    n_samples : int, optional
        The number of samples to generate (default is 1000).

    Returns:
    -------
    tuple
        A tuple containing:
        - genetics (numpy.ndarray): A 2D array of shape (n_samples, n_genetic_features) 
          representing genetic features, with values clipped between 0 and 1.
        - environment_normalized (numpy.ndarray): A 2D array of shape (n_samples, n_env_features) 
          representing normalized environmental features, with values scaled between 0 and 1.
        - yields_normalized (numpy.ndarray): A 1D array of shape (n_samples,) representing 
          the normalized crop yields, scaled between 0 and 1.

    Details:
    -------
    - Genetic features are generated from a normal distribution with a mean of 0.5 
      and a standard deviation of 0.15, then clipped to the range [0, 1].
    
    - Environmental features include:
        - Temperature (°C): Uniformly sampled from 15 to 35.
        - Rainfall (mm/day): Uniformly sampled from 0 to 20.
        - Solar Radiation (MJ/m²/day): Uniformly sampled from 8 to 30.
        - Soil Quality: Sampled from a beta distribution (0 to 1).
        - Relative Humidity (%): Uniformly sampled from 30 to 90.
    
    - Environmental features are normalized to the range [0, 1] for better neural 
      network performance.

    - Yields are calculated based on a base yield (centered around 6 tons/hectare) 
      plus contributions from genetic and environmental factors, with added random noise 
      to simulate variability. Yields are then normalized to the range [0, 1].

    Example:
    --------
    >>> genetics, environment, yields = generate_synthetic_data(n_samples=1000)
    """
    # Generate genetic features
    # Example genetic features: height potential, growth rate, drought resistance, etc.
    n_genetic_features = 5
    genetics = np.random.normal(0.5, 0.15, size=(n_samples, n_genetic_features))
    # Clip values to ensure they're between 0 and 1
    genetics = np.clip(genetics, 0, 1)

    # Generate environmental features
    # Features: temperature, rainfall, solar radiation, soil quality, humidity
    n_env_features = 5
    environment = np.zeros((n_samples, n_env_features))
    
    # Temperature (°C): typical range 15-35
    environment[:, 0] = np.random.uniform(15, 35, n_samples)
    # Rainfall (mm/day): typical range 0-20
    environment[:, 1] = np.random.uniform(0, 20, n_samples)
    # Solar radiation (MJ/m²/day): typical range 8-30
    environment[:, 2] = np.random.uniform(8, 30, n_samples)
    # Soil quality (index 0-1)
    environment[:, 3] = np.random.beta(5, 2, n_samples)
    # Relative humidity (%): typical range 30-90
    environment[:, 4] = np.random.uniform(30, 90, n_samples)

    # Normalize environmental features to 0-1 range for better neural network performance
    environment_normalized = np.zeros_like(environment)
    environment_normalized[:, 0] = (environment[:, 0] - 15) / (35 - 15)  # Temperature
    environment_normalized[:, 1] = environment[:, 1] / 20  # Rainfall
    environment_normalized[:, 2] = (environment[:, 2] - 8) / (30 - 8)  # Solar radiation
    environment_normalized[:, 3] = environment[:, 3]  # Soil quality (already 0-1)
    environment_normalized[:, 4] = (environment[:, 4] - 30) / (90 - 30)  # Humidity

    # Generate synthetic yields based on genetics and environment
    # Base yield (tons/hectare): typical range 2-10
    base_yield = 6 + np.random.normal(0, 1, n_samples)
    
    # Genetic effects (simplified model)
    genetic_effect = np.sum(genetics * np.array([1.5, 1.0, 0.8, 0.7, 0.5]), axis=1)
    
    # Environmental effects (simplified model)
    env_effect = (
        environment_normalized[:, 0] * 1.2 +  # Temperature effect
        environment_normalized[:, 1] * 1.5 +  # Rainfall effect
        environment_normalized[:, 2] * 1.0 +  # Solar radiation effect
        environment_normalized[:, 3] * 0.8 +  # Soil quality effect
        environment_normalized[:, 4] * 0.5    # Humidity effect
    )

    # Combine effects and add some random noise
    yields = base_yield + genetic_effect + env_effect + np.random.normal(0, 0.5, n_samples)
    # Ensure no negative yields
    yields = np.maximum(yields, 0)

    # Normalize yields to 0-1 range
    yields_normalized = (yields - yields.min()) / (yields.max() - yields.min())

    return genetics, environment_normalized, yields_normalized

class CropDataset(Dataset):
    def __init__(self, genetics, environment, yields):
        self.genetics = torch.FloatTensor(genetics)
        self.environment = torch.FloatTensor(environment)
        self.yields = torch.FloatTensor(yields)
        
    def __len__(self):
        return len(self.yields)
    
    def __getitem__(self, idx):
        return self.genetics[idx], self.environment[idx], self.yields[idx]

if __name__ == "__main__":
    # Generate synthetic data
    genetics, environment, yields = generate_synthetic_data(n_samples=1000)
    
    # Create dataset
    dataset = CropDataset(genetics, environment, yields)
    
    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Genetics shape: {dataset.genetics.shape}")
    print(f"Environment shape: {dataset.environment.shape}")
    print(f"Yields shape: {dataset.yields.shape}")
    
    # Save the data to CSV files for later use
    np.savetxt('synthetic_genetics.csv', genetics, delimiter=',')
    np.savetxt('synthetic_environment.csv', environment, delimiter=',')
    np.savetxt('synthetic_yields.csv', yields, delimiter=',')
    
    print("\nFeature descriptions:")
    print("Genetic features (normalized 0-1):")
    print("- Height potential")
    print("- Growth rate")
    print("- Drought resistance")
    print("- Disease resistance")
    print("- Nutrient use efficiency")
    print("\nEnvironmental features (normalized 0-1):")
    print("- Temperature")
    print("- Rainfall")
    print("- Solar radiation")
    print("- Soil quality")
    print("- Relative humidity")
