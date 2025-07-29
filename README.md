[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/releases/3.8.0/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Simulation-based inference for Precision Neutrino Physics through Neural Monte Carlo tuning**

This repository contains the implementation of neural likelihood estimators for Monte Carlo parameter tuning in high-precision neutrino experiments, with a focus on the JUNO (Jiangmen Underground Neutrino Observatory) detector.

## Overview

Precise modeling of detector energy response is crucial for next-generation neutrino experiments. This work develops neural likelihood estimation methods within the simulation-based inference framework to address the computational challenges arising from lack of analytical likelihoods.

### Key features

- **Two complementary neural density estimators:**
  - **TEDE** (Transformer Encoder Density Estimator): Histogram-based binned likelihood analysis
  - **NFDE** (Normalizing Flows Density Estimator): Exact continuous probability density modeling for unbinned analysis

- **Bayesian parameter inference** using nested sampling
- **Energy response parameter tuning** for three correlated parameters: Birks' coefficient (k_B), light yield (Y), and Cherenkov factor (f_C)
- **Comprehensive uncertainty quantification** with additional testing datasets: near-zero systematic biases and uncertainties limited only by the statistics of the calibraion data

<div align="center">
  <img src="docs/tede.pdf" width="600" alt="TEDE Architecture">
  <br><em>Transformer Encoder Density Estimator (TEDE) Architecture</em>
</div>

## Study background

### The JUNO experiment

The Jiangmen Underground Neutrino Observatory (JUNO) is a large-scale liquid scintillator neutrino detector designed to:
- Determine neutrino mass ordering
- Measure oscillation parameters with sub-percent precision
- Study various neutrino phenomena from geoneutrinos to supernovae

### Energy response modeling in large-scale neutrino detectors

The energy response in JUNO depends on three key parameters:

1. **Birks' coefficient (k_B)**: Models non-linear energy quenching at high ionization densities
2. **Light yield (Y)**: Defines scintillation photons emitted per unit energy after quenching
3. **Cherenkov light yield factor (f_C)**: Scales the energy-dependent yield of photons originating from Cherenkov radiation

These parameters exhibit strong correlations and non-linear behavior, making traditional MC tuning approaches computationally prohibitive.

### Neural likelihood estimation approach

Our method uses simulation-based inference to:
1. Train models on simulated calibration events
2. Learn conditional probability densities p(x|φ) where x is the observed energy and φ are the parameters
3. Integrate learned likelihoods with Bayesian nested sampling for parameter inference

<div align="center">
  <img src="docs/nfde_inference_vis.pdf" width="600" alt="NFDE Inference">
  <br><em>Normalizing Flows transforming complex energy distributions to simple Gaussian</em>
</div>

## Models performance

Both models demonstrate excellent performance:

- **Accuracy**: Near-zero systematic bias 
- **Precision**: Statistical uncertainty limited
- **Robustness**: Validated across 1,000 parameter combinations
- **Speed**: apprx. 1000x faster than full MC simulation

<div align="center">
  <img src="docs/model_comparison_spectra.pdf" width="800" alt="Model Comparison">
  <br><em>Comparison of modeled PDFs with true energy spectra for all five calibration sources</em>
</div>

## Repository structure

```
neuromct/
├── neuromct/                   # Main package
│   ├── configs/                 # Configuration files
│   ├── dataset/                 # Data loading and preprocessing
│   ├── models/                  # Models implementations
│   │   └── ml/                   # TEDE and NFDE models
│   ├── fit/                   # Parameter inference methods
│   ├── utils/                 # Utility functions
│   └── plot/                  # Visualization tools
├── scripts/                  # Training and analysis scripts
└── docs/                     # Documentation and figures 
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{neuromct2025,
    title={Simulation-based inference for Precision Neutrino Physics through Neural Monte Carlo tuning},
    author={Gavrikov, Arsenii and Serafini, Andrea and Dolzhikov, Dmitry and others},
    year={2025},
    note={In preparation}
}
```
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We are thankful to the JUNO collaboration for the support and advices provided during the drafting of this manuscript. We are also very grateful to CNAF and JINR cloud services for providing the computing resources necessary for the simulated data production and to CloudVeneto for offering IT support and infrastructure for training the machine learning models used in this study. Arsenii Gavrikov has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101034319 and from the European Union --- NextGenerationEU. Dmitry Dolzhikov and Maxim Gonchar are supported in the framework of the State project ``Science'' by the Ministry of Science and Higher Education of the Russian Federation under the contract 075-15-2024-541.

---

*For detailed methodology and results, see our accompanying paper on simulation-based inference for precision neutrino physics.*
