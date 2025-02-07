from setuptools import setup, find_packages

setup(
    name="neuromct",
    version="0.0.2",
    author='Arsenii Gavrikov, Andrea Serafini',
    author_email="arsenii.gavrikov@pd.infn.it, andrea.serafini@pd.infn.it",
    packages=find_packages(),
    package_dir={"neuromct": "neuromct"},
    include_package_data=True,
    package_data={
        "neuromct": [
            "configs/*",
        ]
    },  # directory which contains your data
    install_requires=[
        "matplotlib==3.7.2",
        "numpy==1.26.3",
        "optuna==3.5.0",
        "pandas==2.1.1",
        "scikit-learn==1.2.2",
        "scipy==1.11.4",
        "seaborn==0.13.2",
        "setuptools==68.2.2",
        "torch==2.4.0",
        "tqdm==4.65.0",
        "entmax @ git+https://github.com/deep-spin/entmax.git@master#egg=entmax",
    ],
)
