from setuptools import setup, find_packages

setup(
    name="neuromct",
    version="0.0.1",
    author='Arsenii Gavrikov, Andrea Serafini',
    author_email="arsenii.gavrikov@pd.infn.it, andrea.serafini@pd.infn.it",
    packages=find_packages(),
    package_dir={"neuromct": "neuromct"},
)
