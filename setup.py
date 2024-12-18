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
)
