# Sonification

Sonification project for METAMERSION

# Installation 


1. Install miniconda or anaconda
2. Clone this repo and its submodule
   ```
   git clone --recurse-submodules --remote-submodules https://github.com/hyungju-jeon/sonification
   ```
3. Install Conda environment from `enviroment.yml`
   ```
   conda env create -f environment.yml
   ```
   Also install the dependencies of the pseyepy package for interfacing the USB camera with Python:
   ```
   python utils/pseyepy/setup.py
   ```
5. `pip install` package
   ```
   pip install -e .
   ```
