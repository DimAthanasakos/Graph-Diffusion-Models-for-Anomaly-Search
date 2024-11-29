#!/usr/bin/bash

# Load tf 
module load tensorflow/2.9.0

module list

# Install additional packages
# The modulefiles automatically set the $PYTHONUSERBASE environment variable for you, 
#   so that you will always have your custom packages every time you load that module.
pip install --user \
energyflow==1.3.2 \
h5py==3.7.0 \
horovod==0.24.3 \
jetnet==0.2.3.post3 \
matplotlib==3.6.2 \
mplhep==0.3.26 \
numpy==1.23.5 \
pandas==1.5.2 \
PyYAML==6.0.1 \
scikit_learn==1.2.0 \
scipy==1.11.2 \
tensorflow_addons==0.19.0 \
seaborn==0.11.2 \



# The following packages are already installed by the pytorch module
#matplotlib==3.5.1 \
#networkx==2.7.1 \
#numpy==1.21.2 \
#pandas==1.4.1 \
#pyyaml==6.0 \
#scikit-learn==1.0.2 \
#torch==1.11 \
#torch-geometric==2.0.4 \
#torch-scatter==2.0.9 \
#torch-sparse==0.6.13