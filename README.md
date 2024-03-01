# ConsistencyDifFeature
## Setup
```
conda create -n cdhf python=3.10
conda activate cdhf
pip install -r requirements.txt
```

## Folder Structure

### configs
####
e.g. train.yaml 
#####
parameters:
```
save_timestep: choose which timestep to aggregate
lcm_model_name: choose the diffusion model
extract_mode:"iid_noise_denoise" means add one step noise
```

### train_aggregation_network.py
set config path as train.yaml, then run train_aggregation_network.py

### test_aggregation_network.py
set config path as test.yaml, then run test_aggregation_network.py

### get the pca outcome
set config path as pca.yaml run pca_visualization.py in utils

### try the correspondence task in any pair of image
using notebook run demo.ipynb



