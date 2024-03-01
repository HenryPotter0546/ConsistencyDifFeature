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
train.yaml for train_aggregation_network.py
####
test.yaml for test_aggregation_network.py
####
pca.yaml for pca_visualization.py


### train_aggregation_network.py
set config path as train.yaml, then run train_aggregation_network.py

### try the correspondence task in any pair of image
using notebook run demo.ipynb

### get the pca outcome
run pca_visualization.py in utils


