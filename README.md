# Neural-Barrier-Function
Learn a NN vector barrier function with a convex optimization-based fine-tuning step. 

## Installation

```bash
conda create --name test python==3.7.13
conda activate test

% install pytorch with a suitable version on your device, see (https://pytorch.org/get-started/previous-versions/). The following versions have been tested and recommended for use.

% CPU only
pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu

or 

% CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


% install packages
pip install -r requirements.txt
conda install -c conda-forge cvxpy
```

## Run experiments
In your terminal, run
```
cd examples/double_integrator
python main_iter_train_double_integrator.py --config ab_crown.yaml 
```
