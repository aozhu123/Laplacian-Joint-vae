# Laplacian-Joint-vae
# Learning Joint-VAE with Continuous Laplacian Latent Variables and Discrete Latent Variables


### Requirements
- Python >= 3.5
- PyTorch >= 1.3
- Pytorch Lightning >= 0.6.0 ([GitHub Repo](https://github.com/PyTorchLightning/pytorch-lightning/tree/deb1581e26b7547baf876b7a94361e60bb200d32))
- CUDA enabled computing device

Pytorch implementation of [Lapalacian-Joint-VAE].

This repo contains an implementation of JointVAE, a framework for jointly disentangling continuous Laplacian and discrete factors of variation in data in an unsupervised manner.


#### Example usage
```python
from jointvae.models import VAE
from jointvae.training import Trainer
from torch.optim import Adam
from viz.visualize import Visualizer as Viz
