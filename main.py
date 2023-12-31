import torch
from jointvae.models import VAE
from jointvae.training import Trainer
from utils.dataloaders import get_mnist_dataloaders
from torch import optim
from viz.visualize import Visualizer as Viz  # 新增加的代码
import numpy as np
import matplotlib.pyplot as plt

#dataset_path = r"./celeba_64"


batch_size = 64 # 原来的批处理大小
lr = 5e-4
epochs = 1#100

# Check for cuda
use_cuda = torch.cuda.is_available()

# Load data
data_loader, _ = get_mnist_dataloaders(batch_size=batch_size)
img_size = (1, 32, 32)


# Define latent spec and model
#latent_spec = {'cont': 10, 'disc': [10]}  # 原来的代码
latent_spec = {'cont': 10, 'disc': [10]}   # 修改的代码
model = VAE(img_size=img_size, latent_spec=latent_spec,
            use_cuda=use_cuda)
if use_cuda:
    model.cuda()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define trainer
trainer = Trainer(model, optimizer,
                  cont_capacity=[0.0, 5.0, 25000, 30],
                  disc_capacity=[0.0, 2.5, 12500, 15],
                  use_cuda=use_cuda)

# Train model for 100 epochs
trainer.train(data_loader, epochs)

# Save trained model
torch.save(trainer.model.state_dict(), 'example-model.pt')



