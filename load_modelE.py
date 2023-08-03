
from utils.load_model import load
from torchvision.utils import save_image

#path_to_model_folder = './trained_models/mnist/'  # 原来的版本
path_to_model_folder = './trained_models/mnist/'

model = load(path_to_model_folder)

# Print the latent distribution info
print(model.latent_spec)
#print("ZHUYONGGUI")

# Print model architecture
print(model)

from viz.visualize import Visualizer as Viz

# Create a Visualizer for the model
viz = Viz(model)
viz.save_images = False  # Return tensors instead of saving images

#%matplotlib inline
import matplotlib.pyplot as plt

samples = viz.samples()
plt.imshow(samples.numpy()[0, :, :], cmap='gray')
#plt.title("Zhuyonggui_1")
plt.show()


traversals = viz.all_latent_traversals(size=10)
plt.imshow(traversals.numpy()[0, :, :], cmap='gray')  # 原来的代码
plt.title("Laplacian-JointVAE")
plt.axis('off')
plt.show()
save_image(traversals, './Laplace.png')

# Traverse 3rd continuous latent dimension across columns and first 
# discrete latent dimension across rows
traversals = viz.latent_traversal_grid(cont_idx=2, cont_axis=1, disc_idx=0, disc_axis=0, size=(10, 10))
#traversals = viz.latent_traversal_grid(cont_idx=4, cont_axis=1, disc_idx=0, disc_axis=0, size=(10, 10))
plt.imshow(traversals.numpy()[0, :, :], cmap='gray')
#plt.title("Zhuyonggui_3")
plt.show()

from viz.visualize import reorder_img

ordering = [9, 3, 0, 5, 7, 6, 4, 8, 1, 2]  # The 9th dimension corresponds to 0, the 3rd to 1 etc...
traversals = reorder_img(traversals, ordering, by_row=True)
plt.imshow(traversals.numpy()[0, :, :], cmap='gray')
#plt.title("Zhuyonggui_4")
plt.show()

#### 从这里从（8，10）变成（12，10）

traversal = viz.latent_traversal_line(cont_idx=6, size=12)
plt.imshow(traversal.numpy()[0, :, :], cmap='gray')
#plt.title("Zhuyonggui_5")
plt.show()

from utils.dataloaders import get_mnist_dataloaders

# Get MNIST test data
_, dataloader = get_mnist_dataloaders(batch_size=32)

# Extract a batch of data
for batch, labels in dataloader:
    break
    
recon = viz.reconstructions(batch, size=(8, 8))

plt.imshow(recon.numpy()[0, :, :], cmap='gray')
#plt.title("Zhuyonggui_6")
plt.show()

from torch.autograd import Variable
encodings = model.encode(Variable(batch))

# Continuous encodings for the first 5 examples
encodings['cont'][0][:5]


