import torch
import torchvision
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image

transform = transforms.Compose([ # reinitialize dataset object
    transforms.ToTensor()
])

mnist = torchvision.datasets.MNIST('', download = True, transform = transform)
data_loader = torch.utils.data.DataLoader(mnist,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=0)
# testing data
test_data = torchvision.datasets.MNIST('',
                           download=True,
                           train = False,
                           transform=transform)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=0)

def patchify(images, n_patches_per_row):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches_per_row ** 2, c * h * w // n_patches_per_row ** 2)
    patch_size = h // n_patches_per_row

    for idx, image in enumerate(images):
        for i in range(n_patches_per_row):
            for j in range(n_patches_per_row):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches_per_row + j] = patch.flatten()

    return patches

class MLP(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.up = nn.Linear(embed_dim, embed_dim*4)
        self.relu = nn.ReLU()
        self.down = nn.Linear(embed_dim*4, embed_dim)

    def forward(self, x):
        x = self.up(x)
        x = self.relu(x)
        x = self.down(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim=16, num_heads=4):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn_output, _ = self.mha(q, k, v)
        x = x + attn_output
        x = self.mlp(x)
        return x
    
class VIT(nn.Module):
    def __init__(self, input_dim, output_dim, n_patches, hidden_d = 32, blocks=3, device="cuda"):
        super().__init__()

        self.chw = input_dim # (C, H, W) 1 x 28 x 28
        self.n_patches = n_patches
        self.hidden_d = hidden_d
        self.blocks = blocks
        self.device = device

        assert self.chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert self.chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (self.chw[1] / n_patches, self.chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(self.chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Adding Classification Token that is learned by our model
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional encoding
        self.pos_emb = nn.Embedding((n_patches**2)+1, self.hidden_d)

        self.blocks = nn.ModuleList([
            Block(self.hidden_d)
            for _ in range(self.blocks)
        ])

        self.classification_head = nn.Linear(self.hidden_d, output_dim)

    def patchify(self, images, n_patches_per_row):
        n, c, h, w = images.shape

        assert h == w, "Patchify method is implemented for square images only"

        patches = torch.zeros(n, n_patches_per_row ** 2, c * h * w // n_patches_per_row ** 2)
        patch_size = h // n_patches_per_row

        for idx, image in enumerate(images):
            for i in range(n_patches_per_row):
                for j in range(n_patches_per_row):
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches_per_row + j] = patch.flatten()

        return patches.to(self.device)

    def tokenize(self, x):
        patches = self.patchify(x, self.n_patches) # 4 x 49 x 16
        N, T, D = patches.shape
        # x = [batch size, height, width]

        tokens = self.linear_mapper(patches) # 4 x 49 x 8

        tokens = torch.cat([self.class_token.repeat(N, 1, 1), tokens], dim=1)
        # appended CLS token


        # expected shape: [4, 50, 16] = [N, T, D]

        pos_embed = self.pos_emb(torch.arange(0,T+1).to(self.device)) # shape: [50, 16]
        pos_embed = pos_embed.unsqueeze(0).repeat(N, 1, 1)

        tokens += pos_embed

        return tokens

    def forward(self, x): # b x c x h x w - batch size x 1 x 28 x 28, turn into output:
        # sequence of embeddings for each batch, b x s x d, s is # patches, d is dimensionality
        # batch size x 49 x dimension of embeddings
        x = self.tokenize(x)
        for block in self.blocks:
            x = block(x)

        cls_tokens = x[:, 0, :] # shape [N, 1, D]
        # cls_tokens = cls_tokens.squeeze(1) # shape [N, D]
        out = self.classification_head(cls_tokens)
        return out
    
    
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    final_model = VIT(
        input_dim=(1, 28, 28), output_dim=10, n_patches=7, hidden_d=16, blocks=2, device=device
    )
    # change this according to where the model weights are saved
    final_model.load_state_dict(torch.load('evenbetterweights.pth', map_location=torch.device('cpu')))
    final_model.eval()
    return final_model

def load_fashion_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    final_model = VIT(
        input_dim=(1, 28, 28), output_dim=10, n_patches=7, hidden_d=16, blocks=2, device=device
    )

    final_model.load_state_dict(torch.load('fashion_weights.pth', map_location=torch.device('cpu')))
    final_model.eval()
    return final_model
       
def get_random_image(dataset, label):
    for image_tensor, target in dataset:
        if target == label:
            # Convert PyTorch tensor to numpy array
            numpy_image = image_tensor.numpy().squeeze()
            # Normalize to [0, 255] and convert to uint8
            numpy_image = (numpy_image * 255).astype(np.uint8)
            return numpy_image
        
def is_canvas_blank(image_data):
    if image_data is None:
        return True
    return np.all(image_data == image_data[0, 0, :])