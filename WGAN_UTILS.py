import torch 
import torch.nn as nn


def gradient_penalty(critic , real_data, fake_data, device = "gpu"):
    BATCH_SIZE, C, H, W = real_data.shape ### Batch size is the number of images in a batch. It's usually specified when creating your dataloader.
    epsilon = torch.rand((BATCH_SIZE , 1,1,1)).repeat(1,C,H,W).to(device) # epsilon is a random tensor with shape (batch size, 1, height of image, width of image)
    interpolated_images = real_data * epsilon + fake_data*(1-epsilon)
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True,
        )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim= 1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty