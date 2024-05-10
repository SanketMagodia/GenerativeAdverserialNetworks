# import pytorch and other libraries
import torch
import hyperParams as params
import dataset
import torchvision.transforms as transforms
import torch.nn as nn
import hyperParams as params
import torch.optim as optim
import torchvision.datasets as datasets
import training
import DCGAN 
import GAN
import WGAN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize((params.imgL, params.imgW)),
    # transforms.CenterCrop((params.imgL, params.imgW)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
])


custom_dataset = dataset.CustomDataset(root_dir='pokemon/pokemon', transform=transform)
custom_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=params.batch_size, shuffle=True, drop_last=True)
#just for WGAN
WGAN.train(custom_loader)
#for normal gan and DCgan, comment the above line and uncomment the below code
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)


# generator = DCGAN.Generator().to(device)
# discriminator = DCGAN.Discriminator().to(device)
# generator.apply(weights_init)
# discriminator.apply(weights_init)

# # Loss and optimizer
# criterion = nn.BCELoss()
# optimizer_g = optim.Adam(generator.parameters(), lr= params.learning_rate, betas = (0.5, 0.999))
# optimizer_d = optim.Adam(discriminator.parameters(), lr= params.learning_rate, betas = (0.5, 0.999))

# training.train(generator, discriminator, criterion, optimizer_d, optimizer_g, custom_loader)


