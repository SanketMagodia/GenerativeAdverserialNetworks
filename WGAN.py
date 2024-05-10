import torch
import torch.nn as nn
import torch.optim as optim
import hyperParams as params
from torchvision.utils import save_image
import torchvision
from torch.utils.tensorboard import SummaryWriter
from WGAN_UTILS import gradient_penalty


def train( custom_loader): 
    def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.conv_transpose1 = nn.ConvTranspose2d(params.latent_size, 64*16, kernel_size=4, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(64*16)
            self.relu1 = nn.ReLU()
            self.conv_transpose2 = nn.ConvTranspose2d(64*16, 64*8, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64*8)
            self.relu2 = nn.ReLU()
            self.conv_transpose3 = nn.ConvTranspose2d(64*8, 64*4, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(64*4)
            self.relu3 = nn.ReLU()
            self.conv_transpose4 = nn.ConvTranspose2d(64*4, 64*2, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn4 = nn.BatchNorm2d(64*2)
            self.relu4 = nn.ReLU()
            self.conv_transpose5 = nn.ConvTranspose2d(64*2, 3, kernel_size=4, stride=2, padding=1, bias=False)
            self.tanh = nn.Tanh()

        def forward(self, input):
            x = self.relu1(self.bn1(self.conv_transpose1(input)))
            x = self.relu2(self.bn2(self.conv_transpose2(x)))
            x = self.relu3(self.bn3(self.conv_transpose3(x)))
            x = self.relu4(self.bn4(self.conv_transpose4(x)))
            x = self.tanh(self.conv_transpose5(x))
            return x
    
#--------------------------------------------------------------------------------------------------
    
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
            self.conv2 = nn.Conv2d(64, 64 * 2, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn2 = nn.InstanceNorm2d(64 * 2, affine = True)
            self.conv3 = nn.Conv2d(64 * 2, 64 * 4, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn3 = nn.InstanceNorm2d(64 * 4, affine = True)
            self.conv4 = nn.Conv2d(64 * 4, 64 * 8, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn4 = nn.InstanceNorm2d(64 * 8, affine = True)
            self.conv5 = nn.Conv2d(64 * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
            self.sigmoid = nn.Sigmoid()
            self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, input):
            x = self.leaky_relu(self.conv1(input))
            x = self.leaky_relu(self.bn2(self.conv2(x)))
            x = self.leaky_relu(self.bn3(self.conv3(x)))
            x = self.leaky_relu(self.bn4(self.conv4(x)))
            x = self.conv5(x)
            return x
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    initialize_weights(generator)
    initialize_weights(discriminator)
    lr = 1e-4  #5e-5
    criticIter = 5
    clip = 0.01
    LAMBDA_GP = 10
    # optG = optim.RMSprop(generator.parameters(), lr=lr)
    # optC = optim.RMSprop(discriminator.parameters(), lr=lr)
    optG = optim.Adam(generator.parameters(), lr=lr, betas = (0.0, 0.9))
    optC = optim.Adam(discriminator.parameters(), lr=lr, betas = (0.0, 0.9))
    fake_writer = SummaryWriter(f"logs/fake")
    real_writer = SummaryWriter(f"logs/real")

    fixed_noise = torch.randn(params.batch_size, params.latent_size, 1, 1).to(device)
    step = 0
    for epoch in range(params.num_epochs):
        for i, (images) in enumerate(custom_loader):
            for _ in range(criticIter):
                images = images.reshape(params.batch_size,3, 64,64).to(device)
                outputsReal = discriminator(images).reshape(-1)
                
                z = torch.randn(params.batch_size, params.latent_size, 1, 1).to(device)
                fake_images = generator(z)
                outputsFake = discriminator(fake_images.detach()).reshape(-1)
                gp = gradient_penalty(discriminator, images, fake_images, device=device)
                loss_discriminator = (-(torch.mean(outputsReal) - torch.mean(outputsFake)) + LAMBDA_GP*gp)
                discriminator.zero_grad()
                loss_discriminator.backward(retain_graph = True)
                optC.step()

                # for p in discriminator.parameters():
                #      p.data.clamp_(-clip, clip)

            output = discriminator(fake_images).reshape(-1)
            loss_generator =  -torch.mean(output)
            generator.zero_grad()
            loss_generator.backward()
            optG.step()
            # Train generator

            fake_writer.add_scalar('Discriminator Loss', loss_discriminator, epoch * len(custom_loader) + i)
            fake_writer.add_scalar('Generator Loss', loss_generator, epoch * len(custom_loader) + i)
            
        print(f"Epoch [{epoch+1}/{params.num_epochs}], d_loss: {loss_discriminator:.4f}, g_loss: {loss_generator:.4f}")
        # Save generated images
        if (epoch+1) == 1:
            images = images.reshape(images.size(0), 3, 64, 64)
            save_image(images, 'generations/real_images.png')
            
        with torch.no_grad():
                Bfake = generator(fixed_noise)
                Breal = images.reshape(images.size(0), 3, 64, 64)
                img_grid_fake = torchvision.utils.make_grid(Bfake[:6], normalize=True)
                img_grid_real = torchvision.utils.make_grid(Breal[:6], normalize=True)

                fake_writer.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                real_writer.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step+=1
            
            
        if epoch % 100 ==0:
            model_path = "models/generator_model_"+str(epoch+1)+".pth"
            torch.save(generator.state_dict(), model_path)
            fake_images = fake_images.reshape(fake_images.size(0), 3, 64, 64)
            save_image(fake_images, f'generations/fake_images-{epoch+1}.png')

    