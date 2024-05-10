import torch
import hyperParams as params
from torchvision.utils import save_image
import torchvision.utils
import os
from torch.utils.tensorboard import SummaryWriter

def train(generator, discriminator, criterion, optimizer_d, optimizer_g, custom_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fake_writer = SummaryWriter(f"logs/fake")
    real_writer = SummaryWriter(f"logs/real")

    fixed_noise = torch.randn(params.batch_size, params.latent_size, 1, 1).to(device)
    step = 0
    for epoch in range(params.num_epochs):
        for i, (images) in enumerate(custom_loader):
            # Move images to device
            images = images.reshape(params.batch_size,3, 64,64).to(device)

            # Create labels for real and fake images
            real_labels = torch.ones(params.batch_size, 1).to(device)
            fake_labels = torch.zeros(params.batch_size, 1).to(device)

            # Train discriminator
            outputs = discriminator(images)
            d_loss_real = criterion(outputs.reshape(-1,1), real_labels)
            real_score = outputs

            # Fake images
            z = torch.randn(params.batch_size, params.latent_size, 1, 1).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs.reshape(-1,1), fake_labels)
            fake_score = outputs

            # Backprop and optimize discriminator
            d_loss = d_loss_real + d_loss_fake
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train generator
            z = torch.randn(params.batch_size, params.latent_size, 1, 1).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs.reshape(-1,1), real_labels)

            # Backprop and optimize generator
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            fake_writer.add_scalar('Discriminator Loss', d_loss.item(), epoch * len(custom_loader) + i)
            fake_writer.add_scalar('Generator Loss', g_loss.item(), epoch * len(custom_loader) + i)
            
        print(f"Epoch [{epoch+1}/{params.num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")
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
            
            
        if epoch % 50 ==0:
            model_path = "models/generator_model_"+str(g_loss.item())+".pth"
            torch.save(generator.state_dict(), model_path)
            fake_images = fake_images.reshape(fake_images.size(0), 3, 64, 64)
            save_image(fake_images, f'generations/fake_images-{epoch+1}.png')
    
