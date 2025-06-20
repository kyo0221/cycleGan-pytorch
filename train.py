import os
import sys
import yaml
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.networks import define_G, define_D, get_scheduler
from util.image_pool import ImagePool
from util.visualizer import SimpleVisualizer
from util.loss import GANLoss

def load_config(config_path="config/train.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_dataset(image_dir_A, image_dir_B):
    dataset_root = Path("datasets/custom")
    shutil.rmtree(dataset_root, ignore_errors=True)

    # Create class folder structure for ImageFolder
    (dataset_root / "trainA" / "images").mkdir(parents=True)
    (dataset_root / "trainB" / "images").mkdir(parents=True)

    for src, dst in [(image_dir_A, dataset_root / "trainA" / "images"),
                     (image_dir_B, dataset_root / "trainB" / "images")]:
        for f in Path(src).glob("*.png"):
            shutil.copy(f, dst / f.name)

    return dataset_root

def create_dataloader(path, image_size, batch_size=1):
    # image_size: int or [width, height]
    if isinstance(image_size, int):
        resize_transform = transforms.Resize(image_size)
        crop_transform = transforms.CenterCrop(image_size)
    elif isinstance(image_size, list) and len(image_size) == 2:
        resize_transform = transforms.Resize((image_size[1], image_size[0]))  # (height, width)
        crop_transform = None
    else:
        raise ValueError("image_size must be int or [width, height]")

    transform_list = [resize_transform]
    if crop_transform:
        transform_list.append(crop_transform)

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ]

    transform = transforms.Compose(transform_list)
    
    # ImageFolder expects class folders, which should already be created by prepare_dataset
    dataset = ImageFolder(path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main(image_dir_A, image_dir_B):
    config = load_config()
    epochs = config.get("epochs", 100)
    image_size = config.get("image_size", 256)
    batch_size = config.get("batch_size", 1)
    model_name = config.get("name", "cyclegan_custom")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = prepare_dataset(image_dir_A, image_dir_B)

    dataloader_A = create_dataloader(dataset_root / "trainA", image_size, batch_size)
    dataloader_B = create_dataloader(dataset_root / "trainB", image_size, batch_size)
    dataloader = zip(dataloader_A, dataloader_B)

    netG_A = define_G(3, 3, 64, netG='resnet_9blocks', device=device)
    netG_B = define_G(3, 3, 64, netG='resnet_9blocks', device=device)
    netD_A = define_D(3, 64, device=device)
    netD_B = define_D(3, 64, device=device)

    criterionGAN = GANLoss().to(device)
    criterionCycle = torch.nn.L1Loss()
    criterionIdt = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(
        list(netG_A.parameters()) + list(netG_B.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        list(netD_A.parameters()) + list(netD_B.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    lr_scheduler_G = get_scheduler(optimizer_G, n_epochs=epochs, n_epochs_decay=epochs)
    lr_scheduler_D = get_scheduler(optimizer_D, n_epochs=epochs, n_epochs_decay=epochs)

    fake_A_pool = ImagePool(50)
    fake_B_pool = ImagePool(50)
    
    visualizer = SimpleVisualizer(f"logs/{model_name}")

    for epoch in range(epochs):
        for i, ((real_A, _), (real_B, _)) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            ##### GENERATOR #####
            optimizer_G.zero_grad()
            fake_B = netG_A(real_A)
            rec_A = netG_B(fake_B)
            fake_A = netG_B(real_B)
            rec_B = netG_A(fake_A)

            idt_A = netG_A(real_B)
            idt_B = netG_B(real_A)

            loss_idt_A = criterionIdt(idt_A, real_B) * 5.0
            loss_idt_B = criterionIdt(idt_B, real_A) * 5.0

            loss_G_A = criterionGAN(netD_A(fake_B), True)
            loss_G_B = criterionGAN(netD_B(fake_A), True)

            loss_cycle_A = criterionCycle(rec_A, real_A) * 10.0
            loss_cycle_B = criterionCycle(rec_B, real_B) * 10.0

            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            loss_G.backward()
            optimizer_G.step()

            ##### DISCRIMINATOR #####
            optimizer_D.zero_grad()

            fake_B_ = fake_B_pool.query(fake_B)
            fake_A_ = fake_A_pool.query(fake_A)

            loss_D_A = (criterionGAN(netD_A(real_B), True) +
                        criterionGAN(netD_A(fake_B_.detach()), False)) * 0.5
            loss_D_B = (criterionGAN(netD_B(real_A), True) +
                        criterionGAN(netD_B(fake_A_.detach()), False)) * 0.5
            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            optimizer_D.step()

            if i % 10 == 0:  # Log every 10 steps
                losses = {
                    'G_loss': loss_G.item(),
                    'D_loss': loss_D.item(),
                    'G_A': loss_G_A.item(),
                    'G_B': loss_G_B.item(),
                    'Cycle_A': loss_cycle_A.item(),
                    'Cycle_B': loss_cycle_B.item(),
                    'Idt_A': loss_idt_A.item(),
                    'Idt_B': loss_idt_B.item(),
                    'D_A': loss_D_A.item(),
                    'D_B': loss_D_B.item()
                }
                visualizer.print_current_losses(epoch, i, losses)

        lr_scheduler_G.step()
        lr_scheduler_D.step()

    # Save final models only after all epochs are completed
    save_dir = Path(f"checkpoints/{model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(netG_A.state_dict(), save_dir / "latest_net_G_A.pth")
    torch.save(netG_B.state_dict(), save_dir / "latest_net_G_B.pth")
    torch.save(netD_A.state_dict(), save_dir / "latest_net_D_A.pth")
    torch.save(netD_B.state_dict(), save_dir / "latest_net_D_B.pth")
    print(f"✅ Training completed. Final models saved to {save_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使い方: python3 train.py <image_dir_A> <image_dir_B>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
