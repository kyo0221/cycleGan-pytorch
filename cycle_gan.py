import os
import sys
import yaml
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from models.networks import ResnetGenerator

def load_config(config_path="config/params.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_generator(config, device):
    netG = ResnetGenerator(
        input_nc=config.get("input_nc", 3),
        output_nc=config.get("output_nc", 3),
        ngf=64,
        norm_type='instance',
        use_dropout=config.get("use_dropout", False),
        n_blocks=config.get("n_blocks", 9),
    )
    netG.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
    netG.eval()
    netG.to(device)
    return netG

def transform_image(img, config):
    image_size = config.get("image_size", [256, 256])
    if isinstance(image_size, int):
        resize = transforms.Resize(image_size)
    else:
        resize = transforms.Resize((image_size[1], image_size[0]))  # (height, width)

    transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(img)

def deprocess(tensor):
    tensor = tensor.cpu().detach()
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor.squeeze(0))

def main(image_dir, config_path="config/params.yaml"):
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    netG = load_generator(config, device)

    image_dir = Path(image_dir)
    suffix = config.get("output_suffix", "_translated")
    output_dir = image_dir.parent / (image_dir.name + suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("*.png"))

    for image_path in tqdm(image_files):
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform_image(img, config).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = netG(input_tensor)

        result_img = deprocess(output_tensor)
        result_img.save(output_dir / image_path.name)

    print(f"\n✅ 変換完了: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使い方: python3 cycle_gan.py <画像ディレクトリ>")
        sys.exit(1)

    image_dir = sys.argv[1]
    main(image_dir)
