import torch
import torchvision.transforms as transforms
from PIL import Image


def augment_tensor(tensor):
    T = tensor.clone()
    pil_images = [Image.fromarray(tensor[i, 0].numpy(), mode='L') for i in range(tensor.size(0))]
    transform1 = transforms.RandomHorizontalFlip(p=1.0)
    transform2 = transforms.RandomVerticalFlip(p=1.0)
    transform3 = transforms.RandomRotation(degrees=(0, 180))

    T1 = torch.stack([transforms.ToTensor()(transform1(img)) for img in pil_images])

    T2 = torch.stack([transforms.ToTensor()(transform2(img)) for img in pil_images])

    T3 = torch.stack([transforms.ToTensor()(transform3(img)) for img in pil_images])


    T4 = tensor.clone()
    c_x, c_y = T4.shape[2] // 2, T4.shape[3] // 2
    block_size = 20

    T4[:, :, c_x - block_size:c_x + block_size, c_y - block_size:c_y + block_size] = 0

    return torch.cat([T, T1, T2, T3, T4], dim=0)