import random
import math
import operations as op
import torch
import torchvision.transforms as transforms
from PIL import Image


def floor_min_one(num):
    result = max(1, math.floor(num))
    return result


def generate_element(set_size):
    return [
        random.randint(1, 5),
        random.randint(1, set_size),
        round(random.uniform(0.0, 1.0), 2),
        round(random.uniform(0.0, 1.0), 2)
    ]


def Individual_generation(set_size):
    result_list = []
    for _ in range(4):
        result_list.extend(generate_element(set_size))
    return result_list


def select_items_and_flatten(list_fenqu, size, number_to_select):
    select_set = random.sample(list_fenqu, size)
    row = [
        item for sublist in select_set
        for item in random.sample(sublist, number_to_select)
    ]
    return row, select_set


def augment(var_ind, list_fenqu, matrix):
    list1_row, set1 = select_items_and_flatten(list_fenqu, var_ind[1], floor_min_one(var_ind[2]))
    list2_row, set2 = select_items_and_flatten(list_fenqu, var_ind[5], floor_min_one(var_ind[6]))
    list3_row, set3 = select_items_and_flatten(list_fenqu, var_ind[9], floor_min_one(var_ind[10]))
    list4_row, set4 = select_items_and_flatten(list_fenqu, var_ind[13], floor_min_one(var_ind[14]))

    if var_ind[0] == 0:
        matrix1 = op.Gaus(matrix, list1_row, var_ind[3])
    if var_ind[0] == 1:
        matrix1 = op.Smooth(matrix, set1, list1_row, var_ind[3])
    if var_ind[0] == 2:
        matrix1 = op.Trim(matrix, list1_row, var_ind[3])
    if var_ind[0] == 3:
        matrix1 = op.Zero(matrix, list1_row, var_ind[3])
    if var_ind[0] == 4:
        matrix1 = op.Amp(matrix, list1_row, var_ind[3])
    if var_ind[0] == 5:
        matrix1 = op.Red(matrix, list1_row, var_ind[3])

    if var_ind[4] == 0:
        matrix2 = op.Gaus(matrix, list2_row, var_ind[7])
    if var_ind[4] == 1:
        matrix2 = op.Smooth(matrix, set2, list2_row, var_ind[7])
    if var_ind[4] == 2:
        matrix2 = op.Trim(matrix, list2_row, var_ind[7])
    if var_ind[4] == 3:
        matrix2 = op.Zero(matrix, list2_row, var_ind[7])
    if var_ind[4] == 4:
        matrix2 = op.Amp(matrix, list2_row, var_ind[7])
    if var_ind[4] == 5:
        matrix2 = op.Red(matrix, list2_row, var_ind[7])

    if var_ind[8] == 0:
        matrix3 = op.Gaus(matrix1, list3_row, var_ind[11])
        matrix4 = op.Gaus(matrix2, list3_row, var_ind[11])
    if var_ind[8] == 1:
        matrix3 = op.Smooth(matrix1, set3, list3_row, var_ind[11])
        matrix4 = op.Smooth(matrix2, set3, list3_row, var_ind[11])
    if var_ind[8] == 2:
        matrix3 = op.Trim(matrix1, list3_row, var_ind[11])
        matrix4 = op.Trim(matrix2, list3_row, var_ind[11])
    if var_ind[8] == 3:
        matrix3 = op.Zero(matrix1, list3_row, var_ind[11])
        matrix4 = op.Zero(matrix2, list3_row, var_ind[11])
    if var_ind[8] == 4:
        matrix3 = op.Amp(matrix1, list3_row, var_ind[11])
        matrix4 = op.Amp(matrix2, list3_row, var_ind[11])
    if var_ind[8] == 5:
        matrix3 = op.Red(matrix1, list3_row, var_ind[11])
        matrix4 = op.Red(matrix2, list3_row, var_ind[11])

    if var_ind[12] == 0:
        matrix5 = op.Gaus(matrix1, list4_row, var_ind[15])
        matrix6 = op.Gaus(matrix2, list4_row, var_ind[15])
    if var_ind[12] == 1:
        matrix5 = op.Smooth(matrix1, set4, list4_row, var_ind[15])
        matrix6 = op.Smooth(matrix2, set4, list4_row, var_ind[15])
    if var_ind[12] == 2:
        matrix5 = op.Trim(matrix1, list4_row, var_ind[15])
        matrix6 = op.Trim(matrix2, list4_row, var_ind[15])
    if var_ind[12] == 3:
        matrix5 = op.Zero(matrix1, list4_row, var_ind[15])
        matrix6 = op.Zero(matrix2, list4_row, var_ind[15])
    if var_ind[12] == 4:
        matrix5 = op.Amp(matrix1, list4_row, var_ind[15])
        matrix6 = op.Amp(matrix2, list4_row, var_ind[15])
    if var_ind[12] == 5:
        matrix5 = op.Red(matrix1, list4_row, var_ind[15])
        matrix6 = op.Red(matrix2, list4_row, var_ind[15])

    return torch.cat([matrix, matrix3, matrix4, matrix5, matrix6], dim=0)


def augment_general(tensor):
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

