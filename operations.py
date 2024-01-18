import numpy as np
import pictures
import torch
import torch.nn as nn

OPS = {
    'base_layer': lambda C_in, C_out, kernel_size: Baselayer(C_in, C_out, kernel_size, 1, 0),
    'Fully_connected': lambda C_in, C_out: FC(C_in, C_out)
}


class Baselayer(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(Baselayer, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            # nn.BatchNorm2d(C_out, affine=True),
            nn.LeakyReLU(0.33),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FC(nn.Module):
    def __init__(self, C_in, C_out):
        super(FC, self).__init__()
        self.op = torch.nn.Sequential(
            nn.Linear(C_in, C_out),
            nn.LeakyReLU(0.33),
        )

    def forward(self, x):
        return self.op(x)


def Gaus(matrixs, rows, std):
    matrix = matrixs.clone()
    batch_size, _, height, width = matrix.size()
    mean = 1
    for i in range(batch_size):
        matrix[i, 0, rows, :] += torch.randn((len(rows), width)) * std + mean
    return matrix


def Smooth(matrixs, row_set, rows, prop):
    matrix = matrixs.clone()
    max_len = max(len(sublist) for sublist in row_set)
    row_set = [sublist + [sublist[-1]] * (max_len - len(sublist)) for sublist in row_set]

    indices = torch.LongTensor(row_set)
    batch_size, _, height, width = matrix.size()
    for i in range(batch_size):
        selected_rows = matrix[i, 0, indices, :]
        averages = torch.mean(selected_rows, dim=1)
        scaled_averages = averages * prop
        matrix[i, 0, rows, :] += matrix[i, 0, rows, :] * (1 - prop) + scaled_averages
    return matrix


def Trim(matrixs, rows, prop):
    a = 1 - prop
    matrix = matrixs.clone()
    for i in range(matrix.shape[0]):
        for row in rows:
            smaller_indices = matrix[i, 0, row, :] < -a
            larger_indices = matrix[i, 0, row, :] > a
            matrix[i, 0, row, :][smaller_indices] = -a
            matrix[i, 0, row, :][larger_indices] = a

    return matrix


def Zero(matrixs, rows, prop):
    matrix = matrixs.clone()
    num_elements = int(matrix.size(2) * prop)
    for i in range(matrix.shape[0]):
        for row in rows:
            indices = torch.randperm(matrix.size(2))[:num_elements]
            matrix[i, 0, row, indices] = 0

    return matrix


def Amp(matrixs, rows, prop):
    matrix = matrixs.clone()
    for i in range(matrix.shape[0]):
        matrix[i, 0, rows, :] += matrix[i, 0, rows, :] * (1 + prop)
    return matrix


def Red(matrixs, rows, prop):
    matrix = matrixs.clone()
    for i in range(matrix.shape[0]):
        matrix[i, 0, rows, :] += matrix[i, 0, rows, :] * (1 - prop)
    return matrix
