
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import lmdb
import numpy as np

def peakiness_score(inputs, ksize=3, dilation=1, name='con'):

    pad_size = ksize // 2 + (dilation - 1)

    pad_inputs = F.pad(inputs.to(torch.float32), (pad_size, pad_size, pad_size, pad_size), "reflect")

    avg_spatial_inputs = nn.AvgPool2d(kernel_size=(ksize, ksize), stride=1, padding=0)(pad_inputs)

    avg_channel_inputs = torch.mean(inputs, dim=1, keepdim=True)

    alpha = F.softplus(inputs - avg_spatial_inputs)
    beta = F.softplus(inputs - avg_channel_inputs)
    return alpha, beta

def exclude_kpds(score_map, size=240, throld=56):
    mask = score_map > 0

    nms_mask = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)(score_map)
    nms_mask = torch.eq(score_map, nms_mask)
    mask = torch.logical_and(nms_mask, mask)

    eof_size = 5

    eof_mask = torch.ones((1, 1, size - 2 * eof_size, size - 2 * eof_size), dtype=torch.float32)
    eof_mask = F.pad(eof_mask, (eof_size, eof_size, eof_size, eof_size), value=0.0)
    eof_mask = eof_mask.to(torch.bool).to(device)

    mask_b = torch.logical_and(eof_mask, mask)
    score_map_b = score_map

    index = []
    score = []
    for i in range(score_map_b.shape[0]):
        mask = torch.reshape(mask_b[i], (size, size))
        score_map = torch.reshape(score_map_b[i], (size, size))

        indices = mask.nonzero()
        scores = score_map[mask]
        sample = torch.argsort(scores)[0:throld]

        indices_x = torch.gather(input=indices[:, 0], dim=0, index=sample)
        indices_y = torch.gather(input=indices[:, 1], dim=0, index=sample)

        indices = torch.cat((indices_x.view(1, throld, 1), indices_y.view(1, throld, 1)), dim=2)
        scores = torch.gather(input=scores, dim=0, index=sample).unsqueeze(0)
        index.append(indices)
        score.append(scores)

    return torch.cat(index, dim=0).to(torch.float32), torch.cat(score, dim=0).to(torch.float32)
