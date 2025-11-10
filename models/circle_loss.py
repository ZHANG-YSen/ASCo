# -*- coding:utf-8 -*-
"""
作者：YSen
日期：2023年02月28日
功能：
"""

from typing import Tuple

import torch
from torch import nn, Tensor


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        """
        sp.shape=torch.Size([256, 1])
        sn.shape=torch.Size([256, 256])
        """
        sp1 = sp.squeeze(1)  # [256, 1]==>[256]
        # print("sp1.shape={}".format(sp1.shape))

        ap = torch.clamp_min(- sp1.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        # print("ap.shape={}\nan.shape={}".format(ap.shape, an.shape))  # an.shape=[256, 256], ap.shape=[256]

        delta_p = 1 - self.m
        delta_n = self.m

        # print("delta_p={}\ndelta_n={}".format(delta_p, delta_n))

        logit_p = - ap * (sp1 - delta_p) * self.gamma
        # print("logit_p.shape={}".format(logit_p.shape))  # logit_p.shape=torch.Size([256])

        sn1 = sn - delta_n
        logit_n1 = an * sn1  # 两矩阵对应元素相乘
        logit_n = logit_n1 * self.gamma

        logit_p1 = logit_p.view(logit_p.shape[0], 1)  # 为了拼接：[256]==>[256, 1]
        # print("logit_p1={}".format(logit_p1))

        positive = torch.logsumexp(logit_p1, 1)  # [256, 1]==>positive=[256]
        negative = torch.logsumexp(logit_n, 1)  # [256, 256]==>negative=[256] 注意计算方式
        # print("positive.shape={}, negative.shape={}".format(positive.shape, negative.shape))

        sum_pos_neg = positive + negative  # sum_pos_neg.shape=torch.Size([256])
        # print("sum_pos_neg.shape={}".format(sum_pos_neg.shape))

        # print("logit_p1.shape={}\nlogit_n.shape={}".format(logit_p1.shape, logit_n.shape))  # logit_p1.shape=torch.Size([256, 1]), logit_n.shape=torch.Size([256, 256])

        # out = torch.cat((logit_p1, logit_n), dim=1)
        # print("out.shape={}".format(out.shape))  # out.shape=torch.Size([256, 257])
        loss = self.soft_plus(sum_pos_neg)  # loss.shape=torch.Size([256])
        # print("loss.shape={}".format(loss.shape))


        # loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=sp.device))

        return loss


if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
    lbl = torch.randint(high=10, size=(256,))

    inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)

    criterion = CircleLoss(m=0.25, gamma=256)
    circle_loss = criterion(inp_sp, inp_sn)

    print(circle_loss)