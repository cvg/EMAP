import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):

    def __init__(self, loss_type="mse"):
        super(EdgeLoss, self).__init__()
        if loss_type == "mse":
            self.loss_func = F.mse_loss
        elif loss_type == "l1":
            self.loss_func = F.l1_loss

    def forward(self, pred_edge, gt_edge):

        edge_loss = self.loss_func(pred_edge, gt_edge, reduction="mean")
        return edge_loss
