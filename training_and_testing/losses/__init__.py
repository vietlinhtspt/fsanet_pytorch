import torch
import torch.nn as nn

class Criterion(nn.Module):
    def __init__(self, loss_type='RANK', **kwargs):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'MSE':
            self.criterion = nn.MSELoss(**kwargs)
        elif loss_type == 'L1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'SMOOTHL1':
            self.criterion = nn.SmoothL1Loss()
        # elif loss_type == 'WING':
        #     self.criterion = Wingloss()
        elif loss_type == 'RANK':
            # self.criterion = Wingloss()
            self.criterion = nn.MSELoss(**kwargs)
            self.rank_criterion = nn.MarginRankingLoss(margin=0., **kwargs)
            # self.rank_criterion = RankingL2loss(margin=0., **kwargs)
        else:
            raise NotImplementedError

        self.class_criterion = nn.CrossEntropyLoss()

    def forward(self, preds, labels):
        if self.loss_type == 'RANK':
            
            class_loss = 0
            rank_loss = self.rank_criterion(torch.abs(preds[0][:,0]),torch.abs(preds[1][:,0]),labels[-1][:,0])
            rank_loss += self.rank_criterion(torch.abs(preds[0][:,1]),torch.abs(preds[1][:,1]),labels[-1][:,1])
            rank_loss += self.rank_criterion(torch.abs(preds[0][:,2]),torch.abs(preds[1][:,2]),labels[-1][:,2])
            
            angles_loss = self.criterion(preds[0],labels[0]) + self.criterion(preds[1],labels[1])
            
            return angles_loss + 1 * rank_loss + 0.1 * class_loss