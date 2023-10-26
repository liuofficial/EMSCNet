import torch
from torch import nn

def logsum(args):
    return multilabel_categorical_crossentropy(args)

class multilabel_categorical_crossentropy(nn.Module):
    def __init__(self,args):
        super(multilabel_categorical_crossentropy, self).__init__()
        self.soft_plus = nn.Softplus()
        self.num_positive=args.num_positive
        self.num_negative=args.num_negative
        self.m=0
        self.y=10
    def forward(self,y_pred):
        x_positive=y_pred[:,:self.num_positive]
        x_negetive=y_pred[:,self.num_positive:self.num_positive+self.num_negative]

        x_negetive=self.y*(x_negetive+self.m)
        x_positive=-self.y*(x_positive)
        x_negetive=torch.logsumexp(x_negetive,dim=1)
        x_positive=torch.logsumexp(x_positive,dim=1)
        loss=self.soft_plus(x_negetive+x_positive).sum()
        return loss/y_pred.shape[0]