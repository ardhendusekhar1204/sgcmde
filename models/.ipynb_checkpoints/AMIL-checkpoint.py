import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from .SoftFilter import SoftFilterLayer


class AMIL_layer(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.1, activation=None):
        super(AMIL_layer, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        if activation == 'sigmoid':
            self.attention_c = nn.Sequential(
                nn.Linear(D, 1),
                nn.Sigmoid()
            )
        elif activation == 'tanh':
            self.attention_c = nn.Sequential(
                nn.Linear(D, 1),
                nn.Tanh()
            )
        elif activation is None:
            self.attention_c = nn.Linear(D, 1)
        else:
            raise NotImplementedError

    def forward(self, x):
        if len(x.shape) == 2:
            # print('ashvahvdsvdbvhdsbhkhknsvdnksvbnkdvnkdjlvewhkvbjvbndsndsnkdsvjlvljvljdvljim')
            x = x.unsqueeze(0)
        a = self.attention_a(x)
        # print('a ka shape', a.shape)
        b = self.attention_b(x)
        # print('b ka shape', b.shape)
        A_without_softmax = a.mul(b)
        # print('A_without_softmax ka shape', A_without_softmax.shape)
        A_without_softmax = self.attention_c(
            A_without_softmax).squeeze(dim=2)  # N x n_classes
        # print('A_without_softmax ka shape after attention c', A_without_softmax.shape)
        A = F.softmax(A_without_softmax, dim=1).unsqueeze(dim=1)
        # print('A shape after softmax', A.shape)
        h = torch.bmm(A, x).squeeze(dim=1)
        # print('final h shape ', h.shape)
        return h, A.squeeze(dim=1), A_without_softmax

    # def forward(self, x):
    #     if x.dim() == 4:
    #         x = x.squeeze(1)  # From (1, 1, N, D) → (1, N, D)
    #     elif x.dim() == 2:
    #         x = x.unsqueeze(0)  # (1, N, D)

    #     a = self.attention_a(x)                # (B, N, d1)
    #     b = self.attention_b(x)                # (B, N, d1)
    #     A_raw = a * b                          # (B, N, d1)

    #     A_raw = self.attention_c(A_raw).squeeze(2)  # (B, N)
    #     A = F.softmax(A_raw, dim=1)                # (B, N)
    #     A = A.unsqueeze(1)                         # (B, 1, N)

    #     h = torch.bmm(A, x).squeeze(1)             # (B, D)

    #     return h, A.squeeze(1), A_raw              # (B, D), (B, N), (B, N)






class AMIL(nn.Module):
    def __init__(self, size_arg="small", input_size=1024, dropout_rate=0.1, n_classes=2, activation=None,as_backbone=False,**kwargs):
        super(AMIL, self).__init__()
        self.as_backbone = as_backbone
        hidden_size = 512
        if input_size>= 512:
            hidden_size = 512
        elif input_size >=384:
            hidden_size = 384
        else:
            raise NotImplementedError
        self.size_dict_path = {
            "small": [input_size, hidden_size , 256], "big": [input_size, 512, 384]}

        size = self.size_dict_path[size_arg]
       
        self.fc1 = nn.Sequential(
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.amil = AMIL_layer(
            L=size[1], D=size[2], dropout=dropout_rate, activation=activation)
        if not self.as_backbone:
            self.classifier = nn.Linear(size[1], n_classes)

    def forward(self, x,**kwargs):
        # print('ina mina dika')
        x = self.fc1(x)
        x, attention, attention_without_softmax = self.amil(x)
        if self.as_backbone:
            return x
        else:
            logits = self.classifier(x)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            Y_prob = F.softmax(logits, dim=1)

            res = {
                "logits": logits,
                "y_hat": Y_hat,
                'y_prob': Y_prob,
                'attention': attention,
                'attention_without_softmax': attention_without_softmax
            }
            return res
