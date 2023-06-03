import torch
import torch.nn.functional as F
import torch.nn as nn

class BaseGate(nn.Module):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None


class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k

    def forward(self, inp, return_all_scores=False):
        pass



class TutelGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size,
            topk=1,gate_capacity=10000):
        super().__init__(d_model, num_expert, world_size, top_k=topk)
        self.capacity = (gate_capacity, gate_capacity)

    def forward(self, x):
        x = x.float()
        wg = self.gate.float()

        gate = wg(x)
        
        out = F.softmax(gate, dim=1)
        
        return None, out