r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from .layers import FMoE
from .linear import FMoELinear
from .fastermoe.config import switch_from_env


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x

class _ExpertTORCH(nn.Module):
    def __init__(self, hidden_size, activation, d_model, emulate_mole=False):
        super().__init__()
        self.emulate_mole = emulate_mole
        self.htoh4 = torch.nn.Linear(d_model, hidden_size, bias=True)
        self.h4toh = torch.nn.Linear(hidden_size, d_model, bias=True)
        self.activation = activation
        if emulate_mole:
            self.w1 = torch.empty_like(self.htoh4.weight, device="cpu")
            self.w2 = torch.empty_like(self.h4toh.weight, device="cpu")
            with torch.no_grad():
                self.w1.copy_(self.htoh4.weight)
                self.w2.copy_(self.h4toh.weight)

    def forward(self, inp, fwd_expert_count=None):
        if self.emulate_mole:
            with torch.no_grad():
                self.htoh4.weight.data.copy_(self.w1.data)
                self.h4toh.weight.data.copy_(self.w2.data)

        x = self.htoh4(inp)
        x = self.activation(x)
        x = self.h4toh(x)
        return x
    
class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        **kwargs
    ):
        #def one_expert(d_model): #_Expert converge worse
        #    return _Expert(1, d_model, d_hidden, activation, rank=0)
        def one_expert(d_model):
            return _ExpertTORCH(d_hidden, activation, d_model, emulate_mole=kwargs.get("emulate_mole", False))
        expert = one_expert
        super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, **kwargs)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)
