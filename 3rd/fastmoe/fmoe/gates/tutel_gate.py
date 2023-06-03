import torch
import torch.nn.functional as F
from .naive_gate import NaiveGate

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