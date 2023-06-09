'''Copyright The Microsoft DeepSpeed Team'''

from .sparsity_config import SparsityConfig, DenseSparsityConfig, FixedSparsityConfig, VariableSparsityConfig, BigBirdSparsityConfig, BSLongformerSparsityConfig, LocalSlidingWindowSparsityConfig
from .sparse_self_attention import SparseSelfAttention
from .bert_sparse_self_attention import BertSparseSelfAttention
from .sparse_attention_utils import SparseAttentionUtils
