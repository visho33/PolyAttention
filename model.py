import math
import torch
import torch.nn as nn

class TreeAttention(nn.Module):

    def __init__(self, d, edges, heads = 1, dropout = 0.0, clip = 20.0):
        super().__init__()
        
        if d%heads != 0:
            raise ValueError("d must be divisible by heads")
        
        self.d = d
        self.heads = heads
        self.d_head = d//heads
        self.edges = [list(children) for children in edges]
        self.root = self.find_root(self.edges)
        self.num_nodes = len(self.edges)
        self.score_clip = clip

        self.WQ = nn.ModuleList(nn.Linear(d, d, bias = False) for _ in range(self.num_nodes))
        self.WV = nn.ModuleList(nn.Linear(d, d, bias = False) for _ in range(self.num_nodes))
        self.WO = nn.Linear(d, d, bias = False) if heads > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        batch, seq_len, _ = x.shape
        q = [self.split_heads(wq(x)) for wq in self.WQ]
        v = [self.split_heads(wv(x)) for wv in self.WV]

        P, R = self.dfs(self.root, q, v, batch, seq_len, x.device, x.dtype)
        
        y = P/R.clamp_min(1e-9)
        y = y.transpose(1, 2).reshape(batch, seq_len, self.d)
        
        return self.WO(y)

    def dfs(self, node, q, v, batch, seq_len, device, dtype):
        
        branch_P = []
        branch_R = []

        for child in self.edges[node]:
            
            child_P, child_R = self.dfs(child, q, v, batch, seq_len, device, dtype)

            scores = q[node]@q[child].transpose(-2, -1)
            scores = scores/math.sqrt(self.d_head)
            scores = scores.clamp(min = -self.score_clip, max = self.score_clip)
            weights = self.dropout(torch.exp(scores))

            P = weights@(v[child]*child_P)
            R = weights@child_R

            branch_P.append(P)
            branch_R.append(R)

        if not branch_P:
            P = torch.ones(batch, self.heads, seq_len, self.d_head, device = device, dtype = dtype)
            R = torch.ones(batch, self.heads, seq_len, 1, device = device, dtype = dtype)
            return P, R

        P = branch_P[0]
        R = branch_R[0]
        for P_i, R_i in zip(branch_P[1:], branch_R[1:]):
            P = P*P_i
            R = R*R_i

        return P, R

    def split_heads(self, x):
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.heads, self.d_head).transpose(1, 2)

    @staticmethod
    def find_root(edges):
        
        indegree = [0 for _ in edges]
        
        for children in edges:
            for child in children:
                if child < 0 or child >= len(edges):
                    raise ValueError(f"Child index out of range: {child}")
                indegree[child] += 1

        roots = [node for node, degree in enumerate(indegree) if degree == 0]
        if len(roots) != 1:
            raise ValueError(f"Directed tree edges must have exactly one root")
        
        return roots[0]

class TreeAttentionBlock(nn.Module):

    def __init__(self, d, edges, heads = 1, d_ff = None, dropout = 0.0):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4*d

        self.attention = TreeAttention(d, edges, heads = heads, dropout = dropout)
        self.norm1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        return self.norm2(x + self.dropout(self.mlp(x)))

class FunctionCompositionTreeModel(nn.Module):

    def __init__(self, t, n, d, heads, layers = 1, d_ff = None, dropout = 0.0, edges = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4*d
        if edges is None:
            edges = [[node + 1] if node + 1 < t+1 else [] for node in range(t+1)]

        self.seq_len = t*n + 1
        self.value_embedding = nn.Embedding(n, d)
        self.position_embedding = nn.Embedding(self.seq_len, d)
        self.blocks = nn.ModuleList(TreeAttentionBlock(d, edges, heads = heads, d_ff = d_ff, dropout = dropout) for _ in range(layers))
        self.output = nn.Linear(d, n)

    def forward(self, x):
        h = self.value_embedding(x[:, :, 0]) + self.position_embedding(x[:, :, 1])
        for block in self.blocks:
            h = block(h)
        return self.output(h[:, -1]).clamp(min = -20, max = 20)
