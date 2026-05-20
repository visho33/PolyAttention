import torch

class Batch:

    def __init__(self, t, n, seed = 33, device = "cpu"):
        self.t = t
        self.n = n
        self.seed = seed
        self.device = torch.device(device)
        self.calls = 0
        self.seq_len = self.t*self.n + 1

    def function_composition(self, batch_size = 32):
        
        seed = self.seed + self.calls
        self.calls += 1

        generator = torch.Generator(device = self.device)
        generator.manual_seed(seed)

        starts = torch.randint(0, self.n, (batch_size,), generator = generator, device = self.device)
        functions = torch.randint(0, self.n, (batch_size, self.t, self.n), generator = generator, device = self.device)

        targets = starts.clone()
        batch_idx = torch.arange(batch_size, device = self.device)
        for step in range(self.t):
            targets = functions[batch_idx, step, targets]

        tokens = torch.zeros(batch_size, self.seq_len, dtype = torch.long, device = self.device)
        tokens[:, : self.t*self.n] = functions.reshape(batch_size, self.t*self.n)
        tokens[:, -1] = starts

        positions = torch.arange(self.seq_len, device = self.device).expand(batch_size, self.seq_len)
        x = torch.stack([tokens, positions], dim = -1)
        
        return x, targets

    def make_batch(self, batch_size):
        x, targets = self.function_composition(batch_size)
        return x.cpu().numpy(), targets.cpu().numpy()
