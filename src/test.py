
import torch
from entmax.root_finding import EntmaxBisectFunction

X = torch.randn(16, 12, 512, 512, requires_grad=True)
alpha = torch.tensor([1.5] * 12, requires_grad=True)

output = EntmaxBisectFunction.apply(X, alpha.view(1, 12, 1, 1))
loss = output.sum()
loss.backward()

print(alpha.grad)  # Should not be None
