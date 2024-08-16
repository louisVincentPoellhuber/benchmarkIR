from models import *
from adaptive_span import AdaptiveMask
from torch.autograd import gradcheck

input_tensor = torch.rand(16, 12, 512, 512, dtype=torch.float64, requires_grad=True)

mask = AdaptiveMask(max_size=1024, ramp_size=32)

test = gradcheck(AdaptiveMask.forward, (mask, input_tensor,), eps=1e-6, atol=1e-4)
print("Gradient check passed:", test)