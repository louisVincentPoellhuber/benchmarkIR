from models import *


model = RobertaForMaskedLM(config)


print(f"Gradient before scaling: {model.alpha.grad}")

def scale_gradients(grad):
    return grad * 1000  # Example scaling factor

model.alpha.register_hook(scale_gradients)

# Perform forward and backward pass
loss = loss_fn(output, target)
loss.backward()

print(f"Scaled gradient: {model.alpha.grad}")

optimizer.step()
optimizer.zero_grad()

print(f"Alpha parameter after step: {model.alpha}")
