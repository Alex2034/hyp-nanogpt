import torch
from model.lorentz import LorentzDot

u = torch.randn(32, 128, 1, 3, requires_grad=True)
v = torch.randn(1, 1, 72, 3, requires_grad=True)

# Forward pass
output = LorentzDot.apply(u, v)

# Create a dummy loss
loss = output.sum()

# Backward pass
loss.backward()

# Check gradients
print("\nTesting LorentzDot backward pass:")
print(f"u.grad shape: {u.grad.shape}")
print(f"v.grad shape: {v.grad.shape}")

# Verify gradients are not None
assert u.grad is not None, "u.grad should not be None"
assert v.grad is not None, "v.grad should not be None"

# Verify gradient shapes match input shapes
assert u.grad.shape == u.shape, "u.grad shape should match u shape"
assert v.grad.shape == v.shape, "v.grad shape should match v shape"
