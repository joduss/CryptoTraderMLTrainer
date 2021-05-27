#%%
# Refreshing memory on gradient computation
# Usage of

import torch


x = torch.tensor([2.3])
y = 2 * x

w = torch.tensor([2.1], requires_grad=True)


y_pred = w * x
loss = (y_pred - y).pow(2).sum()


loss.backward()

w_grad = w.grad

print(loss)
print(f"Auto Gradient {w_grad}")

# (wx - y)^2
# w^2*x^2 - 2wxy + y^2
# gradient of loss  w => 2wx^2 - 2x^2*y
with torch.no_grad():
    manual_grad = 2 * w * x ** 2 - 2 * x * y
    print(f"Manual gradient {manual_grad}")