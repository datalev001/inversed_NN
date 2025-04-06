import sys, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from kan import KAN          # Kolmogorov–Arnold Network

print("Python exe:", sys.executable)
torch.manual_seed(0);  np.random.seed(0)

# ------------------ 1. residual:  sin(x)/x - y ------------------
eps = 1e-6                         # numerical safety
def residual(x, y):                # vectorised
    return torch.sin(x) / (x + eps) - y

# ------------------ 2. build inverse KAN:  y -> x ---------------
def build_inverse_kan():
    return KAN([1, 32, 32, 1])     # 1‑D in, two hidden layers, 1‑D out

inverse_net = build_inverse_kan()
optimiser   = optim.Adam(inverse_net.parameters(), lr=1e-3)
loss_fn     = nn.MSELoss()

# ------------------ 3. constant training target  y = 0.95 -------
y_const  = 0.95
y_train  = torch.full((2000, 1), y_const, dtype=torch.float32)

# ------------------ 4. training loop with early stopping --------
epochs, patience = 4000, 50
best_loss, wait, best_state = float('inf'), 0, None

for epoch in range(epochs):
    inverse_net.train()
    optimiser.zero_grad()

    x_pred = inverse_net(y_train)                 # G(y)
    res    = residual(x_pred, y_train)            # sin(x)/x - y
    loss   = loss_fn(res, torch.zeros_like(res))  # want residual → 0
    loss.backward();  optimiser.step()

    # ---- early‑stopping bookkeeping
    if loss.item() < best_loss:
        best_loss, best_state, wait = loss.item(), inverse_net.state_dict(), 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered."); break

    if (epoch+1) % 400 == 0:
        print(f"epoch {epoch+1:4d} | loss {loss.item():.6e}")

# restore best weights
if best_state is not None:
    inverse_net.load_state_dict(best_state)

# ------------------ 5. inference -----------------
inverse_net.eval()
y_test   = torch.tensor([[y_const]], dtype=torch.float32)
x_root   = inverse_net(y_test).item()
y_check  = float(torch.sin(torch.tensor(x_root)) / x_root)

print(f"\n🧪  target y = {y_const}")
print(f"predicted root  x ≈ {x_root: .10f}")
print(f"check  sin(x)/x = {y_check: .10f}")

# ------------------ 6. quick visual sanity check ---------------
with torch.no_grad():
    y_plot = torch.linspace(0.3, 1.0, 120).view(-1, 1)
    x_plot = inverse_net(y_plot).numpy()

plt.figure(figsize=(8,4))
plt.plot(y_plot.numpy(), x_plot, label="KAN inverse  x = G(y)")
plt.axvline(y_const, color='r', ls='--', alpha=.4)
plt.xlabel("y"); plt.ylabel("x")
plt.title(r"KAN inverse net  solving  $\sin(x)/x = y$")
plt.grid(); plt.legend(); plt.tight_layout(); plt.show()

