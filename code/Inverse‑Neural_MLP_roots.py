# ----------------- MLP‑based inverse neural net -----------------
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
eps = 1e-6
x_vals = np.linspace(0.001, 10, 6000, dtype=np.float32).reshape(-1,1)
y_vals = np.sin(x_vals)/(x_vals+eps)
x_train, y_train = torch.tensor(x_vals), torch.tensor(y_vals)

class InverseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,64),  nn.Tanh(),
            nn.Linear(64,1)
        )
    def forward(self, y): return self.model(y)

net = InverseNN()
opt = optim.Adam(net.parameters(), lr=2e-3)
mse = nn.MSELoss();  λ = 1e-4

for epoch in range(10_000):
    opt.zero_grad()
    x_pred = net(y_train)
    y_pred = torch.sin(x_pred)/(x_pred+eps)
    loss = mse(y_pred, y_train) + λ*(x_pred**2).mean()
    loss.backward(); opt.step()
    if (epoch+1) % 400 == 0:
        print(f"Epoch {epoch+1:4d} | loss_f = {mse(y_pred,y_train):.6e}")

net.eval()
y_star = torch.tensor([[0.95]], dtype=torch.float32)
x_star = net(y_star).item()
print("\n🧪  Target y=0.95  →  x ≈", x_star)
print("Check  sin(x)/x =", np.sin(x_star)/x_star)