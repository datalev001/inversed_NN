# =============================================================
# Inverse‑Neural‑Net Demo
# Build → Train → Segment customer profiles
# =============================================================
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------------------------------------
# 0.  Synthetic customer data  (6 534 rows, four numeric fields)
# -------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

# load data
df = pd.read_csv('spend.csv')

# -------------------------------------------------------------
# 1.  Train / test split and scaling
# -------------------------------------------------------------
X = df[['age','education','income']].values.astype(np.float32)
y = df[['spend']].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=1)

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

Xtr = torch.tensor(x_scaler.transform(X_train))
Xte = torch.tensor(x_scaler.transform(X_test))
Ytr = torch.tensor(y_scaler.transform(y_train))
Yte = torch.tensor(y_scaler.transform(y_test))

# tensor bounds (scaled) for the box penalty
x_min_t = torch.tensor(
    x_scaler.transform(np.array([[18, 1, 0]], dtype=np.float32))[0])
x_max_t = torch.tensor(
    x_scaler.transform(np.array([[93,10,300_000]], dtype=np.float32))[0])

# -------------------------------------------------------------
# 2.  Forward model  F : X → spend
# -------------------------------------------------------------
class ForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

F_net = ForwardNet()
opt_F = optim.Adam(F_net.parameters(), lr=1e-3)
loss_mse = nn.MSELoss()

for epoch in range(2000):
    F_net.train(); opt_F.zero_grad()
    loss = loss_mse(F_net(Xtr), Ytr)
    loss.backward(); opt_F.step()
    if (epoch+1) % 400 == 0:
        print(f"F‑net epoch {epoch+1:4d} | loss {loss.item():.4f}")

F_net.eval()
r2 = 1 - loss_mse(F_net(Xte), Yte).item() / Yte.var().item()
print(f"\nForward model R² on test: {r2:.3%}")

# -------------------------------------------------------------
# 3.  Inverse model  G : spend → X
# -------------------------------------------------------------
class InverseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )
    def forward(self, y): return self.net(y)

G_net = InverseNet()
opt_G = optim.Adam(G_net.parameters(), lr=1e-3)

w_recon = 1.0     # match true X (only for training pairs)
w_cycle = 1.0     # F(G(y)) ≈ y
w_box   = 10.0    # keep X within valid range

def box_penalty(x_pred):
    xmin = x_min_t.to(x_pred)
    xmax = x_max_t.to(x_pred)
    below = F.relu(xmin - x_pred)**2
    above = F.relu(x_pred - xmax)**2
    return (below + above).mean()

for epoch in range(4000):
    G_net.train(); opt_G.zero_grad()

    x_hat   = G_net(Ytr)          # predicted X (scaled)
    y_cycle = F_net(x_hat)        # round‑trip spend

    loss_recon = loss_mse(x_hat, Xtr)
    loss_cycle = loss_mse(y_cycle, Ytr)
    loss_range = box_penalty(x_hat)

    loss = w_recon*loss_recon + w_cycle*loss_cycle + w_box*loss_range
    loss.backward(); opt_G.step()

    if (epoch+1) % 400 == 0:
        print(f"G‑net epoch {epoch+1:4d} | total loss {loss.item():.4f}")

# -------------------------------------------------------------
# 4.  Evaluate inverse performance
# -------------------------------------------------------------
G_net.eval()
with torch.no_grad():
    x_pred_test  = G_net(Yte)
    y_cycle_test = F_net(x_pred_test)

inv_rmse   = torch.sqrt(loss_mse(x_pred_test, Xte)).item()
cycle_rmse = torch.sqrt(loss_mse(y_cycle_test, Yte)).item()
print(f"\nInverse RMSE on X (scaled): {inv_rmse:.4f}")
print(f"Cycle RMSE on y (scaled)  : {cycle_rmse:.4f}")

# convert predictions back to original units
X_pred_real = x_scaler.inverse_transform(x_pred_test.numpy())
df_pred = pd.DataFrame(X_pred_real, columns=['age','education','income'])

# -------------------------------------------------------------
# 5.  Customer segmentation
# -------------------------------------------------------------
k = 4
kmeans = KMeans(n_clusters=k, random_state=0).fit(X_pred_real)
df_pred['segment'] = kmeans.labels_

print("\nSegment profiles (means):")
print(df_pred.groupby('segment').mean().round(1))

print("\nSegment counts:", df_pred['segment'].value_counts().to_dict())

# ---- scatter for TRUE data

# PCA projection of real X
pca_real = PCA(n_components=2)
df_real[['pc1','pc2']] = pca_real.fit_transform(df_real[['age','education','income']])

plt.figure(figsize=(8,6))
colors = ['tab:red','tab:blue','tab:green','tab:orange']
for seg in sorted(df_real['segment'].unique()):
    sub = df_real[df_real['segment'] == seg]
    plt.scatter(sub['pc1'], sub['pc2'], s=20, alpha=0.6, color=colors[seg],
                label=f"Real Segment {seg}")
plt.title("Real X Segment Clusters (PCA Projection)")
plt.xlabel("PCA Component 1"); plt.ylabel("PCA Component 2")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

kmeans_real = KMeans(n_clusters=4, random_state=0).fit(X)  # 3 segments
df_real = pd.DataFrame(X, columns=['age','education','income'])
df_real['segment'] = kmeans_real.labels_

print("\nSegment profiles (means) from REAL X:")
print(df_real.groupby('segment').mean().round(1))

print("\nSegment counts from REAL X:", df_real['segment'].value_counts().to_dict())

# -------------------------------------------------------------
# 6.  Scatter plot: Predicted X with canonical cluster layout
# -------------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# PCA to project 3D predicted X to 2D for visualization
pca = PCA(n_components=2)
coords = pca.fit_transform(X_pred_real)
df_pred['pc1'], df_pred['pc2'] = coords[:, 0], coords[:, 1]

# Assign colors for segments
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

# Plot the predicted X in PCA space, colored by inferred cluster
plt.figure(figsize=(8, 6))
for seg in sorted(df_pred['segment'].unique()):
    seg_data = df_pred[df_pred['segment'] == seg]
    plt.scatter(seg_data['pc1'], seg_data['pc2'],
                color=colors[seg % len(colors)],
                label=f"Predicted Segment {seg}",
                alpha=0.6, s=25)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Predicted X from INN (PCA Projection)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''
F‑net epoch  400 | loss 0.1982
F‑net epoch  800 | loss 0.1967
F‑net epoch 1200 | loss 0.1951
F‑net epoch 1600 | loss 0.1931
F‑net epoch 2000 | loss 0.1907

Forward model R² on test: 78.385%
G‑net epoch  400 | total loss 0.3129
G‑net epoch  800 | total loss 0.3119
G‑net epoch 1200 | total loss 0.3117
G‑net epoch 1600 | total loss 0.3115
G‑net epoch 2000 | total loss 0.3113
G‑net epoch 2400 | total loss 0.3110
G‑net epoch 2800 | total loss 0.3108
G‑net epoch 3200 | total loss 0.3107
G‑net epoch 3600 | total loss 0.3106
G‑net epoch 4000 | total loss 0.3105

Inverse RMSE on X (scaled): 0.5556
Cycle RMSE on y (scaled)  : 0.0945

Segment profiles (means):
               age  education         income
segment                                     
0        66.699997        7.8  120146.101562
1        44.500000        3.9   66921.000000
2        54.299999        5.9   92030.203125
3        32.799999        2.0   38344.500000

Segment counts: {1: 461, 2: 452, 0: 213, 3: 181}

Segment profiles (means) from REAL X:
               age  education         income
segment                                     
0        33.500000        2.4   41137.601562
1        55.599998        6.0   93128.203125
2        68.000000        7.9  120643.601562
3        45.299999        4.3   69615.000000

'''