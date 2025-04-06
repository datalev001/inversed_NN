import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.stats as st

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Define a complex nonlinear function combining sin, log, exp, and sqrt.
# This function is defined for x > 0.
def f(x):
    return torch.sin(x) - 0.3 * torch.log(x + 1) + 0.1 * torch.exp(-x / 3) + 0.2 * torch.sqrt(x) - 0.5

# Inverse NN: maps a latent code z (sampled uniformly from [0,1]) to a candidate solution x.
# We force the output to be positive using a softplus activation.
class InverseNN(nn.Module):
    def __init__(self):
        super(InverseNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 128),  # increased from 64 to 128
            nn.Tanh(),
            nn.Linear(128, 128),  # increased from 64 to 128
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, z):
        raw = self.model(z)
        return F.softplus(raw)  # ensures x > 0

# Instantiate network, optimizer, loss function, and learning rate scheduler.
net = InverseNN()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)

# The function assign_range partitions the latent space into three intervals and assigns
# a target range for x:
#   - For z < 1/3, target x in [0, 1.5]
#   - For 1/3 <= z < 2/3, target x in [1, 3.5]
#   - For z >= 2/3, target x in [5, 7.5]
def assign_range(z):
    lower = torch.zeros_like(z)
    upper = torch.zeros_like(z)
    
    mask1 = (z < 1/3)
    lower[mask1] = 0.0
    upper[mask1] = 1.5

    mask2 = (z >= 1/3) & (z < 2/3)
    lower[mask2] = 1.0
    upper[mask2] = 3.5

    mask3 = (z >= 2/3)
    lower[mask3] = 5.0
    upper[mask3] = 7.5

    return lower, upper

# Range penalty: if the predicted x is outside the assigned [lower, upper] interval,
# a quadratic penalty is added.
def range_penalty(x, lower, upper, lambda_range=10.0):
    penalty_lower = torch.clamp(lower - x, min=0) ** 2
    penalty_upper = torch.clamp(x - upper, min=0) ** 2
    return lambda_range * (penalty_lower + penalty_upper)

# Training parameters
batch_size = 128
epochs = 8000  # increased number of epochs
patience = 300
best_loss = float('inf')
wait = 0
best_model = None

# Adjusted loss weights
multiplier = 5.0   # Multiply the primary loss to emphasize f(x) ≈ 0
lambda_cluster = 30.0  # Clustering loss weight

# Training loop: optimize so that f(x_pred) ≈ 0, x_pred falls in target range,
# and x_pred is near the center of the interval.
for epoch in range(epochs):
    net.train()
    optimizer.zero_grad()
    
    # Sample latent codes uniformly in [0,1]
    z = torch.rand(batch_size, 1)
    x_pred = net(z)  # predicted x, forced to be > 0
    
    # Compute function value and primary loss: we want f(x_pred) ≈ 0.
    f_val = f(x_pred)
    loss_f = loss_fn(f_val, torch.zeros_like(f_val))
    
    # Get target ranges and compute range penalty.
    lower, upper = assign_range(z)
    penalty = range_penalty(x_pred, lower, upper, lambda_range=10.0)
    
    # Clustering penalty: force x_pred toward the center of the interval.
    target = (lower + upper) / 2.0
    cluster_loss = loss_fn(x_pred, target)
    
    # Total loss: emphasize the function loss by multiplying it.
    loss = multiplier * loss_f + penalty.mean() + lambda_cluster * cluster_loss
    
    loss.backward()
    optimizer.step()
    
    # Update learning rate scheduler
    scheduler.step(loss)
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model = net.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
            
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {loss.item():.8f}, Avg |f(x)|: {f_val.abs().mean().item():.8f}")

if best_model is None:
    best_model = net.state_dict()

net.load_state_dict(best_model)
net.eval()

# --------------------------
# Evaluation & Validation
# --------------------------
with torch.no_grad():
    # Sample a dense set of latent codes to obtain a collection of predicted x's
    z_samples = torch.linspace(0, 1, 300).unsqueeze(1)
    x_samples = net(z_samples)
    f_samples = f(x_samples)

z_np = z_samples.squeeze().numpy()
x_np = x_samples.squeeze().numpy()
f_np = f_samples.squeeze().numpy()

mae_f = np.mean(np.abs(f_np))
rmse_f = np.sqrt(np.mean(f_np**2))
print(f"\nOverall Mean Absolute Error of f(x): {mae_f:.6f}")
print(f"Overall RMSE of f(x): {rmse_f:.6f}\n")

print("Sample validation results:")
for i in np.linspace(0, 299, 5, dtype=int):
    print(f"z = {z_np[i]:.3f} -> Predicted x = {x_np[i]:.3f}, f(x) = {f_np[i]:.6f}")

# --------------------------
# Clustering of Predicted x's
# --------------------------
# Use KMeans to cluster the predicted x values into several segments.
# You can adjust the number of clusters (k) as needed; here we choose k = 3.
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
# Reshape x_np to a 2D array for clustering
x_np_reshaped = x_np.reshape(-1, 1)
clusters = kmeans.fit_predict(x_np_reshaped)

# For each cluster, compute average error and 95% confidence interval of the error.
print("\nCluster-wise error analysis:")
for cluster_id in range(k):
    # Extract indices for this cluster
    indices = np.where(clusters == cluster_id)[0]
    # Extract errors for this cluster (absolute error of f(x))
    cluster_errors = np.abs(f_np[indices])
    # Compute mean error and standard error
    mean_error = np.mean(cluster_errors)
    std_error = np.std(cluster_errors, ddof=1) / np.sqrt(len(cluster_errors))
    # Determine t-critical value for 95% confidence interval
    t_crit = st.t.ppf(0.975, df=len(cluster_errors)-1)
    ci = t_crit * std_error
    print(f"Cluster {cluster_id}:")
    print(f"  Number of samples: {len(cluster_errors)}")
    print(f"  Mean absolute error: {mean_error:.6f}")
    print(f"  95% confidence interval: [{mean_error - ci:.6f}, {mean_error + ci:.6f}]\n")

# --------------------------
# Visualization
# --------------------------
# 1. Mapping from latent code z to predicted x.
plt.figure(figsize=(10, 6))
plt.scatter(z_np, x_np, c=clusters, cmap='viridis', s=20, label="Predicted x (colored by cluster)")
plt.xlabel("Latent code z")
plt.ylabel("Predicted x")
plt.title("Mapping from Latent Code to Predicted x (Clustered)")
plt.colorbar(label="Cluster ID")
plt.grid(True)
plt.legend()
plt.show()

# 2. Plot f(x) over x values and mark predicted x clusters.
x_plot = np.linspace(0.001, 8, 500)
x_plot_tensor = torch.tensor(x_plot, dtype=torch.float32).unsqueeze(1)
f_plot = f(x_plot_tensor).detach().numpy().squeeze()

plt.figure(figsize=(10, 6))
plt.plot(x_plot, f_plot, label="f(x)")
plt.axhline(0, color='black', lw=0.5, label="Target f(x)=0")
# Plot vertical lines for each cluster's centroid
for cluster_id in range(k):
    centroid = kmeans.cluster_centers_[cluster_id][0]
    plt.axvline(centroid, color='red', linestyle='--', alpha=0.7, label=f"Cluster {cluster_id} centroid" if cluster_id==0 else None)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("f(x) and Predicted Roots with Cluster Centroids")
plt.legend()
plt.show()

# 3. Plot error f(x) vs. latent code z.
plt.figure(figsize=(10, 6))
plt.scatter(z_np, f_np, c=clusters, cmap='viridis', s=20)
plt.xlabel("Latent code z")
plt.ylabel("f(Predicted x)")
plt.title("Error of f(x) vs. Latent Code (Colored by Cluster)")
plt.axhline(0, color='black', lw=0.5)
plt.colorbar(label="Cluster ID")
plt.grid(True)
plt.show()

# 4. Print sorted predicted x-values.
print("\nPredicted x-values (sorted):")
print(np.sort(x_np))

# --------------------------
# Additional Code: Top 5 Smallest Absolute Errors and Plotting
# --------------------------

# Compute absolute errors from f(x) for each predicted x estimate.
abs_errors = np.abs(f_np)

# Find the indices of the top 5 smallest absolute errors.
top5_indices = np.argsort(abs_errors)[:5]
top5_x = x_np[top5_indices]
top5_errors = f_np[top5_indices]

# Print the top 5 smallest absolute error values and their corresponding x estimates.
print("\nTop 5 smallest absolute errors:")
for i, idx in enumerate(top5_indices):
    print(f"Rank {i+1}: x = {x_np[idx]:.6f}, f(x) = {f_np[idx]:.6f}")

# Plot all x estimates vs. their error values, and overlay red dots for the top 5 smallest errors.
plt.figure(figsize=(10, 6))
plt.scatter(x_np, f_np, color='blue', s=20, label='All x estimates')
plt.scatter(top5_x, top5_errors, color='red', s=50, label='Top 5 smallest abs error')
plt.xlabel("Predicted x")
plt.ylabel("f(Predicted x) Error")
plt.title("Error of f(x) vs. Predicted x with Top 5 Smallest Errors Highlighted")
plt.axhline(0, color='black', lw=0.5)
plt.legend()
plt.grid(True)
plt.show()