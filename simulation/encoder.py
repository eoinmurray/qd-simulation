import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from parameters import generate_parameters
from qd import generate_quantum_dot_spectrum
from generate import generate_data, generate_qd_data
import matplotlib.pyplot as plt

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_space_size = 16
num_epochs = 500
num_samples = 1000  # Increased from 200
seq_length = 1000
batch_size = 512
learning_rate = 0.0005  # Adjusted learning rate

# Generate data
data = generate_data(num_samples, seq_length)
qd_data, parameters = generate_qd_data(num_samples, seq_length)
data = qd_data

# Normalize data to have zero mean and unit variance
data_mean = data.mean()
data_std = data.std()
data = (data - data_mean) / data_std
data = torch.tensor(data, dtype=torch.float32).unsqueeze(1).to(device)

# Create dataset and dataloader
dataset = TensorDataset(data, data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Residual block as per paper's structure
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
                    if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return torch.relu(out)

# Updated Autoencoder architecture based on the paper
class Conv1dAutoEncoder(nn.Module):
    def __init__(self, seq_length, latent_space_size):
        super(Conv1dAutoEncoder, self).__init__()
        self.latent_space_size = latent_space_size

        # Encoder: residual blocks with max pooling
        self.enc1 = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool1d(kernel_size=3)
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool1d(kernel_size=3)
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool1d(kernel_size=3)
        )
        self.enc4 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool1d(kernel_size=3)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_length)
            e1 = self.enc1(dummy_input)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            self.enc_output_shape = e4.shape
            self.enc_output_size = e4.numel()

        # Fully connected layers for latent space encoding and decoding
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.enc_output_size, latent_space_size)

        self.fc2 = nn.Linear(latent_space_size, self.enc_output_size)
        self.unflatten = nn.Unflatten(1, self.enc_output_shape[1:])

        # Decoder: Transposed convolutions for upsampling
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=3, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=3),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=3),
            nn.BatchNorm1d(32),
            nn.ReLU(True)
        )
        # Removed Sigmoid activation
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=3, output_padding=1)
        )

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        f = self.flatten(e4)
        latent = self.fc1(f)
        return latent

    def decode(self, latent):
        f = self.fc2(latent)
        d4_input = self.unflatten(f)
        d4 = self.dec4(d4_input)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        return d1

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

# Initialize model
model = Conv1dAutoEncoder(seq_length=seq_length, latent_space_size=latent_space_size).to(device)

# Define loss function and optimizer with adjusted learning rate and weight decay
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training loop with simplified loss calculation
losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, _ in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)
    losses.append(average_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.6f}")

# Plot loss curve
plt.figure()
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()

# Test and plot results
test_inputs, _ = next(iter(dataloader))
test_inputs = test_inputs.to(device)
test_outputs = model(test_inputs)

test_inputs_np = test_inputs.squeeze().detach().cpu().numpy()
test_outputs_np = test_outputs.squeeze().detach().cpu().numpy()

n = 5
for i in range(n):
    plt.figure(figsize=(12, 4))
    plt.xlim([1583.5, 1587.5])  # Adjust this range based on your data
    plt.plot(parameters['energies'], test_inputs_np[i], label='Original', linestyle='--')
    plt.plot(parameters['energies'], test_outputs_np[i], label='Reconstructed', alpha=0.7)
    plt.legend()
    plt.title(f'Sample {i+1}')
    plt.show()