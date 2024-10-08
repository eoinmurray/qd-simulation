import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Generate synthetic data for demonstration purposes
def generate_synthetic_data(num_samples, seq_length):
    # Simple sine waves with added noise
    x = np.linspace(0, 4 * np.pi, seq_length)
    data = []
    for _ in range(num_samples):
        freq = np.random.uniform(0.1, 1.0)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.5, 1.5)
        signal = amplitude * np.sin(freq * x + phase)
        noise = np.random.normal(0, 0.1, seq_length)
        data.append(signal + noise)
    return np.array(data)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_space_size = 16  # Updated to match the paper
num_epochs = 50
num_samples = 200
seq_length = 1000
batch_size = 32

# Generate data
data = generate_synthetic_data(num_samples, seq_length)
data_mean = data.mean()
data_std = data.std()
data = (data - data_mean) / data_std
data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
data = data.to(device)

dataset = TensorDataset(data, data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Residual block as per paper's structure
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
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

        # Encoder: residual blocks with adjusted pooling
        self.enc1 = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.enc4 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Calculate the size of the encoder output
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_length)
            e1 = self.enc1(dummy_input)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            self.enc_output_shape = e4.shape  # Save shape for decoder
            self.enc_output_size = e4.numel()  # Flattened size

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
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=3, output_padding=1),
            nn.Sigmoid()  # Output scaled between 0 and 1
        )

        # Adjust the final output size to match input size
        self.output_adjust = nn.ConstantPad1d((0, seq_length - self.calculate_output_seq_length()), 0)

    def calculate_output_seq_length(self):
        # Calculate the output sequence length after the decoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_length)
            e1 = self.enc1(dummy_input)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            f = self.flatten(e4)
            latent = self.fc1(f)
            f = self.fc2(latent)
            d4_input = self.unflatten(f)
            d4 = self.dec4(d4_input)
            d3 = self.dec3(d4)
            d2 = self.dec2(d3)
            d1 = self.dec1(d2)
            output_seq_length = d1.shape[2]
        return output_seq_length

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
        # Adjust output size
        output = self.output_adjust(d1)
        return output

    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)
        return output

# Initialize model
model = Conv1dAutoEncoder(seq_length=seq_length, latent_space_size=latent_space_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    running_loss_with_skip = 0.0
    running_loss_no_skip = 0.0
    for inputs, _ in dataloader:
        optimizer.zero_grad()
        outputs_with_skip = model(inputs)
        latent = model.encode(inputs)
        outputs_no_skip = model.decode(latent)

        # Compute loss
        loss_with_skip = criterion(outputs_with_skip, inputs)
        loss_no_skip = criterion(outputs_no_skip, inputs)

        total_loss = loss_with_skip + loss_no_skip

        total_loss.backward()
        optimizer.step()
        running_loss_with_skip += loss_with_skip.item()
        running_loss_no_skip += loss_no_skip.item()
    
    average_loss_with_skip = running_loss_with_skip / len(dataloader)
    average_loss_no_skip = running_loss_no_skip / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss with skip: {average_loss_with_skip:.6f}, Loss without skip: {average_loss_no_skip:.6f}")

# Test and plot results
test_inputs, _ = next(iter(dataloader))

test_outputs_with_skip = model(test_inputs)
latent_reps = model.encode(test_inputs)
test_outputs_no_skip = model.decode(latent_reps)

test_inputs_np = test_inputs.squeeze().detach().cpu().numpy()
test_outputs_with_skip_np = test_outputs_with_skip.squeeze().detach().cpu().numpy()
test_outputs_no_skip_np = test_outputs_no_skip.squeeze().detach().cpu().numpy()

n = 5
for i in range(n):
    plt.figure(figsize=(12, 4))
    plt.plot(test_inputs_np[i], label='Original', linestyle='--')
    plt.plot(test_outputs_with_skip_np[i], label='Reconstructed (with skip connections)', alpha=0.7)
    plt.plot(test_outputs_no_skip_np[i], label='Reconstructed (from latent only)', alpha=0.7)
    plt.legend()
    plt.title(f'Sample {i+1}')
    plt.show()