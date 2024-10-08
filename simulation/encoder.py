import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from parameters import generate_parameters
from qd import generate_quantum_dot_spectrum
from vary_parameters import generate_parameter_variators
from generate import generate_data, generate_qd_data
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

latent_space_size = 32
num_epochs = 50
num_samples = 200
seq_length = 1000
batch_size = 512

data = generate_data(num_samples, seq_length)
qd_data, parameters = generate_qd_data(num_samples, seq_length)
data = qd_data
data_mean = data.mean()
data_std = data.std()
data = (data - data_mean) / data_std
data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
data = data.to(device)

batch_size = 32
dataset = TensorDataset(data, data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Conv1dAutoEncoder(nn.Module):
    def __init__(self, seq_length, latent_space_size):
        super(Conv1dAutoEncoder, self).__init__()
        self.latent_space_size = latent_space_size

        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(True)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_length)
            e1 = self.enc1(dummy_input)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            self.enc_output_shape = e3.shape
            self.enc_output_size = e3.numel()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.enc_output_size, self.latent_space_size)

        self.fc2 = nn.Linear(self.latent_space_size, self.enc_output_size)
        self.unflatten = nn.Unflatten(1, self.enc_output_shape[1:])

        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64 * 2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32 * 2, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.dec3_no_skip = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(True)
        )
        self.dec2_no_skip = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(True)
        )
        self.dec1_no_skip = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        f = self.flatten(e3)
        latent = self.fc1(f)
        return latent

    def decode(self, latent):
        f = self.fc2(latent)
        d3_input = self.unflatten(f)
        d3 = self.dec3_no_skip(d3_input)
        d2 = self.dec2_no_skip(d3)
        d1 = self.dec1_no_skip(d2)
        return d1

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        f = self.flatten(e3)
        latent = self.fc1(f)
        f = self.fc2(latent)
        d3_input = self.unflatten(f)
        d3 = self.dec3(d3_input)
        d3 = torch.cat((d3, e2), dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat((d2, e1), dim=1)
        d1 = self.dec1(d2)
        return d1

model = Conv1dAutoEncoder(seq_length=seq_length, latent_space_size=latent_space_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(num_epochs):
    running_loss_with_skip = 0.0
    running_loss_no_skip = 0.0
    for inputs, _ in dataloader:
        optimizer.zero_grad()
        outputs_with_skip = model(inputs)
        latent = model.encode(inputs)
        outputs_no_skip = model.decode(latent)

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
    plt.xlim([1583.5, 1587.5])
    plt.plot(parameters['energies'], test_inputs_np[i], label='Original', linestyle='--')
    plt.plot(parameters['energies'], test_outputs_with_skip_np[i], label='Reconstructed (with skip connections)', alpha=0.7)
    plt.plot(parameters['energies'], test_outputs_no_skip_np[i], label='Reconstructed (from latent only)', alpha=0.7)
    plt.legend()
    plt.title(f'Sample {i+1}')
    plt.show()