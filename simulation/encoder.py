import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from parameters import generate_parameters
from qd import generate_quantum_dot_spectrum
from generate import generate_data, generate_qd_data
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_space_size = 16
num_epochs = 50
num_samples = 1000
seq_length = 1000
batch_size = 512
learning_rate = 0.0005

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

# Residual Block class
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
        out += residual
        return torch.relu(out)

# Autoencoder class
class Conv1dAutoEncoder(nn.Module):
    def __init__(self, seq_length, latent_space_size):
        super(Conv1dAutoEncoder, self).__init__()
        self.latent_space_size = latent_space_size

        # Encoder
        self.enc1 = nn.Sequential(ResidualBlock(1, 32), nn.MaxPool1d(kernel_size=3))
        self.enc2 = nn.Sequential(ResidualBlock(32, 64), nn.MaxPool1d(kernel_size=3))
        self.enc3 = nn.Sequential(ResidualBlock(64, 128), nn.MaxPool1d(kernel_size=3))
        self.enc4 = nn.Sequential(ResidualBlock(128, 256), nn.MaxPool1d(kernel_size=3))

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_length)
            e1 = self.enc1(dummy_input)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            self.enc_output_shape = e4.shape
            self.enc_output_size = e4.numel()

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.enc_output_size, latent_space_size)

        self.fc2 = nn.Linear(latent_space_size, self.enc_output_size)
        self.unflatten = nn.Unflatten(1, self.enc_output_shape[1:])

        # Decoder
        self.dec4 = nn.Sequential(nn.ConvTranspose1d(256, 128, kernel_size=3, stride=3, output_padding=1), nn.BatchNorm1d(128), nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.ConvTranspose1d(128, 64, kernel_size=3, stride=3), nn.BatchNorm1d(64), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=3, stride=3), nn.BatchNorm1d(32), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.ConvTranspose1d(32, 1, kernel_size=3, stride=3, output_padding=1))

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

# Initialize autoencoder
model = Conv1dAutoEncoder(seq_length=seq_length, latent_space_size=latent_space_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Train autoencoder
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

# Extract latent features
def extract_features(autoencoder, dataloader):
    features = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            latent = autoencoder.encode(inputs)
            features.append(latent.cpu().numpy())
    return np.vstack(features)

latent_features = extract_features(model, dataloader)

# Calculate spectral features
def calculate_spectral_features(spectrum):
    peaks, _ = find_peaks(spectrum)
    num_peaks = len(peaks)
    peak_values = spectrum[peaks]
    if len(peak_values) > 1:
        dominance = peak_values.max() / sorted(peak_values)[-2]
    else:
        dominance = 1.0
    half_max = peak_values.max() / 2
    closest_peak_idx = np.abs(spectrum - half_max).argmin()
    fwhm = np.abs(closest_peak_idx - peaks[0])
    return np.array([num_peaks, dominance, fwhm])

spectral_features = np.array([calculate_spectral_features(test.squeeze()) for test in data.cpu().numpy()])

# Combine latent and spectral features
combined_features = np.hstack((latent_features, spectral_features))

# Define regressor model
class QDRegressor(nn.Module):
    def __init__(self, input_size):
        super(QDRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = combined_features.shape[1]
regressor_model = QDRegressor(input_size).to(device)

# Loss for confidence-based regression
def confidence_loss(y_pred, y_true):
    s_pred, sigma = y_pred[:, 0], torch.exp(y_pred[:, 1])
    loss = torch.mean(((y_true - s_pred) ** 2) / (2 * sigma ** 2) + torch.log(sigma))
    print('loss', loss)
    return loss

regressor_optimizer = optim.Adam(regressor_model.parameters(), lr=0.001)

# Training regressor
def train_regressor(model, features, labels, epochs=500, batch_size=64):
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            regressor_optimizer.zero_grad()
            outputs = model(inputs)
            loss = confidence_loss(outputs, targets)
            loss.backward()
            regressor_optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(dataloader)
        losses.append(average_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.6f}")
    return losses

# Dummy labels for demonstration
labels = np.random.uniform(0, 1, size=(combined_features.shape[0], 1))

# Train regressor
train_losses = train_regressor(regressor_model, combined_features, labels)

# Test regressor
def test_regressor(model, test_features):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(test_features, dtype=torch.float32).to(device))
        predicted_scores, confidences = predictions[:, 0], predictions[:, 1]
        return predicted_scores.cpu().numpy(), confidences.cpu().numpy()

test_features = combined_features[:5]
predicted_scores, predicted_confidences = test_regressor(regressor_model, test_features)

# Print results
for i in range(len(predicted_scores)):
    print(f"Test {i+1}: Predicted Score = {predicted_scores[i]:.2f}, Confidence = {np.exp(predicted_confidences[i]):.2f}")