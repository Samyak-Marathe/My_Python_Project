import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import graphics
from torch.utils.data import TensorDataset, DataLoader
import csv
import os

# Set Seed
seed = 1234
np.random.seed(seed)


# =================== Model Definition ===================
class FFNN(nn.Module):
    """Feedforward Neural Network."""

    def __init__(self):
        super(FFNN, self).__init__()
        torch.set_default_dtype(torch.float64)

        self.loss_function = nn.BCEWithLogitsLoss()
        self.activation = nn.LeakyReLU(negative_slope=.01)

        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x): # Very necessary. Forward function is called internally by the model. We don't call it explicitly unless required.
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        return x

    def loss(self, predicted, true):
        return self.loss_function(predicted, true)

    def optimize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# =================== Utilities ===================
def initialize(s, past=False, mode="train"):
    """Initialize the model and load past training data if required."""
    global model, optimizer, epochs, batch, learning_rate, scheduler

    torch.set_default_dtype(torch.float64)

    if not past:
        torch.manual_seed(6161)
        np.random.seed(6161)
        model = FFNN()
    else:
        checkpoint = torch.load(f'./training_data/models/model_{shapes[shape_id]}_{num_points}.pth')
        model = FFNN()
        torch.manual_seed(checkpoint['seed'])
        np.random.seed(checkpoint['seed'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if mode == "retrain":
            learning_rate = float(
                input('To use a new learning rate enter it, or enter 0 to use the previous rate: ')) or \
                            checkpoint['learning_rate']
            epochs = int(input('To use new epoch size enter it, or enter 0 to use the previous epoch size: ')) or \
                     checkpoint['epoch']
            batch = int(input('To use new batch size enter it, or enter 0 to use the previous batch size: ')) or \
                    checkpoint['batch_idx']

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        s = shapes[checkpoint['shape_id']]
        print(s)

    shape_classes = {
        'CUBE': graphics.Cube,
        'TETRAHEDRON': graphics.Tetrahedron,
        'OCTAHEDRON': graphics.Octahedron,
        'DODECAHEDRON': graphics.Dodecahedron,
        'ICOSAHEDRON': graphics.Icosahedron
    }

    return shape_classes.get(s.upper(), lambda: None)()


def fetch_data(num_points, shape, test=False):
    """Fetch training or testing data from CSV."""
    path = 'testing_data' if test else 'training_data'
    file_path = f'{path}/{shape.status}_data_{num_points}.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        points, points_inside_shape_label = [], []
        next(reader)  # Skip header

        for i in reader:
            try:
                points.append([float(i[0]), float(i[1]), float(i[2])])
                points_inside_shape_label.append(int(i[3]))
            except:
                pass
    return np.array(points).reshape(-1, 3), np.array(points_inside_shape_label).reshape(-1, 1)


def train_model(x, y, epochs, batch):
    """Train the model."""
    global optimizer, scheduler

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    dataset_xy = TensorDataset(x, y)
    data_loader = DataLoader(dataset_xy, batch_size=batch, shuffle=True)

    for epoch in range(epochs):
        for data in data_loader:
            input_point, true_label = data
            optimizer.zero_grad()
            predicted_label = model(input_point)
            loss = model.loss(predicted_label, true_label)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')


def estimate_pi(model, shape, num_points):
    """Estimate pi using trained model and testing data."""
    print('\n'.join([f'{i}: {shapes[i]}' for i in range(len(shapes))]))
    s, p = (int(input('Enter shape index of the trained/training model: ')),
            int(input('Number of points of the trained/training model: ')))
    points, _ = fetch_data(p, shape, test=True)
    points = torch.tensor(points)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        predicted_labels = model(points.to(device))

    predicted_labels = (predicted_labels.cpu().numpy() >= 0.5).astype(int)
    print(predicted_labels, np.sum(predicted_labels))
    df = pd.DataFrame(predicted_labels)
    df.to_csv("data.csv", index=False)

    pi = p * 3 * graphics.Cube().volume / (
            4 * np.sum(predicted_labels) * (graphics.Cube().radius_outer_circle ** 3))
    print(f"Estimated π: {pi}")


# =================== Main Execution ===================
shapes = ['Tetrahedron', 'Cube', 'Octahedron', 'Dodecahedron', 'Icosahedron']
trained_before = int(input(
    'Enter 1 to train from scratch, 0 to continue training, -1 to estimate π with trained model: '))

print('\n'.join([f'{i}: {shapes[i]}' for i in range(len(shapes))]))
shape_id = int(input('Enter shape index of the trained/training model: '))
num_points = int(input('Number of points of the trained/training model: '))

if trained_before == 1:
    model = FFNN()
    model.optimize()
    epochs = int(input('Epoch: '))
    batch = int(input('Batch: '))
    learning_rate = 0.005
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    shape = initialize(shapes[shape_id], past=False)

elif trained_before == 0:
    shape = initialize(shapes[shape_id], past=True, mode="retrain")

elif trained_before == -1:
    shape = initialize(shapes[shape_id], past=True, mode="inference")
    estimate_pi(model, shape, num_points)
    exit()

# Training path continues here
points, points_labels = fetch_data(num_points, shape)

ratio = np.sum(points_labels) / (points_labels.shape[0] - np.sum(points_labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight = torch.tensor([ratio]).to(device)
model.loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

random_indices = np.arange(num_points)
np.random.shuffle(random_indices)
training_indices = random_indices[:int(num_points * .7)]
training_points, training_labels = points[training_indices], points_labels[training_indices]

# Train
start_time = time.time()
train_model(training_points, training_labels, epochs, batch)
elapsed = time.time() - start_time

# Final evaluation on testing_data
points, _ = fetch_data(num_points, shape, test=True)
points = torch.tensor(np.array(points))
model.to(device)
model.eval()
with torch.no_grad():
    predicted_labels = model(points.to(device))

predicted_labels = (predicted_labels.cpu().numpy() >= 0.5).astype(int)
df = pd.DataFrame(predicted_labels)
df.to_csv("data.csv", index=False)

pi = num_points * 3 * graphics.Cube().volume / (
        4 * np.sum(predicted_labels) * (graphics.Cube().radius_outer_circle ** 3))
print(f"Estimated π: {pi}")

# Save model checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epochs,
    'batch_idx': batch,
    'learning_rate': learning_rate,
    'time': elapsed,
    'seed': seed,
    'shape_id': shape_id,
    'num_points': num_points
}

os.makedirs('training_data/models', exist_ok=True)
torch.save(checkpoint, f'./training_data/models/model_{shape.status.capitalize()}_{num_points}.pth')
