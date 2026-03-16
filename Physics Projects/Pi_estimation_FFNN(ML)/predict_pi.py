import torch
import torch.nn as nn
import numpy as np
import graphics


# ✅ Define the FFNN Model (Same as used during training)
class FFNN(nn.Module):
    """Feedforward Neural Network used for π estimation"""

    def __init__(self):
        super(FFNN, self).__init__()
        torch.set_default_dtype(torch.float64)

        self.activation = nn.LeakyReLU(negative_slope=.01)

        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        return x


# ✅ Function to Estimate π Using the Trained Model
def estimate_pi(model, s, num_samples=100000):
    """
    Estimate the value of π using Monte Carlo method and a trained model.

    Args:
        model (torch.nn.Module): Trained FFNN model.
        num_samples (int): Number of random points to generate.

    Returns:
        float: Estimated value of π.
    """

    # Generate random points in the cube [-1, 1] x [-1, 1] x [-1, 1]
    points = np.random.uniform(low=-1, high=1, size=(num_samples, 3))

    # Convert to PyTorch tensor and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points_tensor = torch.tensor(points, dtype=torch.float64).to(device)

    # Model inference (Predict whether points are inside the sphere)
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(points_tensor)

    # Convert logits to binary classification (0 or 1)
    inside_sphere = (predictions >= 0.5).cpu().numpy()

    # Count number of points inside the sphere
    inside_count = np.sum(inside_sphere)

    # Estimate π using the volume ratio
    pi = num_samples * 3 * s.volume / (4 * inside_count * (shape.radius_outer_circle ** 3))

    return pi


# ✅ Load the Trained Model (Without Training Again)
model_path = './training_data/models/model_Cube_100000.pth'  # Change this filename as needed

# Load the saved model checkpoint
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))  # Load on CPU if no GPU available

# ✅ Remove any extra keys in the saved checkpoint that don't match the model
if 'loss_function.pos_weight' in checkpoint['model_state_dict']:
    del checkpoint['model_state_dict']['loss_function.pos_weight']

# Initialize the model
model = FFNN()
model.load_state_dict(checkpoint['model_state_dict'])  # ✅ Load trained weights & biases
model.eval()  # Set to evaluation mode


cube = graphics.Cube()
tetrahedron = graphics.Tetrahedron()
octahedron = graphics.Octahedron()
icosahedron = graphics.Icosahedron()
dodecahedron = graphics.Dodecahedron()

shape = cube
# ✅ Estimate π Using the Pre-trained Model
estimated_pi = estimate_pi(model, shape, num_samples=100000)
print(f"Estimated π: {estimated_pi:.6f}")  # Print the estimated value of π
