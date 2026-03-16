import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import time
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float)
torch.manual_seed(2316)
np.random.seed(2316)


def training_data():
    n_samples_init = np.random.randint(1, modes + 1, size=(X.shape[1], 1))
    n_samples_bound = np.random.randint(1, modes + 1, size=(X.shape[0], 1))

    initial = np.hstack((X[0, :][:, None], T[0, :][:, None], n_samples_init))
    u_initial = torch.from_numpy(
        A * np.sin(initial[:, 2][:, None] * np.pi * (initial[:, 0][:, None] - lbx) / (ubx - lbx)))
    initial = torch.from_numpy(initial)

    left_boundary = torch.from_numpy(np.hstack((X[:, 0][:, None], T[:, 0][:, None], n_samples_bound)))
    u_left = torch.from_numpy(np.zeros(shape=(left_boundary.shape[0], 1)))

    # Fix: Ensure dimension 0 is exactly X.shape[0] (501)
    right_boundary = torch.from_numpy(np.hstack((X[:, -1][:, None], T[:, 0][:, None], n_samples_bound)))
    u_right = torch.from_numpy(np.zeros(shape=(right_boundary.shape[0], 1)))

    x_flat, t_flat = X.flatten()[:, None], T.flatten()[:, None]
    n_flat = np.random.randint(1, modes + 1, size=(x_flat.shape[0], 1))
    x_coll = np.hstack((x_flat, t_flat, n_flat))
    idx = np.random.choice(x_coll.shape[0], N_coll, replace=False)
    x_coll = torch.from_numpy(x_coll[idx])

    training_input = [initial.float(), left_boundary.float(), right_boundary.float(), x_coll.float()]
    training_output = [u_initial.float(), u_left.float(), u_right.float()]

    return training_input, training_output


class PINN(nn.Module):
    def __init__(self, rff_info):
        super().__init__()
        self.rff = rff_info
        if self.rff:
            self.fc1 = nn.Linear(2 * self.rff + 1, 164)
            self.B = nn.Parameter(torch.cat([torch.normal(0., 0.5, size=(1, self.rff)),
                                             torch.normal(0., 1.0, size=(1, self.rff))], dim=0), requires_grad=False)
        else:
            self.fc1 = nn.Linear(3, 164)
        self.fc2 = nn.Linear(164, 164)
        self.fc3 = nn.Linear(164, 164)
        self.fc4 = nn.Linear(164, 1)
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')

    def random_ff(self, raw_x, raw_t, raw_n):
        freq = torch.cat([raw_x * raw_n * modes, raw_t * raw_n * modes], dim=0) @ self.B
        rff_inp = torch.cat([torch.sin(2 * torch.pi * freq), torch.cos(2 * torch.pi * freq), raw_n], dim=1)
        return rff_inp

    def forward(self, xtn):
        x_norm = (xtn[:, [0]] - lbx) / (ubx - lbx)
        t_norm = (xtn[:, [1]] - lbt) / (ubt - lbt)
        n_norm = xtn[:, [2]] / modes
        if self.rff:
            xtn_normalized = self.random_ff(x_norm, t_norm, n_norm)
        else:
            xtn_normalized = torch.cat([x_norm, t_norm, n_norm], dim=1)
        a = self.activation(self.fc1(xtn_normalized))
        a = self.activation(self.fc2(a))
        a = self.activation(self.fc3(a))
        return A * self.fc4(a)

    def loss(self, init, bound_l, bound_r, col, uboundl, uboundr, uinit):
        cp = col.clone()
        cp.requires_grad = True
        ic = init.clone()
        ic.requires_grad = True

        u_physics = self.forward(cp)
        u_bound_l = self.forward(bound_l)
        u_bound_r = self.forward(bound_r)
        u_init = self.forward(ic)

        u_xt = autograd.grad(u_physics.sum(), cp, retain_graph=True, create_graph=True)[0]
        u_x, u_t = u_xt[:, [0]], u_xt[:, [1]]
        u_xx = autograd.grad(u_x.sum(), cp, retain_graph=True, create_graph=True)[0][:, [0]]
        u_tt = autograd.grad(u_t.sum(), cp, retain_graph=True, create_graph=True)[0][:, [1]]

        u_t_ic = autograd.grad(u_init.sum(), ic, retain_graph=True, create_graph=True)[0][:, [1]]

        f = v * v * u_xx - u_tt
        f_hat = torch.zeros_like(f)
        ut_hat = torch.zeros_like(u_t_ic)

        loss_initial = self.loss_function(u_init, uinit) + self.loss_function(u_t_ic, ut_hat)
        loss_boundary = self.loss_function(u_bound_l, uboundl) + self.loss_function(u_bound_r, uboundr)
        loss_physics = self.loss_function(f, f_hat)

        loss_weights = [1, 50, 10]
        return loss_weights[0] * loss_physics + loss_weights[1] * loss_initial + loss_weights[2] * loss_boundary


# Parameters restored to original scale
ubx, lbx = 10.0, 0.
A, v, modes = 1., 1., 5
ubt, lbt = 2 * ubx / v, 0.
x = np.linspace(lbx, ubx, 501)
t = np.linspace(lbt, ubt, 501)
X, T = np.meshgrid(x, t)

N_coll = 16000
frequencies = int(input('Enter Frequency for RFF. Enter 0 to start without RFF: '))
input_p, output_p = training_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pinn = PINN(frequencies)
pinn.to(device)

input_p = [i.to(device) for i in input_p]
output_p = [i.to(device) for i in output_p]
start_time = time.time()
max_iter = 20000
optimizer = optim.Adam(pinn.parameters(), lr=1e-3)

if frequencies:
    print('Initializing training with RFF.')
else:
    print('Initializing training without RFF.')
for i in range(max_iter):
    loss = pinn.loss(input_p[0], input_p[1], input_p[2], input_p[3], output_p[1], output_p[2], output_p[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 200 == 0: print(f"Iter {i}: {loss.item()}")

print("Starting L-BFGS fine-tuning...")
lbfgs_optimizer = optim.LBFGS(pinn.parameters(), max_iter=5000, tolerance_grad=1e-5, tolerance_change=1e-9,
                              history_size=50)


def closure():
    lbfgs_optimizer.zero_grad()
    loss = pinn.loss(input_p[0], input_p[1], input_p[2], input_p[3], output_p[1], output_p[2], output_p[0])
    loss.backward()
    return loss


lbfgs_optimizer.step(closure)
print("L-BFGS Complete.")

elapsed = time.time() - start_time

print(f'Time taken to train: {elapsed}')

# Visualization at t = 13 for mode n = 1
plt.figure(figsize=(10, 5))
plt.grid(True)
target_t = ubt / 2
test_mode = 3
# Input vector: [x, 13, 1]
xtn_test = np.vstack((x, np.full(x.shape[0], target_t), np.full(x.shape[0], test_mode))).T

test_tensor = torch.from_numpy(xtn_test).float().to(device)
predictions = pinn.forward(test_tensor).cpu().detach().numpy().flatten()
true_sol = A * np.sin(test_mode * np.pi * x / (ubx - lbx)) * np.cos(test_mode * np.pi * v * target_t / (ubx - lbx))

plt.plot(x, predictions, label=f'Predicted Solution')
plt.plot(x, true_sol, color='orange', label=f'True Solution')

l2_error = np.linalg.norm(predictions - true_sol) / np.linalg.norm(true_sol)
print(f"Relative L2 Error: {l2_error * 100:.2f}%")

plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title(f'Solution for String Problem (n={test_mode} at t={target_t})')
plt.legend()
plt.show()

if int(input('Do you want to save the model? 1 for Yes, 0 for No.')):
    if frequencies:
        torch.save(pinn.state_dict(), 'string_solver_with_rff.pth')
    else:
        torch.save(pinn.state_dict(), 'string_solver_without_rff.pth')
    print("Model saved successfully")
