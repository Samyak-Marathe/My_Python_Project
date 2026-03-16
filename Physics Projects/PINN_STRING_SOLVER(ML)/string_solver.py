import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np


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
        freq = torch.cat([raw_x * raw_n * modes, raw_t * raw_n * modes], dim=1) @ self.B
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
        return self.fc4(a)

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


def load_model(winx, winy, freq=256):
    global pinn, device, L, A, wx, wy
    wx = winx
    wy = winy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn = PINN(freq)
    pinn.to(device)
    if freq:
        pinn.load_state_dict(torch.load('string_solver_with_rff.pth', weights_only=True))
    else:
        pinn.load_state_dict(torch.load('string_solver_without_rff.pth', weights_only=True))
    pinn.eval()


def predict(px, pt, pn):
    global v, A, x, t, n, ubx, lbx, ubt, lbt, modes
    ubx, lbx, ubt, lbt = 10., 0., 20., 0
    x, t = px.reshape(len(px), 1), pt.reshape(len(pt), 1)
    v = 1
    n = pn.reshape(len(pn), 1)
    modes = 5
    input_tensor = torch.tensor(np.hstack((x, t, n)), dtype=torch.float32).to(device)
    with torch.no_grad():
        sol = pinn(input_tensor).cpu().numpy()

    return np.round(sol, 2), np.round(np.sin(n * np.pi * x / ubx) * np.cos(n * np.pi * v * t / ubx), 2)

def transform(x, y, s=0):
    if not s:
        return np.hstack((((wx[1] - wx[0]) * x.reshape(-1, 1) / 10) + wx[0], (y * (wy[0] - wy[1]) / 2) + (wy[0] + wy[1]) / 2))
    else:
        return np.hstack((((wx[1] - wx[0]) * x.reshape(-1, 1) / 10) + wx[0] + s, (y * (wy[0] - wy[1]) / 2) + (wy[0] + wy[1]) / 2))
