import d4rl
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import random
import time
from tqdm import trange

# -------------------- Models --------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, s):
        h = self.net(s)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-20, 2)
        return mu, log_std

    def get_deterministic_action(self, s):
        mu, _ = self.forward(s)
        return mu

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.model(x)

# -------------------- Utilities --------------------

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------- Train dynamics model on D4RL dataset --------------------
def train_dynamics_on_d4rl(env_name='hopper-medium-v2',
                            save_path='./dynamics.pth',
                            epochs=50, batch_size=256, lr=1e-3, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    obs = dataset['observations']
    acts = dataset['actions']
    next_obs = dataset['next_observations']

    state_dim = obs.shape[1]
    action_dim = acts.shape[1]

    # Convert to torch tensors and optionally normalize
    obs_mean = obs.mean(0)
    obs_std = obs.std(0) + 1e-6

    def norm(x):
        return (x - obs_mean) / obs_std

    # train/test split
    N = obs.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(N * 0.9)
    train_idx = idx[:split]
    val_idx = idx[split:]

    model = DynamicsModel(state_dim, action_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_obs = torch.tensor(obs[train_idx], dtype=torch.float32).to(device)
    train_acts = torch.tensor(acts[train_idx], dtype=torch.float32).to(device)
    train_next = torch.tensor(next_obs[train_idx], dtype=torch.float32).to(device)

    val_obs = torch.tensor(obs[val_idx], dtype=torch.float32).to(device)
    val_acts = torch.tensor(acts[val_idx], dtype=torch.float32).to(device)
    val_next = torch.tensor(next_obs[val_idx], dtype=torch.float32).to(device)

    n_batches = int(np.ceil(train_obs.shape[0] / batch_size))

    for ep in range(epochs):
        perm = torch.randperm(train_obs.shape[0])
        model.train()
        epoch_loss = 0.0
        for i in range(n_batches):
            batch_idx = perm[i*batch_size:(i+1)*batch_size]
            s = train_obs[batch_idx]
            a = train_acts[batch_idx]
            s_next = train_next[batch_idx]

            # predict delta s
            pred = model(s, a)
            loss = loss_fn(pred, s_next)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * s.shape[0]

        # val
        model.eval()
        with torch.no_grad():
            val_pred = model(val_obs, val_acts)
            val_loss = loss_fn(val_pred, val_next).item()

        print(f"[Dynamics] Ep {ep+1}/{epochs} train_loss={epoch_loss/train_obs.shape[0]:.6f} val_loss={val_loss:.6f}")

    torch.save({'state_dict': model.state_dict(), 'obs_mean': obs_mean, 'obs_std': obs_std}, save_path)
    print(f"Dynamics saved to {save_path}")
    return save_path

# -------------------- Load dynamics model --------------------
def load_dynamics(model, path, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    obs_mean = ckpt.get('obs_mean')
    obs_std = ckpt.get('obs_std')
    if obs_mean is not None:
        obs_mean = torch.tensor(obs_mean, dtype=torch.float32, device=device)
        obs_std = torch.tensor(obs_std, dtype=torch.float32, device=device)
    return model, obs_mean, obs_std

# -------------------- Compute real grad (Critic TD) --------------------
def compute_real_grad_batch(critic, target_critic, actor, target_actor,
                            s_batch, a_batch, r_batch, s_next_batch,
                            gamma=0.99, device=None):
    """
    s_batch: np.array or torch.Tensor, shape (B, state_dim)
    a_batch: shape (B, action_dim)
    r_batch: shape (B,) or (B,1)
    s_next_batch: shape (B, state_dim)
    return : real_grad (1D tensor) = concatenated gradients of Critic parameters
            for loss = sum_i MSE(Q(s_i,a_i), y_i)
    """
    device = device or torch.device("cpu")
    # transition tensor
    s_t = torch.tensor(s_batch, dtype=torch.float32, device=device)
    a_t = torch.tensor(a_batch, dtype=torch.float32, device=device)
    r_t = torch.tensor(r_batch, dtype=torch.float32, device=device).reshape(-1, 1)
    s_next_t = torch.tensor(s_next_batch, dtype=torch.float32, device=device)

    #  Q(s,a) -> shape (B, 1) or (B,)
    q_sa = critic(s_t, a_t).reshape(-1, 1)

    with torch.no_grad():
        if target_actor is not None:
            a_next = target_actor.get_deterministic_action(s_next_t)
        else:
            a_next = actor.get_deterministic_action(s_next_t)
        q_next = target_critic(s_next_t, a_next).reshape(-1, 1)
        y = r_t + gamma * q_next

    # use sum reduction to ensure scalarity (sum over the entire batch)
    loss = F.mse_loss(q_sa, y, reduction='sum')

    
    grads = torch.autograd.grad(loss, critic.parameters(), allow_unused=False)
    real_grad = torch.cat([g.reshape(-1) for g in grads]).detach()
    return real_grad


# ----------  RGIA attack ----------
def rgia_attack_ac_batch(critic, target_critic, actor, target_actor, real_grad,
                         dynamics_model, mu_s_batch, action_low, action_high,
                         n_iters=1000, lr=1e-2, gamma=0.99, lambda_reg=1.0,
                         alpha=1.0, beta=1.0, gamma_dyn=1.0,
                         r_min=-100.0, r_max=100.0, device=None):

    device = device or torch.device("cpu")
    critic.to(device).eval()
    dynamics_model.to(device).eval()
    actor.to(device).eval()
    if target_actor is not None:
        target_actor.to(device).eval()
    if target_critic is not None:
        target_critic.to(device).eval()

    B = mu_s_batch.shape[0]
    state_dim = mu_s_batch.shape[1]
    action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
    action_high = torch.tensor(action_high, dtype=torch.float32, device=device)
    action_mean = (action_high + action_low) / 2.0
    action_scale = (action_high - action_low) / 2.0

    # Initialize learnable batch variables (initialization centered on mu_s_batch is more stable)
    mu_s_t = mu_s_batch.to(device).float()
    tilde_s = (mu_s_t + 0.01 * torch.randn_like(mu_s_t)).detach().clone().requires_grad_(True)       # (B, state_dim)
    tilde_s_next = (mu_s_t + 0.01 * torch.randn_like(mu_s_t)).detach().clone().requires_grad_(True)  # (B, state_dim)
    # raw action (B, action_dim)
    action_dim = action_low.shape[0]
    tilde_a_raw = (0.01 * torch.randn((B, action_dim), device=device)).detach().clone().requires_grad_(True)
    tilde_r = (0.01 * torch.randn((B, 1), device=device)).detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([tilde_s, tilde_s_next, tilde_a_raw, tilde_r], lr=lr)
    real_grad = real_grad.to(device).float()



    for it in range(n_iters):
        optimizer.zero_grad()

        # map raw -> tanh -> scaled action, shape (B, action_dim)
        tilde_a = action_mean + action_scale * torch.tanh(tilde_a_raw)

        # compute fake TD loss over batch (use target networks same as compute_real_grad_batch)
        q_sa = critic(tilde_s, tilde_a).reshape(-1, 1)   # (B,1)

        with torch.no_grad():
            if target_actor is not None:
                a_next_pred = target_actor.get_deterministic_action(tilde_s_next)
            else:
                a_next_pred = actor.get_deterministic_action(tilde_s_next)
            q_next = target_critic(tilde_s_next, a_next_pred).reshape(-1, 1)
        tilde_y = tilde_r + gamma * q_next  # (B,1)

        # fake_loss
        fake_loss = F.mse_loss(q_sa, tilde_y, reduction='sum')

        # compute pseudo gradients of critic parameters
        grads = torch.autograd.grad(fake_loss, critic.parameters(), create_graph=True)
        fake_grad = torch.cat([g.reshape(-1) for g in grads])  # 1D vector

        # gradient matching 
        grad_loss = F.mse_loss(fake_grad, real_grad)

        # Regularization term
        R_s = F.mse_loss(tilde_s, mu_s_t)         # average over batch
        R_r = F.relu(tilde_r - r_max).pow(2).mean() + F.relu(r_min - tilde_r).pow(2).mean()
        pred_next = dynamics_model(tilde_s, tilde_a)         # (B, state_dim)
        R_dyn = F.mse_loss(pred_next, tilde_s_next)
        R_a = torch.mean(torch.tanh(tilde_a_raw).pow(2))

        reg_loss = alpha * R_s + beta * R_r + gamma_dyn * R_dyn + 0.1 * R_a

        total_loss = grad_loss + lambda_reg * reg_loss

        
        total_loss.backward()

        
        for p in critic.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        optimizer.step()

        if (it + 1) % max(1, n_iters//10) == 0:
            print(f"[BatchRGIA] it {it+1}/{n_iters} grad_loss={grad_loss.item():.6e} reg={reg_loss.item():.6e}")

   
    
    return tilde_s.detach().cpu().numpy(), tilde_a.detach().cpu().numpy(), tilde_r.detach().cpu().numpy().reshape(-1), tilde_s_next.detach().cpu().numpy()


def main(args):
    set_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    env_name = args.env_name  # requires d4rl
    dynamics_path = args.dynamics_path

    # 1) Train dynamics model on d4rl dataset (only if not saved)
    if not os.path.exists(dynamics_path):
        print('Training dynamics model on D4RL dataset...')
        train_dynamics_on_d4rl(env_name=env_name, save_path=dynamics_path, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                               device=device)
    else:
        print('Found existing dynamics model, skipping training.')

    # 2) Prepare networks and load dynamics
    env = gym.make(env_name)
    dataset = env.get_dataset()
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]
    action_low = env.action_space.low
    action_high = env.action_space.high

    actor = Actor(state_dim, action_dim).to(device)
    target_actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    dyn = DynamicsModel(state_dim, action_dim)
    dyn, obs_mean, obs_std = load_dynamics(dyn, dynamics_path, device=device)

    # choose a real transition from dataset (we are not sampling from env here)
    B = args.reconstruct_batch
    idxs = np.random.choice(len(dataset['observations']), size=B, replace=False)
    s_batch = dataset['observations'][idxs]  # (B, state_dim)
    a_batch = dataset['actions'][idxs]  # (B, action_dim)
    r_batch = dataset['rewards'][idxs]  # (B,)
    s_next_batch = dataset['next_observations'][idxs]

    # calculate the true gradient
    real_grad = compute_real_grad_batch(critic, target_critic, actor, target_actor,
                                        s_batch, a_batch, r_batch, s_next_batch,
                                        gamma=0.99, device=device)


    mu_s_batch = torch.tensor(s_batch, dtype=torch.float32, device=device)

    tilde_s, tilde_a, tilde_r, tilde_s_next = rgia_attack_ac_batch(
        critic=critic, target_critic=target_critic, actor=actor, target_actor=target_actor,
        real_grad=real_grad, dynamics_model=dyn,
        mu_s_batch=mu_s_batch, action_low=env.action_space.low, action_high=env.action_space.high,
        n_iters=args.n_iters, lr=args.reconstruct_lr, device=device
    )

    print('MSE:', np.mean((s_batch - tilde_s) ** 2))

# -------------------- Main demo flow --------------------
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--env-name', type=str, default='walker2d-medium-v2')
    parser.add_argument('--device',type=str, default='cuda:0')
    parser.add_argument('--dynamics_path',type=str, default='models/dynamics_walker2d.pth')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--reconstruct_batch',type=int, default=1)
    parser.add_argument('--n_iters',type=int,default=20000)
    parser.add_argument('--reconstruct_lr',type=float,default=2e-3)
    main(parser.parse_args())




