import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
#import wandb  


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class TrajectoryDataset(Dataset):
    def __init__(self, init_hs, mask, final_reward, vf_scores):
        """
        init_hs:      (N, 3072)
        mask:         (N, 2048)
        final_reward: (N, 5)
        vf_scores:    (N, 2048, 5)
        """
        super().__init__()
        self.init_hs = init_hs
        self.mask = mask
        self.final_reward = final_reward
        self.vf_scores = vf_scores

    def __len__(self):
        return self.init_hs.shape[0]

    def __getitem__(self, idx):
        return (
            self.init_hs[idx],      # (3072,)
            self.mask[idx],         # (2048,)
            self.final_reward[idx], # (5,)
            self.vf_scores[idx]     # (2048, 5)
        )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (B, T, d_model)
        return x with positional encoding
        """
        T = x.size(1)
        return x + self.pe[:T, :]

class TrajectoryTransformer(nn.Module):
    """
    fusing init_hs (3072-d), final_reward (5-d) and step (5-d)
    """
    def __init__(
        self,
        hidden_state_dim=3072,
        reward_dim=5,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=4096
    ):
        super().__init__()
        self.d_model = d_model

        self.init_proj = nn.Linear(hidden_state_dim, 128)

        total_input_dim = 5 + reward_dim + 128
        self.input_proj = nn.Linear(total_input_dim, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, reward_dim)

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, init_hs, final_reward, trajectory=None, pad_mask=None, max_steps=50):
        """
        if we do have ground truth trajectory, we use teacher forcing to train
        if we dont have ground truth trajectory, we use autoregressive inference
        """
        device = init_hs.device
        init_hs_proj = self.init_proj(init_hs)  # (B, 128)

        if trajectory is not None:
            return self._forward_teacher_forcing(init_hs_proj, final_reward, trajectory, pad_mask)
        else:
            return self._forward_autoregressive(init_hs_proj, final_reward, max_steps)

    def _forward_teacher_forcing(self, init_hs_proj, final_reward, trajectory, pad_mask):
        """
        trajectory: (B, T, 5)
        """
        B, T, _ = trajectory.shape
        device = trajectory.device

        # 将 ground-truth 序列向右平移1步
        input_traj = torch.roll(trajectory, shifts=1, dims=1)
        input_traj[:, 0, :] = 0.0

        # 扩展 final_reward 到 (B, T, 5)
        final_reward_exp = final_reward.unsqueeze(1).expand(B, T, 5)
        # 扩展 init_hs_proj 到 (B, T, 128)
        init_hs_exp = init_hs_proj.unsqueeze(1).expand(B, T, 128)

        # 合并输入 => (B, T, 138)
        combined_in = torch.cat([input_traj, final_reward_exp, init_hs_exp], dim=-1)

        # 经过投影和位置编码
        x = self.input_proj(combined_in)  # (B, T, d_model)
        x = self.pos_encoding(x)

        # 生成因果掩码
        causal_mask = self._generate_causal_mask(T, device=device)

        # Transformer 编码
        out = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        pred = self.output_proj(out)  # (B, T, 5)
        return pred

    @torch.no_grad()
    def _forward_autoregressive(self, init_hs_proj, final_reward, max_steps):
        B = init_hs_proj.shape[0]
        device = init_hs_proj.device

        generated_steps = []
        prev_step = torch.zeros(B, 5, device=device)  # 初始步为零向量

        for t in range(max_steps):
            if len(generated_steps) > 0:
                partial_traj = torch.stack(generated_steps, dim=1)  # (B, t, 5)
            else:
                partial_traj = torch.empty(B, 0, 5, device=device)

            partial_traj = torch.cat([partial_traj, prev_step.unsqueeze(1)], dim=1)
            steps_so_far = partial_traj.size(1)

            final_reward_exp = final_reward.unsqueeze(1).expand(B, steps_so_far, 5)
            init_hs_exp = init_hs_proj.unsqueeze(1).expand(B, steps_so_far, 128)

            combined_in = torch.cat([partial_traj, final_reward_exp, init_hs_exp], dim=-1)
            x = self.input_proj(combined_in)
            x = self.pos_encoding(x)

            causal_mask = self._generate_causal_mask(steps_so_far, device=device)
            out = self.transformer_encoder(x, mask=causal_mask)
            next_step_pred = self.output_proj(out[:, -1, :])

            generated_steps.append(next_step_pred)
            prev_step = next_step_pred

        return torch.stack(generated_steps, dim=1)

def train_one_epoch(model, train_loader, optimizer, device="cuda", alpha=5.0):
    import torch.nn.functional as F

    model.to(device)
    model.train()

    total_loss = 0.0
    total_raw_loss = 0.0
    for init_hs, mask, final_reward, vf_scores in train_loader:
        init_hs      = init_hs.to(device)      # (B, 3072)
        mask         = mask.to(device)         # (B, T)
        final_reward = final_reward.to(device)   # (B, 5)
        vf_scores    = vf_scores.to(device)      # (B, T, 5)

        # Teacher-forcing forward
        pred = model(
            init_hs=init_hs,
            final_reward=final_reward,
            trajectory=vf_scores,
            pad_mask=(mask == 0)
        )  # (B, T, 5)

        B, T, _ = pred.shape
        weights = torch.ones_like(pred) 

        for b in range(B):
            valid_steps = int(mask[b].sum().item())
            if valid_steps > 0:
                weights[b, 0, :] = 1.0 
                for t in range(1, valid_steps):
                    diff = torch.norm(vf_scores[b, t, :] - vf_scores[b, t-1, :], p=2).item()
                    weights[b, t, :] = 1.0 + alpha * diff

        valid_mask = mask.unsqueeze(-1).expand(-1, -1, 5).bool()
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        
        squared_errors = (pred - vf_scores) ** 2
        weighted_loss = (squared_errors * weights.to(device)).sum()
        
        raw_loss = F.mse_loss(pred[valid_mask], vf_scores[valid_mask], reduction='sum')

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        total_loss += weighted_loss.item()
        total_raw_loss += raw_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_raw_loss = total_raw_loss / (len(train_loader) * 5)
    return avg_loss

def validate_one_epoch(model, test_loader, device="cuda"):
    import torch.nn.functional as F

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_valid_elems = 0

    with torch.no_grad():
        for init_hs, mask, final_reward, vf_scores in test_loader:
            init_hs      = init_hs.to(device)
            mask         = mask.to(device)
            final_reward = final_reward.to(device)
            vf_scores    = vf_scores.to(device)

            pred = model(
                init_hs=init_hs,
                final_reward=final_reward,
                trajectory=vf_scores,
                pad_mask=(mask == 0)
            )

            valid_mask = mask.unsqueeze(-1).expand(-1, -1, 5).bool()
            loss = torch.nn.functional.mse_loss(pred[valid_mask], vf_scores[valid_mask], reduction='sum')

            total_loss += loss.item()
            total_valid_elems += (mask.sum().item() * 5)

    avg_mse = total_loss / total_valid_elems
    return avg_mse

def load_train_data(device):
    init_h = torch.load("init_hs_train.pth", map_location=device).float()
    reward = torch.load("response_train_scores.pth", map_location=device).float()
    mask = torch.load("mask_train.pth", map_location=device).float()
    trajectory = torch.load("vf_scores_train.pth", map_location=device).float()
    return init_h, mask, reward, trajectory


def load_test_data(device):
    init_h = torch.load("init_hs_test.pth", map_location=device).float()
    reward = torch.load("response_test_scores.pth", map_location=device).float()
    mask = torch.load("mask_test.pth", map_location=device).float()
    trajectory = torch.load("vf_scores_test.pth", map_location=device).float()
    return init_h, mask, reward, trajectory


def test_and_plot_single_sample(device, model, max_steps=2048, sample_idx=None):
    model.eval()
    
    init_h, mask, reward, trajectory = load_test_data(device)
    N = init_h.shape[0]
    if sample_idx is None:
        sample_idx = random.randrange(N)

    init_h_sample = init_h[sample_idx].unsqueeze(0).to(device)
    reward_sample = reward[sample_idx].unsqueeze(0).to(device)
    real_traj     = trajectory[sample_idx].unsqueeze(0).to(device)
    mask_sample   = mask[sample_idx].to(device)
    T = real_traj.shape[1]

    valid_length = int((mask_sample == 1).sum().item())

    with torch.no_grad():
        pred_traj = model(
            init_hs=init_h_sample,
            final_reward=reward_sample,
            trajectory=None,
            max_steps=max_steps
        )

    real_traj_np = real_traj.squeeze(0).cpu().numpy()
    pred_traj_np = pred_traj.squeeze(0).cpu().numpy()
    compare_length = min(valid_length, max_steps)

    plt.figure(figsize=(12, 8))
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(real_traj_np[:compare_length, i], label=f'Real trajectory {i+1}')
        plt.plot(pred_traj_np[:compare_length, i], '--', label=f'Predicted trajectory {i+1}')
        plt.ylabel(f'Traj dim {i+1}')
        plt.legend()
        plt.grid(True)
        if i == 0:
            plt.title(f"Single Sample idx={sample_idx}, valid_length={valid_length}\nReward={reward_sample}")
    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.savefig(f"single_sample_{sample_idx}.png", dpi=200)
    plt.show()


def test_and_plot_multiple_samples(device, model, max_steps=2048, token_threshold=1000):

    init_h, mask, reward, trajectory = load_test_data(device)
    N = init_h.shape[0]
    indices_to_plot = []
    for idx in range(N):
        valid_length = int((mask[idx] == 1).sum().item())
        if valid_length > token_threshold:
            indices_to_plot.append(idx)
    if len(indices_to_plot) == 0:
        print(f"No samples with valid_length >{token_threshold}")
        return

    if len(indices_to_plot) > 10:
        indices_to_plot = random.sample(indices_to_plot, 10)
    for idx in indices_to_plot:
        test_and_plot_single_sample(device, model, max_steps=max_steps, sample_idx=idx)

if __name__ == "__main__":
    #wandb.init(project="rnn", name="rnn-train-no-linear")
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    model = TrajectoryTransformer(
        reward_dim=5,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 20

    init_hs_train, mask_train, response_train_scores, vf_scores_train = load_train_data(device)
    init_hs_test, mask_test, response_test_scores, vf_scores_test = load_test_data(device)
    print("Data load complete.")

    train_dataset = TrajectoryDataset(init_hs_train, mask_train, response_train_scores, vf_scores_train)
    test_dataset  = TrajectoryDataset(init_hs_test, mask_test, response_test_scores, vf_scores_test)
    print("Datasets built.")

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_loss_list = []
    test_loss_list = []
    
    best_test_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, alpha=5.0)
        test_loss = validate_one_epoch(model, test_loader, device)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model, "best_change_model.pth")
            print(f"Epoch {epoch+1}: New best test loss: {best_test_loss:.6f}")

        #wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")
        print(f"Test Loss: {test_loss:.6f}")
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png", dpi=200)
    plt.show()

    best_model = torch.load("best_change_model.pth")
    test_and_plot_multiple_samples(device, best_model, max_steps=2048, token_threshold=1000)
    # torch.save(model, "rnn_trajectory_predict_model_transformer.pth")