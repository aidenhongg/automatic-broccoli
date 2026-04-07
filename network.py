import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

USE_LOGIC_GATE_NETWORK = True


class DiffLogicLayer(nn.Module):
    """Sequential-fold differentiable logic over 4 input vectors.

    Chains gates to respect temporal ordering:
        h1 = gate_0(v0, v1)
        h2 = gate_1(h1, v2)
        out = gate_2(h2, v3)

    Each step accumulates context from all prior timesteps,
    so the n -> n+1 dependency chain is preserved.
    """

    def __init__(self, num_bits: int = 16):
        super().__init__()
        # window_size - 1 sequential gate steps, each with per-bit weights over {AND, OR, XOR}
        self.gate_weights = nn.Parameter(torch.randn(3, num_bits, 3))

    @staticmethod
    def _apply_gates(a, b, gate_weights):
        """Apply soft AND/OR/XOR between a and b using learned weights."""
        probs = F.softmax(gate_weights, dim=-1)        # (num_bits, 3)

        out_and = a * b
        out_or  = a + b - a * b
        out_xor = a + b - 2 * a * b

        stacked = torch.stack([out_and, out_or, out_xor], dim=-1)  # (batch, num_bits, 3)
        return torch.sum(probs * stacked, dim=-1)                  # (batch, num_bits)

    def forward(self, x):
        """
        Args:
            x: (batch, 4 * num_bits) -- 4 flattened bit-vectors in temporal order.
        Returns:
            (batch, num_bits)
        """
        num_bits = self.gate_weights.shape[1]
        v0 = x[:, 0 * num_bits : 1 * num_bits]
        v1 = x[:, 1 * num_bits : 2 * num_bits]
        v2 = x[:, 2 * num_bits : 3 * num_bits]
        v3 = x[:, 3 * num_bits : 4 * num_bits]

        # Sequential fold -- each step sees accumulated state + next timestep
        h = self._apply_gates(v0, v1, self.gate_weights[0])
        h = self._apply_gates(h,  v2, self.gate_weights[1])
        h = self._apply_gates(h,  v3, self.gate_weights[2])
        return h


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            self.linear.weight.uniform_(-1 / in_features, 1 / in_features)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class PCGDataset(Dataset):
    """Dataset for PCG sequence prediction using sliding windows."""
    
    def __init__(self, csv_path: str, window_size: int = 4):
        self.window_size = window_size
        self.data = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if USE_LOGIC_GATE_NETWORK:
                    self.data.append([0 if int(x) < 0 else 1 for x in row])
                else:
                    self.data.append([int(x) for x in row])

        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size].flatten()  # (window_size * 16,)
        y = self.data[idx + self.window_size]                  # (16,)
        return x, y


class DiffLogicNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(64, 256),
            DiffLogicLayer(64),
            nn.Linear(64, 256),
            DiffLogicLayer(64),
            nn.Linear(64, 256),
            DiffLogicLayer(64),
            nn.Linear(64, 16),
        )

    def forward(self, x):
        return self.layers(x)


def get_dataloaders(csv_path: str, batch_size: int = 64, train_ratio: float = 0.8):
    """Create train/test DataLoaders with an 80/20 random split."""
    dataset = PCGDataset(csv_path, window_size=4)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch and return the average loss."""
    model.train()
    total_loss = 0.0

    for x, target in train_loader:
        x, target = x.to(device), target.to(device)
        target = torch.where(target < 0, torch.tensor(0.0, device=device), target)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(train_loader.dataset)


def test_epoch(model, test_loader, criterion, device):
    """Evaluate on the test set; returns (loss, bit accuracy, decimal MAE)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    decimal_error_sum = 0

    with torch.no_grad():
        for x, target in test_loader:
            x, target = x.to(device), target.to(device)
            target = torch.where(target < 0, torch.tensor(0.0, device=device), target)

            output = model(x)
            loss = criterion(output, target)
            total_loss += loss.item() * x.size(0)

            # Bit-level accuracy
            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == target).sum().item()
            total += target.numel()

            # Decimal-level mean absolute error (MSB-first encoding)
            powers = 2 ** torch.arange(15, -1, -1, device=device, dtype=torch.float32)
            target_decimal = (target * powers).sum(dim=-1)
            preds_decimal = (preds * powers).sum(dim=-1)
            decimal_error_sum += torch.abs(preds_decimal - target_decimal).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / total
    decimal_mae = decimal_error_sum / len(test_loader.dataset)
    return avg_loss, accuracy, decimal_mae


if __name__ == "__main__":
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders("./data.csv", batch_size=batch_size)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = DiffLogicNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    with open("training_log.txt", "w") as log_file:
        for epoch in range(1, num_epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, decimal_mae = test_epoch(model, test_loader, criterion, device)
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                f"Dec MAE: {decimal_mae:.4f} | LR: {current_lr:.6f}"
            )
            log_file.write(
                f"{epoch},{train_loss:.4f},{test_loss:.4f},{test_acc:.4f},"
                f"{decimal_mae:.4f},{current_lr:.6f}\n"
            )