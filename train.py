import torch
import torch.nn as nn
import torch.nn.functional as F
from LandVit import SoilViT, GeoDataset, CombinedLoss, SpectralConsistencyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

class WeightedL1Loss(nn.Module):
    def __init__(self, device):
        super(WeightedL1Loss, self).__init__()
        std_values = [0.09907163707875082, 0.021866454094841583,0.0822928954774687, 0.04990219943453045, 0.039165479266202265, 0.062032485604882254]
        self.weights = 1.0 / (torch.tensor(std_values) + 1e-8)

        self.weights = self.weights / self.weights.sum()
        self.weights = self.weights.to(device)
    def forward(self, y_pred, y_true):
        l1 = torch.abs(y_pred - y_true)

        weighted_l1 = l1 * self.weights.view(1, -1, 1, 1)
        return weighted_l1.mean()


# Structured training loop
class Trainer:
    def __init__(self, model, train_loader, val_loader, device, pretained_weights=None):
        self.model = model.to(device)
        if pretained_weights:
            self.model.load_state_dict(pretained_weights, strict=False)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = WeightedL1Loss(self.device).to(self.device) # nn.L1Loss()
        # self.criterion = CombinedLoss(
        #     mse_weight=0.8,
        #     ssim_weight=0.1,
        #     r2_weight=0.1,
        #     channel_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 根据通道重要性设置权重
        # )
        self.spectral_loss = SpectralConsistencyLoss()
        self.val_loss = nn.L1Loss()
        self.best_eval_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y, lat_lon in tqdm(self.train_loader):
            x, y, lat_lon = x.to(self.device), y.to(self.device), lat_lon.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x, lat_lon)
            loss_recon = self.criterion(pred, y)
            # loss_spec = self.spectral_loss(pred)
            loss = loss_recon # + 0.01 * loss_spec  # weight spectral consistency
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y, lat_lon in self.val_loader:
                x, y, lat_lon = x.to(self.device), y.to(self.device), lat_lon.to(self.device)
                pred = self.model(x, lat_lon)
                loss_recon = self.val_loss(pred, y)
                # loss_spec = self.spectral_loss(pred)
                loss = loss_recon # + 0.01 * loss_spec
                total_loss += loss.item()
        if total_loss < self.best_eval_loss:
            self.best_eval_loss = total_loss
            torch.save(self.model.state_dict(), 'model.pth')
        return total_loss / len(self.val_loader)


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoilViT(in_chans=13, out_chans=6, embed_dim=768, depth=12, num_heads=12)  
    best_model = 'best_model_00402.pth'  # Path to save the best model weights
    pretained_weights = None #torch.load(best_model)
    train_loader = DataLoader(GeoDataset('train_patch_data.json'), batch_size=16, shuffle=True)
    val_loader = DataLoader(GeoDataset('eval_patch_data.json'), batch_size=1, shuffle=False)  # Define your validation DataLoader

    trainer = Trainer(model, train_loader, val_loader, device, pretained_weights)
    for epoch in range(50):  # Replace with your desired number of epochs
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")