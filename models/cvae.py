import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from opacus import GradSampleModule, PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager


class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_dim, device):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x, y):
        concat_input = torch.cat([x, y], dim=-1)
        h = self.encoder(concat_input)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        concat_input = torch.cat([z, y], dim=-1)
        return self.decoder(concat_input)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar


class CVAETrainer:
    def __init__(self, model, args, device, chkpt_path, n_classes, num_epochs=100, lr=1e-3, beta=0.1): 
        self.model = model.to(device)
        self.device = device
        self.n_classes = n_classes
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) 
        self.beta = beta 
        self.chkpt_path = chkpt_path

        # Differential Privacy
        self.enable_dp = args.enable_dp
        self.max_grad_norm = args.max_grad_norm
        self.epsilon = args.epsilon
        self.delta = args.delta 

        # FedAvg training
        self.lr = lr
        self.lr_scheduler = None
        self.num_epochs = num_epochs 
        self.is_ready = True

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean') 
        # KLD loss: D_KL(Q(z|X) || P(z))
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
        return recon_loss + self.beta * kld_loss 

    def train_epoch(self, train_loader): 
        self.model.train()
        total_loss = 0.0

        if self.enable_dp:
            with BatchMemoryManager(
                data_loader=train_loader, 
                max_physical_batch_size=256, 
                optimizer=self.optimizer
            ) as memory_safe_data_loader:
                for x, y in memory_safe_data_loader: 
                    x, y = x.to(self.device), y.to(self.device)
                    y_oh = F.one_hot(y, num_classes=self.n_classes).float()

                    self.optimizer.zero_grad()
                    recon_x, mu, logvar = self.model(x, y_oh)
                    loss = self.loss_function(recon_x, x, mu, logvar)
                    loss.backward()
                    self.optimizer.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()

                    total_loss += loss.item()

        else:
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_oh = F.one_hot(y, num_classes=self.n_classes).float()

                self.optimizer.zero_grad()
                recon_x, mu, logvar = self.model(x, y_oh)
                loss = self.loss_function(recon_x, x, mu, logvar)
                loss.backward()
                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                total_loss += loss.item()

        return total_loss / len(train_loader) 

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_oh = F.one_hot(y, num_classes=self.n_classes).float()

                recon_x, mu, logvar = self.model(x, y_oh)
                loss = self.loss_function(recon_x, x, mu, logvar)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save_checkpoint(self, checkpoint):
        torch.save(checkpoint, self.chkpt_path)
        print(f"Checkpoint saved successfully at {self.chkpt_path}.")

    def load_checkpoint(self):
        if os.path.exists(self.chkpt_path):
            checkpoint = torch.load(self.chkpt_path, map_location=self.device, weights_only=True)

            if isinstance(self.model, GradSampleModule):
                self.model._module.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print(f"Checkpoint loaded successfully from {self.chkpt_path}.")

    def fit(self, train_loader, val_loader, global_model=None, num_epochs=None, lr_scheduler=None):
        if self.enable_dp and not isinstance(self.model, GradSampleModule):
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=train_loader,
                epochs=20,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )
            print(f"[DP] Using noise_multiplier={self.optimizer.noise_multiplier}")

        if num_epochs is not None:
            self.num_epochs = num_epochs

        if lr_scheduler is not None:
            from utils.optim import get_lr_scheduler
            self.lr_scheduler = get_lr_scheduler(
                optimizer=self.optimizer,
                scheduler_name=lr_scheduler,
                n_rounds=self.num_epochs
            )

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_loader) 
            val_loss = self.evaluate(val_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] | Train_Loss: {train_loss:.3f} | Val_Loss: {val_loss:.3f}")
            if self.enable_dp:
                print((f"[DP] Using epsilon = {self.epsilon:.3f}, delta = {self.delta:.6f} (noise_multiplier = {self.optimizer.noise_multiplier:.3f})"))
        
        #Save final epoch checkpoint
        self.save_checkpoint(
            {
                'num_epochs': self.num_epochs,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_state_dict': self.model._module.state_dict() if isinstance(self.model, GradSampleModule) else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
        )
