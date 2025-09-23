import torch
import torch.nn as nn
from opacus import GradSampleModule, PrivacyEngine

from utils.constants import EMBEDDING_DIM, N_CLASSES


class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim, embedding_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),  
        )

    def forward(self, latent, labels):
        inputs = torch.cat((latent, labels), dim=1)
        return self.model(inputs)


class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + label_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings, labels):
        inputs = torch.cat((embeddings, labels), dim=1)
        return self.model(inputs)


class CGANTrainer:
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, args, device, chkpt_path, latent_dim=100, num_epochs=100, lr=1e-3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.label_dim = N_CLASSES[args.experiment] 
        self.embedding_dim = EMBEDDING_DIM[args.backbone] 
        self.chkpt_path = chkpt_path

        self.generator = generator 
        self.discriminator = discriminator 
        self.g_optimizer = g_optimizer 
        self.d_optimizer = d_optimizer 
        self.criterion = nn.BCELoss()

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

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            # Xavier initialization for weights
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def train_epoch(self, train_loader):
        g_loss_epoch, d_loss_epoch = 0, 0

        for real_emb, real_lbl in train_loader:
            real_emb, real_lbl = real_emb.to(self.device), real_lbl.to(self.device)

            if len(real_emb) != len(real_lbl):
                continue
            
            # Train Discriminator
            self.discriminator.train()

            self.d_optimizer.zero_grad(set_to_none=True)
            real_targets = torch.ones((real_emb.size(0), 1), device=self.device) 
            fake_targets = torch.zeros((real_emb.size(0), 1), device=self.device)

            latent = torch.randn(real_emb.size(0), self.latent_dim, device=self.device)
            fake_emb = self.generator(latent, real_lbl)
            
            real_loss = self.criterion(self.discriminator(real_emb, real_lbl), real_targets)
            fake_loss = self.criterion(self.discriminator(fake_emb.detach(), real_lbl), fake_targets)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            self.d_optimizer.zero_grad(set_to_none=True)
            
            # Train Generator
            self.generator.train()
            gen_targets = torch.ones((real_emb.size(0), 1), device=self.device)
            
            self.g_optimizer.zero_grad()
            g_loss = self.criterion(self.discriminator(fake_emb, real_lbl), gen_targets)
            g_loss.backward()
            self.g_optimizer.step()
            
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
        
        g_loss_epoch /= len(train_loader)
        d_loss_epoch /= len(train_loader)

        return g_loss_epoch, d_loss_epoch
    
    def evaluate(self, val_loader):
        self.generator.eval()
        self.discriminator.eval()

        g_loss_total = 0.0
        correct_real, correct_fake = 0, 0
        total_samples = 0

        with torch.no_grad():
            for real_emb, real_lbl in val_loader:
                real_emb, real_lbl = real_emb.to(self.device), real_lbl.to(self.device)
                
                if len(real_emb) != len(real_lbl):
                    continue

                batch_size = real_emb.size(0)
                total_samples += batch_size

                # Generate synthetic embeddings
                latent = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_emb = self.generator(latent, real_lbl)

                # Evaluate Generator loss
                gen_targets = torch.ones((batch_size, 1), device=self.device)
                g_loss = self.criterion(self.discriminator(fake_emb, real_lbl), gen_targets)
                g_loss_total += g_loss.item()

                # Evaluate Discriminator accuracy
                real_preds = self.discriminator(real_emb, real_lbl)
                fake_preds = self.discriminator(fake_emb, real_lbl)
                correct_real += (real_preds >= 0.5).sum().item()
                correct_fake += (fake_preds < 0.5).sum().item()

        # Compute averages
        avg_g_loss = g_loss_total / len(val_loader)
        d_accuracy = (correct_real + correct_fake) / (2 * total_samples)  # Both real and fake samples are considered

        return avg_g_loss, d_accuracy

    def save_checkpoint(self, num_epochs, g_loss, d_loss):
        torch.save(
            {
                'num_epochs': num_epochs,
                'g_loss': g_loss,
                'd_loss': d_loss,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator._module.state_dict() if isinstance(self.discriminator, GradSampleModule) else self.discriminator.state_dict(),
                'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            }, 
            self.chkpt_path
        )
        print(f"Checkpoint saved successfully at {self.chkpt_path}.")
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.chkpt_path, map_location=self.device, weights_only=True)
        if isinstance(self.discriminator, GradSampleModule):
            self.discriminator._module.load_state_dict(checkpoint['discriminator_state_dict'])
        else:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

        print(f"Checkpoint loaded successfully from {self.chkpt_path}.")
    
    def fit(self, train_loader, val_loader, global_model=None, num_epochs=None, lr_scheduler=None):
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)

        if self.enable_dp and not isinstance(self.discriminator, GradSampleModule):
            privacy_engine = PrivacyEngine()
            self.discriminator, self.d_optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=self.discriminator,
                optimizer=self.d_optimizer,
                data_loader=train_loader,
                epochs=20,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )
            print(f"[DP] Discriminator using noise_multiplier={self.d_optimizer.noise_multiplier}")
        
        if num_epochs is not None:
            self.num_epochs = num_epochs

        if lr_scheduler is not None:
            from utils.optim import get_lr_scheduler
            self.lr_scheduler = get_lr_scheduler(
                optimizer=self.optimizer,
                scheduler_name=lr_scheduler,
                n_rounds=self.num_epochs
            )

        for epoch in range(num_epochs):
            train_g_loss, train_d_loss = self.train_epoch(train_loader)
            val_g_loss, val_d_accuracy = self.evaluate(val_loader)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train G_Loss: {train_g_loss:.3f} | Train D_Loss: {train_d_loss:.3f} | Val G_Loss: {val_g_loss:.3f} | Val D_Accuracy: {val_d_accuracy * 100:.2f}")
            if self.enable_dp:
                print((f"[DP] Using epsilon = {self.epsilon:.3f}, delta = {self.delta:.6f} (noise_multiplier = {self.d_optimizer.noise_multiplier:.3f})"))
            
        #Save final epoch checkpoint
        self.save_checkpoint(num_epochs, train_g_loss, train_d_loss)

    

