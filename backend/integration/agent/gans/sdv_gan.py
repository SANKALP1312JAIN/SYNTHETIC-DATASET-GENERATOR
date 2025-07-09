import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from .base_gan import BaseGAN

class SDVGAN(BaseGAN):
    """SDV-based GAN for general tabular data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        config = config or {}
        self.latent_dim = config.get('latent_dim', 100)
        self.hidden_dim = config.get('hidden_dim', 256)
        
    def _build_networks(self, input_dim: int):
        """Build generator and discriminator networks"""
        
        class Generator(nn.Module):
            def __init__(self, latent_dim, hidden_dim, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, output_dim),
                    nn.Tanh()
                )
            
            def forward(self, x):
                return self.net(x)
        
        class Discriminator(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.net(x)
        
        self.generator = Generator(self.latent_dim, self.hidden_dim, input_dim).to(self.device)
        self.discriminator = Discriminator(input_dim, self.hidden_dim).to(self.device)
        
    def generate(self, data: pd.DataFrame, num_samples: int = None) -> pd.DataFrame:
        if num_samples is None:
            num_samples = len(data)
        
        # Preprocess data
        processed_data, metadata = self.preprocess_data(data)
        metadata['original_columns'] = data.columns.tolist()
        
        # Build networks
        self._build_networks(processed_data.shape[1])
        
        # Train GAN (simplified training)
        self._train_gan(processed_data)
        
        # Generate synthetic data
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            synthetic_data = self.generator(noise).cpu()
        
        return self.postprocess_data(synthetic_data, metadata)
    
    def _train_gan(self, real_data: torch.Tensor, epochs: int = 100):
        """Simplified GAN training"""
        g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        
        real_data = real_data.to(self.device)
        
        for epoch in range(epochs):
            # Train discriminator
            d_optimizer.zero_grad()
            real_labels = torch.ones(real_data.size(0), 1).to(self.device)
            fake_labels = torch.zeros(real_data.size(0), 1).to(self.device)
            
            real_outputs = self.discriminator(real_data)
            d_real_loss = criterion(real_outputs, real_labels)
            
            noise = torch.randn(real_data.size(0), self.latent_dim).to(self.device)
            fake_data = self.generator(noise)
            fake_outputs = self.discriminator(fake_data.detach())
            d_fake_loss = criterion(fake_outputs, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Train generator
            g_optimizer.zero_grad()
            fake_outputs = self.discriminator(fake_data)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            g_optimizer.step() 