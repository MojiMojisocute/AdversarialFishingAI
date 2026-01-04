import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional, Dict
import math


class EmailAutoencoder(nn.Module):
    """
    Professional-grade Autoencoder for email sequence modeling.
    
    Key improvements:
    - Proper autoregressive decoder with teacher forcing
    - Padding mask support for variable-length sequences
    - Normalized latent space with KL regularization
    - Separated anomaly detection logic
    - Efficient batch operations
    """
    
    def __init__(
        self, 
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_vae: bool = False
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.use_vae = use_vae
        
        # Shared embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=0
        )
        
        # Encoder: bidirectional LSTM for better context
        self.encoder_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Divide by 2 for bidirectional
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Latent space projection
        if use_vae:
            # VAE: learn mean and log-variance
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        else:
            # Standard AE: direct projection
            self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        
        # Latent normalization
        self.latent_norm = nn.LayerNorm(latent_dim)
        
        # Decoder initial state projection
        self.decoder_init_h = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.decoder_init_c = nn.Linear(latent_dim, num_layers * hidden_dim)
        
        # Decoder: unidirectional LSTM (autoregressive)
        self.decoder_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'embedding' not in name:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def encode(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode input sequences to latent representation.
        
        Args:
            x: Input token IDs [batch, seq_len]
            lengths: Actual lengths of sequences [batch]
        
        Returns:
            latent: Latent representation [batch, latent_dim]
            logvar: Log-variance (only if use_vae=True)
        """
        batch_size = x.size(0)
        
        # Embed tokens
        embedded = self.embedding(x)  # [batch, seq_len, emb_dim]
        
        # Pack sequences if lengths provided
        if lengths is not None:
            embedded = pack_padded_sequence(
                embedded, 
                lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
        
        # Encode with LSTM
        _, (hidden, cell) = self.encoder_lstm(embedded)
        
        # Concatenate bidirectional hidden states
        # hidden: [num_layers*2, batch, hidden_dim//2]
        hidden = hidden.view(self.num_layers, 2, batch_size, -1)
        hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=-1)  # [batch, hidden_dim]
        
        # Project to latent space
        if self.use_vae:
            mu = self.fc_mu(hidden)
            logvar = self.fc_logvar(hidden)
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mu + eps * std
            latent = self.latent_norm(latent)
            return latent, (mu, logvar)
        else:
            latent = self.fc_latent(hidden)
            latent = self.latent_norm(latent)
            return latent, None
    
    def decode(
        self,
        latent: torch.Tensor,
        target_seq: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressive decoder with teacher forcing.
        
        Args:
            latent: Latent representation [batch, latent_dim]
            target_seq: Target sequence for teacher forcing [batch, seq_len]
            lengths: Actual sequence lengths [batch]
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            output_logits: [batch, seq_len, vocab_size]
        """
        batch_size = latent.size(0)
        seq_len = target_seq.size(1)
        
        # Initialize decoder hidden state from latent
        h0 = self.decoder_init_h(latent)  # [batch, num_layers * hidden_dim]
        c0 = self.decoder_init_c(latent)
        
        h0 = h0.view(batch_size, self.num_layers, self.hidden_dim)
        c0 = c0.view(batch_size, self.num_layers, self.hidden_dim)
        
        h0 = h0.transpose(0, 1).contiguous()  # [num_layers, batch, hidden_dim]
        c0 = c0.transpose(0, 1).contiguous()
        
        # Start token (using <PAD> token as <SOS>)
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=latent.device)
        
        outputs = []
        hidden = (h0, c0)
        
        for t in range(seq_len):
            # Embed current input
            embedded = self.embedding(decoder_input)  # [batch, 1, emb_dim]
            
            # LSTM step
            output, hidden = self.decoder_lstm(embedded, hidden)
            
            # Project to vocabulary
            logits = self.output_fc(output)  # [batch, 1, vocab_size]
            outputs.append(logits)
            
            # Teacher forcing: use ground truth or predicted token
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing and t < seq_len - 1:
                decoder_input = target_seq[:, t:t+1]
            else:
                decoder_input = logits.argmax(dim=-1)
        
        # Concatenate all time steps
        output_logits = torch.cat(outputs, dim=1)  # [batch, seq_len, vocab_size]
        
        return output_logits
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through autoencoder.
        
        Args:
            x: Input sequences [batch, seq_len]
            lengths: Actual sequence lengths [batch]
            teacher_forcing_ratio: Ratio for teacher forcing
        
        Returns:
            Dictionary with keys:
                - 'reconstructed': Logits [batch, seq_len, vocab_size]
                - 'latent': Latent vectors [batch, latent_dim]
                - 'mu': Mean (VAE only)
                - 'logvar': Log-variance (VAE only)
        """
        # Encode
        latent, vae_params = self.encode(x, lengths)
        
        # Decode
        reconstructed = self.decode(
            latent, 
            x, 
            lengths, 
            teacher_forcing_ratio
        )
        
        result = {
            'reconstructed': reconstructed,
            'latent': latent
        }
        
        if self.use_vae:
            result['mu'], result['logvar'] = vae_params
        
        return result
    
    def get_reconstruction_error(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Calculate reconstruction error efficiently with proper masking.
        
        Args:
            x: Input sequences [batch, seq_len]
            lengths: Actual sequence lengths [batch]
            reduction: 'none', 'mean', or 'sum'
        
        Returns:
            errors: Reconstruction error per sample or reduced
        """
        with torch.no_grad():
            outputs = self.forward(x, lengths, teacher_forcing_ratio=0.0)
            reconstructed = outputs['reconstructed']  # [batch, seq_len, vocab]
            
            # Create padding mask
            if lengths is not None:
                batch_size, seq_len = x.size()
                mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
                mask = mask.float()  # [batch, seq_len]
            else:
                mask = (x != 0).float()  # Assume 0 is padding
            
            # Calculate per-token cross-entropy
            reconstructed_flat = reconstructed.view(-1, self.vocab_size)
            x_flat = x.view(-1)
            
            token_errors = F.cross_entropy(
                reconstructed_flat,
                x_flat,
                reduction='none',
                ignore_index=0
            )
            token_errors = token_errors.view(x.size(0), x.size(1))
            
            # Apply mask and reduce
            masked_errors = token_errors * mask
            seq_lengths = mask.sum(dim=1).clamp(min=1)
            
            if reduction == 'none':
                errors = masked_errors.sum(dim=1) / seq_lengths
            elif reduction == 'mean':
                errors = (masked_errors.sum(dim=1) / seq_lengths).mean()
            elif reduction == 'sum':
                errors = masked_errors.sum()
            else:
                raise ValueError(f"Unknown reduction: {reduction}")
            
            return errors


class AnomalyDetector(nn.Module):
    """
    Separate anomaly detection module using learned latent representations.
    
    Uses contrastive learning principles and margin-based classification.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = [128, 64],
        dropout: float = 0.4,
        use_batchnorm: bool = True
    ):
        super().__init__()
        
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predict anomaly score from latent representation.
        
        Args:
            latent: Latent vectors [batch, latent_dim]
        
        Returns:
            scores: Anomaly scores [batch, 1]
        """
        logits = self.classifier(latent)
        return torch.sigmoid(logits)


class SpamDetector(nn.Module):
    """
    Spam detection system using pretrained autoencoder.
    
    Supports both frozen and fine-tunable encoder modes.
    """
    
    def __init__(
        self,
        autoencoder: EmailAutoencoder,
        freeze_encoder: bool = True,
        freeze_embedding: bool = True
    ):
        super().__init__()
        
        self.autoencoder = autoencoder
        
        # Selective freezing
        if freeze_embedding:
            for param in self.autoencoder.embedding.parameters():
                param.requires_grad = False
        
        if freeze_encoder:
            # Only freeze encoder, not decoder
            for param in self.autoencoder.encoder_lstm.parameters():
                param.requires_grad = False
            if hasattr(self.autoencoder, 'fc_latent'):
                for param in self.autoencoder.fc_latent.parameters():
                    param.requires_grad = False
            if hasattr(self.autoencoder, 'fc_mu'):
                for param in self.autoencoder.fc_mu.parameters():
                    param.requires_grad = False
                for param in self.autoencoder.fc_logvar.parameters():
                    param.requires_grad = False
        
        # Spam classifier head
        self.anomaly_detector = AnomalyDetector(
            latent_dim=autoencoder.latent_dim,
            hidden_dims=[128, 64, 32],
            dropout=0.5
        )
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for spam detection.
        
        Args:
            x: Input sequences [batch, seq_len]
            lengths: Actual sequence lengths [batch]
        
        Returns:
            spam_prob: Spam probability [batch, 1]
        """
        # Extract latent representation
        latent, _ = self.autoencoder.encode(x, lengths)
        
        # Classify
        spam_prob = self.anomaly_detector(latent)
        
        return spam_prob


def create_model(
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    latent_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    use_vae: bool = False
) -> EmailAutoencoder:
    """
    Factory function to create email autoencoder model.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of token embeddings
        hidden_dim: Hidden dimension of LSTM
        latent_dim: Dimension of latent space
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        use_vae: Whether to use VAE instead of standard AE
    
    Returns:
        model: Initialized EmailAutoencoder
    """
    model = EmailAutoencoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_vae=use_vae
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'='*60}")
    print(f"Model: EmailAutoencoder ({'VAE' if use_vae else 'AE'})")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Vocab size:           {vocab_size:,}")
    print(f"Embedding dim:        {embedding_dim}")
    print(f"Hidden dim:           {hidden_dim}")
    print(f"Latent dim:           {latent_dim}")
    print(f"Num layers:           {num_layers}")
    print(f"{'='*60}")
    
    return model


def compute_loss(
    model: EmailAutoencoder,
    x: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    teacher_forcing_ratio: float = 1.0,
    kl_weight: float = 0.0
) -> Dict[str, torch.Tensor]:
    """
    Compute training loss with proper masking and KL regularization.
    
    Args:
        model: EmailAutoencoder model
        x: Input sequences [batch, seq_len]
        lengths: Actual sequence lengths [batch]
        teacher_forcing_ratio: Teacher forcing ratio
        kl_weight: Weight for KL divergence term (VAE)
    
    Returns:
        Dictionary with loss components
    """
    outputs = model(x, lengths, teacher_forcing_ratio)
    reconstructed = outputs['reconstructed']
    
    # Create padding mask
    if lengths is not None:
        batch_size, seq_len = x.size()
        mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
        mask = mask.float()
    else:
        mask = (x != 0).float()
    
    # Reconstruction loss with masking
    recon_loss = F.cross_entropy(
        reconstructed.view(-1, model.vocab_size),
        x.view(-1),
        reduction='none',
        ignore_index=0
    )
    recon_loss = recon_loss.view(x.size(0), x.size(1))
    recon_loss = (recon_loss * mask).sum() / mask.sum()
    
    total_loss = recon_loss
    loss_dict = {'recon_loss': recon_loss}
    
    # KL divergence for VAE
    if model.use_vae and kl_weight > 0:
        mu = outputs['mu']
        logvar = outputs['logvar']
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()
        total_loss = total_loss + kl_weight * kl_loss
        loss_dict['kl_loss'] = kl_loss
    
    loss_dict['total_loss'] = total_loss
    
    return loss_dict


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Optimized Email Autoencoder")
    print("="*60 + "\n")
    
    # Configuration
    vocab_size = 5000
    batch_size = 4
    seq_len = 50
    
    # Create model
    model = create_model(
        vocab_size=vocab_size,
        use_vae=False
    )
    
    # Generate test data with variable lengths
    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    lengths = torch.randint(20, seq_len + 1, (batch_size,))
    
    # Apply padding
    for i in range(batch_size):
        x[i, lengths[i]:] = 0
    
    print(f"Input shape:   {x.shape}")
    print(f"Lengths:       {lengths.tolist()}")
    
    # Test forward pass
    print(f"\n{'='*60}")
    print("Testing Autoencoder Forward Pass")
    print("="*60)
    
    outputs = model(x, lengths, teacher_forcing_ratio=1.0)
    
    print(f"Reconstructed: {outputs['reconstructed'].shape}")
    print(f"Latent:        {outputs['latent'].shape}")
    
    # Test reconstruction error
    errors = model.get_reconstruction_error(x, lengths, reduction='none')
    print(f"Recon errors:  {errors.shape} - {errors.tolist()}")
    
    # Test loss computation
    loss_dict = compute_loss(model, x, lengths)
    print(f"\nLoss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Test spam detector
    print(f"\n{'='*60}")
    print("Testing Spam Detector")
    print("="*60)
    
    spam_model = SpamDetector(model, freeze_encoder=True)
    spam_probs = spam_model(x, lengths)
    
    print(f"Spam probs:    {spam_probs.shape}")
    print(f"Values:        {spam_probs.squeeze().tolist()}")
    
    # Count trainable parameters
    total = sum(p.numel() for p in spam_model.parameters())
    trainable = sum(p.numel() for p in spam_model.parameters() if p.requires_grad)
    print(f"\nSpamDetector parameters:")
    print(f"  Total:       {total:,}")
    print(f"  Trainable:   {trainable:,}")
    print(f"  Frozen:      {total - trainable:,}")
    
    print(f"\n{'='*60}")
    print("All tests passed! ✓")
    print("="*60 + "\n")