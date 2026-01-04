import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np

def train_autoencoder(model, dataloader, epochs=50, lr=0.001, device='cuda', save_dir='data'):
    """Train autoencoder model"""
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    
    train_losses = []
    recon_losses = []
    anomaly_losses = []
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Batches per epoch: {len(dataloader)}")
    print("="*60 + "\n")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_anomaly_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in pbar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                reconstructed, latent, anomaly_score = model(batch)
                
                batch_size, seq_len, vocab_size = reconstructed.shape
                reconstructed_flat = reconstructed.view(-1, vocab_size)
                target_flat = batch.view(-1)
                
                loss_recon = criterion(reconstructed_flat, target_flat)
                
                loss_anomaly = torch.var(anomaly_score)
                
                loss = loss_recon + 0.1 * loss_anomaly
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            epoch_recon_loss += loss_recon.item()
            epoch_anomaly_loss += loss_anomaly.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{loss_recon.item():.4f}'
            })
                
        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon_loss / len(dataloader)
        avg_anomaly = epoch_anomaly_loss / len(dataloader)
        
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        anomaly_losses.append(avg_anomaly)
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, Anomaly: {avg_anomaly:.4f}')
        
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, save_dir)
    
    save_checkpoint(model, optimizer, epochs-1, train_losses[-1], save_dir, final=True)
    
    plot_training_curves(train_losses, recon_losses, anomaly_losses, save_dir)
    
    return train_losses, recon_losses, anomaly_losses


def save_checkpoint(model, optimizer, epoch, loss, save_dir, final=False):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    filename = 'model_final.pth' if final else f'model_epoch_{epoch+1}.pth'
    filepath = os.path.join(save_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    
    print(f'[Checkpoint] Saved to {filepath}')


def load_checkpoint(model, filepath, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f'[Checkpoint] Loaded from {filepath}')
    print(f'[Checkpoint] Epoch: {checkpoint["epoch"]}, Loss: {checkpoint["loss"]:.4f}')
    
    return model


def plot_training_curves(train_losses, recon_losses, anomaly_losses, save_dir='data'):
    """Plot and save training curves"""
    
    analysis_dir = os.path.join(save_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Analysis', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(epochs, recon_losses, 'g-', linewidth=2, label='Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].plot(epochs, anomaly_losses, 'r-', linewidth=2, label='Anomaly Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Anomaly Detection Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].plot(epochs, train_losses, 'b-', linewidth=2, label='Total', alpha=0.7)
    axes[1, 1].plot(epochs, recon_losses, 'g-', linewidth=2, label='Reconstruction', alpha=0.7)
    axes[1, 1].plot(epochs, anomaly_losses, 'r-', linewidth=2, label='Anomaly', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Combined Losses')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    filepath = os.path.join(analysis_dir, 'training_curves.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'[Graph] Saved training curves to {filepath}')
    
    plt.close()
    
    loss_file = os.path.join(analysis_dir, 'losses.txt')
    with open(loss_file, 'w') as f:
        f.write('Epoch,Total_Loss,Recon_Loss,Anomaly_Loss\n')
        for i, (tl, rl, al) in enumerate(zip(train_losses, recon_losses, anomaly_losses)):
            f.write(f'{i+1},{tl:.6f},{rl:.6f},{al:.6f}\n')
    
    print(f'[Graph] Saved loss data to {loss_file}')


def plot_reconstruction_examples(model, dataloader, num_examples=5, save_dir='data'):
    """Plot reconstruction examples"""
    
    analysis_dir = os.path.join(save_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        batch = next(iter(dataloader)).to(device)
        reconstructed, latent, anomaly_scores = model(batch)
        
        errors = model.get_reconstruction_error(batch[:num_examples])
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3*num_examples))
    if num_examples == 1:
        axes = [axes]
    
    for i in range(num_examples):
        original = batch[i].cpu().numpy()
        recon = torch.argmax(reconstructed[i], dim=-1).cpu().numpy()
        error = errors[i].item()
        anomaly = anomaly_scores[i].item()
        
        axes[i].plot(original, 'b-', label='Original', alpha=0.6)
        axes[i].plot(recon, 'r--', label='Reconstructed', alpha=0.6)
        axes[i].set_title(f'Example {i+1} - Error: {error:.4f}, Anomaly: {anomaly:.4f}')
        axes[i].set_xlabel('Position')
        axes[i].set_ylabel('Token ID')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = os.path.join(analysis_dir, 'reconstruction_examples.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'[Graph] Saved reconstruction examples to {filepath}')
    
    plt.close()


if __name__ == "__main__":
    from dataloader import create_dataloader
    from model import create_model
    
    print("="*60)
    print("Training Script Test")
    print("="*60)
    
    dataloader, loader, email_data = create_dataloader(
        data_folder='data/email',
        batch_size=8
    )
    
    if dataloader:
        model = create_model(vocab_size=loader.vocab_size)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        train_losses, recon_losses, anomaly_losses = train_autoencoder(
            model=model,
            dataloader=dataloader,
            epochs=20,
            lr=0.001,
            device=device,
            save_dir='data'
        )
        
        plot_reconstruction_examples(model, dataloader, num_examples=3, save_dir='data')
        
        print("\n[Training] Complete!")