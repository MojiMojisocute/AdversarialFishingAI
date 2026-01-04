import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import os
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size]) if hasattr(data, '__array__') 
                     else torch.tensor(data[i:i+block_size], dtype=torch.long) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1]) if hasattr(data, '__array__') 
                     else torch.tensor(data[i+1:i+block_size+1], dtype=torch.long) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.inference_mode()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    out = {}
    model.eval()
    
    for split_name, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            with autocast('cuda', dtype=torch.bfloat16): 
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean()
    
    model.train()
    return out


def train(model, train_data, val_data, config, device):
    batch_size = config.get('batch_size', 32)
    block_size = config.get('block_size', 256)
    max_iters = config.get('max_iters', 5000)
    eval_interval = config.get('eval_interval', 500)
    learning_rate = config.get('learning_rate', 3e-4)
    eval_iters = config.get('eval_iters', 200)
    weight_decay = config.get('weight_decay', 0.1)
    grad_clip = config.get('grad_clip', 1.0)
    warmup_iters = config.get('warmup_iters', 100)
    lr_decay_iters = config.get('lr_decay_iters', max_iters)
    min_lr = config.get('min_lr', learning_rate / 10)
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        fused=True if device == 'cuda' else False 
    )
    
    scaler = GradScaler('cuda')
    
    use_compile = False
    if use_compile and hasattr(torch, 'compile') and device == 'cuda':
        try:
            model = torch.compile(model, mode='default')
            print("✓ Model compiled with torch.compile!")
        except Exception as e:
            print(f"⚠ Could not compile model: {e}")
            print("  Continuing without compilation...")
    else:
        print("⚠ Skipping torch.compile to save VRAM")
    
    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * decay_ratio)))
        return min_lr + coeff * (learning_rate - min_lr)
    
    model.train()
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    print(f"✓ Using mixed precision: bfloat16")
    print(f"✓ Fused optimizer: {optimizer.defaults.get('fused', False)}")
    print(f"✓ Gradient accumulation: Disabled (single step per iteration)")
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    for iter in tqdm(range(max_iters), desc="Training"):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, block_size, 
                                 batch_size, eval_iters, device)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            
            print(f"\n✓ Step {iter}: train={losses['train']:.4f}, val={losses['val']:.4f}, lr={lr:.6f}")

            if iter > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': iter,
                    'config': config,
                }
                checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_{iter}.pt')
                torch.save(checkpoint, checkpoint_path)
                
            if device == 'cuda':
                torch.cuda.empty_cache()

        X, Y = get_batch(train_data, block_size, batch_size, device)

        with autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)  
        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
    
    print("\n✓ Training complete!")
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
    }


def train_with_dataloader(model, train_dataset, val_dataset, config, device): 
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader