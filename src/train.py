import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import argparse
import gc
import numpy as np
import json
from datetime import datetime
import torch.nn.functional as F

from dataset import ChromosomeDataset, get_transform
from models import get_model

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, optimizer):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, optimizer)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': val_loss,
        }, 'best_model.pth')
        self.val_loss_min = val_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        # Convert target to long type for one_hot encoding
        target = target.long()
        target_onehot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        
    def forward(self, pred, target):
        if isinstance(pred, tuple):
            # Handle deep supervision
            main_pred, *aux_preds = pred
            # Resize target to match prediction size if needed
            if main_pred.shape[2:] != target.shape[1:]:
                target = F.interpolate(target.unsqueeze(1).float(), size=main_pred.shape[2:], mode='nearest').squeeze(1).long()
            main_loss = self.alpha * self.ce(main_pred, target) + (1 - self.alpha) * self.dice(main_pred, target)
            aux_losses = []
            for p in aux_preds:
                if p.shape[2:] != target.shape[1:]:
                    resized_target = F.interpolate(target.unsqueeze(1).float(), size=p.shape[2:], mode='nearest').squeeze(1).long()
                    aux_losses.append(self.alpha * self.ce(p, resized_target) + (1 - self.alpha) * self.dice(p, resized_target))
                else:
                    aux_losses.append(self.alpha * self.ce(p, target) + (1 - self.alpha) * self.dice(p, target))
            return main_loss + 0.3 * sum(aux_losses)
        else:
            # Resize target to match prediction size if needed
            if pred.shape[2:] != target.shape[1:]:
                target = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[2:], mode='nearest').squeeze(1).long()
            return self.alpha * self.ce(pred, target) + (1 - self.alpha) * self.dice(pred, target)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, experiment_dir, patience=7):
    os.makedirs(experiment_dir, exist_ok=True)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')
        
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
            
            # Clear memory
            del images, masks, outputs, loss
            gc.collect()
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['learning_rates'].append(current_lr)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                del images, masks, outputs, loss
                gc.collect()
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(experiment_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        
        # Save training history
        with open(os.path.join(experiment_dir, 'history.json'), 'w') as f:
            json.dump(history, f)
        
        early_stopping(avg_val_loss, model, epoch, optimizer)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='/Users/nuno/programação/tese/cromossomas/train_labelme')
    parser.add_argument('--val_dir', type=str, default='/Users/nuno/programação/tese/cromossomas/test_labelme')
    parser.add_argument('--model_type', type=str, default='enhanced_unet', 
                      choices=['enhanced_unet', 'deeplabv3', 'segformer'],
                      help='Type of model to use')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join('experiments', f'{args.model_type}_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment configuration
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    device = torch.device('mps')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = ChromosomeDataset(
        args.train_dir,
        transform=get_transform(mode='train', img_size=args.img_size),
        img_size=args.img_size
    )
    val_dataset = ChromosomeDataset(
        args.val_dir,
        transform=get_transform(mode='val', img_size=args.img_size),
        img_size=args.img_size
    )
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model, criterion, optimizer and scheduler
    model = get_model(args.model_type, in_channels=3, out_channels=24).to(device)
    print(f"\nModel Parameters:")
    print(json.dumps(model.get_parameters(), indent=4))
    
    criterion = CombinedLoss(alpha=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart interval after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Train the model
    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        args.num_epochs,
        device,
        experiment_dir,
        patience=args.patience
    )
    
    print(f"\nTraining completed. Results saved in {experiment_dir}")

if __name__ == '__main__':
    main()
