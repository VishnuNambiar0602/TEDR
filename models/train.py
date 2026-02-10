"""Training pipeline for DETR model on custom datasets."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DetrForObjectDetection, DetrImageProcessor
from pathlib import Path
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import yaml


class COCODataset(Dataset):
    """COCO format dataset for DETR training."""
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        processor: DetrImageProcessor,
        transform=None
    ):
        """Initialize COCO dataset.
        
        Args:
            image_dir: Directory containing images
            annotation_file: Path to COCO format annotation JSON
            processor: DETR image processor
            transform: Optional image transforms
        """
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        
        self.images = self.coco['images']
        self.annotations = self._group_annotations()
        
    def _group_annotations(self) -> Dict:
        """Group annotations by image_id."""
        grouped = {}
        for ann in self.coco['annotations']:
            image_id = ann['image_id']
            if image_id not in grouped:
                grouped[image_id] = []
            grouped[image_id].append(ann)
        return grouped
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get image and annotations."""
        # Load image
        img_info = self.images[idx]
        img_path = self.image_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        image_id = img_info['id']
        anns = self.annotations.get(image_id, [])
        
        # Prepare target
        target = {
            'image_id': torch.tensor([image_id]),
            'boxes': [],
            'labels': []
        }
        
        for ann in anns:
            # Convert COCO bbox [x, y, width, height] to [x_min, y_min, x_max, y_max]
            x, y, w, h = ann['bbox']
            bbox = [x, y, x + w, y + h]
            target['boxes'].append(bbox)
            target['labels'].append(ann['category_id'])
        
        target['boxes'] = torch.tensor(target['boxes']) if target['boxes'] else torch.zeros((0, 4))
        target['labels'] = torch.tensor(target['labels']) if target['labels'] else torch.zeros(0, dtype=torch.long)
        
        # Process image
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        # Remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        
        return pixel_values, target


class DETRTrainer:
    """Trainer for DETR model."""
    
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        num_classes: int = 91,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = None
    ):
        """Initialize trainer.
        
        Args:
            model_name: Pretrained model name
            num_classes: Number of object classes
            learning_rate: Learning rate
            weight_decay: Weight decay
            device: Device to train on
        """
        self.device = torch.device(
            device if device and torch.cuda.is_available() 
            else 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"Training on device: {self.device}")
        
        # Load model and processor
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.learning_rate = learning_rate
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (pixel_values, targets) in enumerate(pbar):
            pixel_values = pixel_values.to(self.device)
            
            # Move targets to device
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for pixel_values, targets in tqdm(dataloader, desc="Validating"):
                pixel_values = pixel_values.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                outputs = self.model(pixel_values=pixel_values, labels=targets)
                loss = outputs.loss
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        num_epochs: int = 50,
        checkpoint_dir: str = "./checkpoints",
        save_every: int = 5
    ):
        """Full training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_dataloader, epoch)
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                print(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(checkpoint_dir / "best_model")
                    print(f"Saved best model with val_loss: {val_loss:.4f}")
            
            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch}")
                print(f"Saved checkpoint at epoch {epoch}")
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint.
        
        Args:
            path: Directory to save checkpoint
        """
        path.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint.
        
        Args:
            path: Directory containing checkpoint
        """
        self.model = DetrForObjectDetection.from_pretrained(path)
        self.processor = DetrImageProcessor.from_pretrained(path)
        self.model.to(self.device)


def train_model(config_path: str = "config.yaml"):
    """Main training function.
    
    Args:
        config_path: Path to configuration file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup trainer
    trainer = DETRTrainer(
        model_name=config['model']['name'],
        num_classes=config['model'].get('num_classes', 91),
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Load datasets
    print("Loading datasets...")
    # Note: This is a placeholder. In practice, you'd load actual COCO format data
    # train_dataset = COCODataset(...)
    # val_dataset = COCODataset(...)
    # train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'])
    # val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    
    print("Dataset loading not implemented - add your COCO format dataset")
    print("See README for dataset format requirements")
    
    # Train
    # trainer.train(train_loader, val_loader, num_epochs=config['training']['num_epochs'])


if __name__ == "__main__":
    train_model()
