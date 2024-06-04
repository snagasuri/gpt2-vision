import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPModel, CLIPProcessor, get_scheduler
from transformers.optimization import AdamW
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the projection matrix class
class ProjectionMatrix(nn.Module):
    def __init__(self, clip_dim, gpt2_dim):
        super(ProjectionMatrix, self).__init__()
        self.projection = nn.Linear(clip_dim, gpt2_dim)
    
    def forward(self, visual_features):
        projected_features = self.projection(visual_features)
        return projected_features

# Custom dataset for image-text pairs
class ImageTextDataset(Dataset):
    def __init__(self, json_data, meta_data, image_dir, processor, tokenizer):
        self.json_data = json_data
        self.meta_data = {entry['id']: entry for entry in meta_data}
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        entry = self.json_data[idx]
        image_filename = entry['image']
        caption = entry['conversations'][0]['value']

        # Load the image from the local directory
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        inputs = self.tokenizer(caption, return_tensors="pt", padding='max_length', truncation=True, max_length=512).input_ids.squeeze()

        return pixel_values, inputs

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def print_memory_stats(device):
    logger.info(f"Device {device}: Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
    logger.info(f"Device {device}: Cached: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
    logger.info(f"Device {device}: Total: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.2f} GB")

def load_checkpoint(projection_matrix, optimizer, scheduler, filepath):
    if os.path.isfile(filepath):
        logger.info(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath)
        
        # Print checkpoint keys for debugging
        logger.info(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Load the state dict for the projection matrix directly
        projection_matrix.module.load_state_dict({
            'projection.weight': checkpoint['module.projection.weight'],
            'projection.bias': checkpoint['module.projection.bias']
        })  # Use .module for DDP
        
        # Load optimizer state if it exists
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logger.warning("Optimizer state not found in checkpoint, reinitializing optimizer.")
        
        # Load scheduler state if it exists
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            logger.warning("Scheduler state not found in checkpoint, reinitializing scheduler.")
        
        # Load epoch and loss if they exist
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)
        
        return epoch, loss
    else:
        logger.info(f"No checkpoint found at {filepath}")
        return None, None

def main_worker(rank, world_size, args):
    setup(rank, world_size)

    # Paths to the JSON files and image directory
    json_file_path = args.json_file
    meta_file_path = args.meta_file
    image_dir = args.image_dir

    # Load the JSON data
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    with open(meta_file_path, 'r') as f:
        meta_data = json.load(f)

    # Initialize processor and tokenizer
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    model = AutoModelForCausalLM.from_pretrained(args.language_model)

    # Freeze GPT-2 model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Print memory stats before moving models
    print_memory_stats(rank)

    # Move models to GPU and print memory stats
    model.to(rank)
    logger.info(f"Moved Language Model to device {rank}")
    print_memory_stats(rank)

    clip_model = CLIPModel.from_pretrained(args.clip_model)
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.to(rank)
    logger.info(f"Moved CLIP model to device {rank}")
    print_memory_stats(rank)

    # Initialize the projection matrix
    clip_dim = clip_model.config.projection_dim
    gpt2_dim = model.config.hidden_size
    projection_matrix = ProjectionMatrix(clip_dim, gpt2_dim).to(rank)
    logger.info(f"Moved Projection Matrix to device {rank}")
    print_memory_stats(rank)

    # Use Distributed Data Parallel for projection matrix
    projection_matrix = DDP(projection_matrix, device_ids=[rank])

    # Optimizer and scheduler setup
    optimizer = AdamW(projection_matrix.parameters(), lr=args.learning_rate, weight_decay=0.0)
    num_epochs = args.num_epochs
    
    train_size = int(0.8 * len(json_data))
    val_size = len(json_data) - train_size
    train_data, val_data = torch.utils.data.random_split(json_data, [train_size, val_size])
    
    train_dataset = ImageTextDataset(train_data, meta_data, image_dir, processor, tokenizer)
    val_dataset = ImageTextDataset(val_data, meta_data, image_dir, processor, tokenizer)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=val_sampler)
    
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=int(0.03 * num_training_steps), num_training_steps=num_training_steps)

    # Mixed precision training
    scaler = GradScaler()

    # Load checkpoint if exists
    start_epoch, start_loss = load_checkpoint(projection_matrix, optimizer, lr_scheduler, os.path.join(args.output_dir, 'checkpoint_epoch_1.pt'))
    if start_epoch is None:
        start_epoch = 0  # Start from scratch if no checkpoint is found

    model.eval()  # Set the Language Model to evaluation mode
    projection_matrix.train()

    accumulation_steps = args.accumulation_steps

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Training loop
        projection_matrix.train()
        for step, batch in enumerate(train_dataloader):
            pixel_values, inputs = batch
            pixel_values = pixel_values.to(rank, non_blocking=True)
            inputs = inputs.to(rank, non_blocking=True)
            
            with torch.no_grad():
                visual_features = clip_model.get_image_features(pixel_values)
            projected_features = projection_matrix(visual_features)
            
            # Prepare inputs for the language model
            input_embeds = model.transformer.wte(inputs)
            input_embeds = torch.cat((projected_features.unsqueeze(1), input_embeds), dim=1)

            # Create attention mask
            visual_attention_mask = torch.ones(projected_features.shape[0], 1, dtype=torch.long).to(inputs.device)
            attention_mask = torch.cat((visual_attention_mask, (inputs != tokenizer.pad_token_id).long()), dim=1)

            # Inside the training loop, before calling the model
            labels = inputs.clone()  # Clone the inputs to use as labels
            labels = torch.cat((torch.full((labels.size(0), 1), -100).to(rank), labels), dim=1)  # Adjust labels to account for visual embeddings

            with autocast():
                outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps

            # Synchronize the loss across all processes
            reduced_loss = loss.clone()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= world_size

            scaler.scale(reduced_loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()  # Add this line to clear the cache
                lr_scheduler.step()
                gc.collect()  # Explicitly call garbage collection

            if step % args.log_interval == 0:
                logger.info(f"Epoch {epoch + 1}, Step {step}, Loss: {reduced_loss.item() * accumulation_steps:.4f}")
                print_memory_stats(rank)  # Print memory stats
        
        # Validation loop
        projection_matrix.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                pixel_values, inputs = batch
                pixel_values = pixel_values.to(rank, non_blocking=True)
                inputs = inputs.to(rank, non_blocking=True)
                
                visual_features = clip_model.get_image_features(pixel_values)
                projected_features = projection_matrix(visual_features)
                
                input_embeds = model.transformer.wte(inputs)
                input_embeds = torch.cat((projected_features.unsqueeze(1), input_embeds), dim=1)
                
                visual_attention_mask = torch.ones(projected_features.shape[0], 1, dtype=torch.long).to(inputs.device)
                attention_mask = torch.cat((visual_attention_mask, (inputs != tokenizer.pad_token_id).long()), dim=1)
                
                labels = inputs.clone()
                labels = torch.cat((torch.full((labels.size(0), 1), -100).to(rank), labels), dim=1)
                
                outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        logger.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        if rank == 0:
            checkpoint_dir = args.output_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'module.projection.weight': projection_matrix.module.projection.weight,
                'module.projection.bias': projection_matrix.module.projection.bias,
                'optimizer_state_dict': optimizer.state_dict(),  # Ensure optimizer state is saved
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': reduced_loss.item()
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Print memory stats after each epoch
    logger.info(f"End of Training")
    print_memory_stats(rank)

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='chat.json', help='Path to the chat JSON file')
    parser.add_argument('--meta_file', type=str, default='metadata.json', help='Path to the metadata JSON file')
    parser.add_argument('--image_dir', type=str, default='/workspace/images/', help='Path to the image directory')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14', help='CLIP model name')
    parser.add_argument('--language_model', type=str, default='KnutJaegersberg/gpt2-chatbot', help='Language model name')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/llava-v1.5-13b-pretrain', help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
