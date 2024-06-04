import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel, get_scheduler
from transformers.optimization import AdamW
from torch.cuda.amp import GradScaler, autocast
import argparse
import logging
import gc
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk

nltk.download('punkt')
random_names = ["C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000056240.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000272967.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000012756.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000472893.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000004575.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000016509.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000210010.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000126734.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000120615.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000119525.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000571562.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000529193.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000048571.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000294823.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000481732.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000102488.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000567571.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000384325.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000306620.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000139561.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000439486.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000275613.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000267203.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000353881.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000432967.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000517362.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000078550.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000425184.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000412194.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000509419.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000241761.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000174015.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000134835.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000511967.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000041730.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000351796.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000027977.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000169988.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000396213.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000402042.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000015174.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000353893.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000386161.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000290511.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000338741.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000542674.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000153031.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000217201.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000129278.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000369268.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000198881.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000262554.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000481954.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000461333.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000136945.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000347243.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000341328.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000441062.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000449808.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000164751.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000555808.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000181156.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000094499.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000192236.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000530730.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000496374.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000551609.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000339796.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000139150.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000058405.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000330808.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000023929.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000290122.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000381112.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000531366.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000360606.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000355051.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000343696.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000551679.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000053691.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000201655.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000096809.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000477162.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000352081.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000106799.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000559005.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000511271.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000238255.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000535342.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000543534.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000133791.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000252608.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000097841.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000088499.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000135894.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000068482.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000483069.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000249384.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000091543.jpg", "C:/Users/ramna/Documents/gpt2-multimodal/train2017/000000499801.jpg"]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the projection matrix class with dropout
class ProjectionMatrix(nn.Module):
    def __init__(self, clip_dim, gpt2_dim, dropout=0.1):
        super(ProjectionMatrix, self).__init__()
        self.projection = nn.Linear(clip_dim, gpt2_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, visual_features):
        projected_features = self.dropout(self.projection(visual_features))
        return projected_features

# Custom dataset for image-text pairs
class ImageTextDataset(Dataset):
    def __init__(self, json_data, image_dir, processor, tokenizer):
        self.json_data = json_data
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        entry = self.json_data[idx]
        image_filename = entry['image']
        conversations = entry['conversations']

        # Construct the conversation history
        history = ""
        for i, conv in enumerate(conversations):
            if conv['from'] == 'human':
                history += f"Human: {conv['value']}\n"
            elif conv['from'] == 'gpt':
                history += f"Assistant: {conv['value']}\n"
            if i == 1:
                answer = conv['value']

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        inputs = self.tokenizer(history, return_tensors="pt", padding='max_length', truncation=True, max_length=512).input_ids.squeeze()
        labels = self.tokenizer(answer, return_tensors="pt", padding='max_length', truncation=True, max_length=512).input_ids.squeeze()

        return pixel_values, inputs, labels

def load_checkpoint(projection_matrix, filepath):
    if os.path.isfile(filepath):
        logger.info(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath)
        projection_matrix.load_state_dict({
            'projection.weight': checkpoint['module.projection.weight'],
            'projection.bias': checkpoint['module.projection.bias']
        })
    else:
        logger.info(f"No checkpoint found at {filepath}")

def evaluate_model(projection_matrix, model, clip_model, eval_dataloader, device, tokenizer):
    model.eval()
    projection_matrix.eval()
    eval_loss = 0
    bleu_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    with torch.no_grad():
        for pixel_values, inputs, labels in eval_dataloader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            visual_features = clip_model.get_image_features(pixel_values)
            projected_features = projection_matrix(visual_features)
            
            input_embeds = model.transformer.wte(inputs)
            input_embeds = torch.cat((projected_features.unsqueeze(1), input_embeds), dim=1)
            
            visual_attention_mask = torch.ones(projected_features.shape[0], 1, dtype=torch.long).to(inputs.device)
            attention_mask = torch.cat((visual_attention_mask, (inputs != tokenizer.pad_token_id).long()), dim=1)
            
            labels = torch.cat((torch.full((labels.size(0), 1), -100).to(device), labels), dim=1)
            
            outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            eval_loss += loss.item()
            
            # Generate text for qualitative evaluation
            generated_tokens = model.generate(
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                max_length=labels.size(1),
                num_return_sequences=1
            )
            generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            target_text = tokenizer.decode(labels[0], skip_special_tokens=True)
            
            print(f"Generated Text: {generated_text}")
            print(f"Target Text: {target_text}")

            # Compute BLEU and ROUGE scores
            reference = [target_text.split()]
            candidate = generated_text.split()
            bleu_score = sentence_bleu(reference, candidate)
            bleu_scores.append(bleu_score)
            
            rouge_score = scorer.score(target_text, generated_text)
            rouge_scores.append(rouge_score)

    eval_loss /= len(eval_dataloader)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = {
        'rouge1': sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores),
        'rougeL': sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    }

    logger.info(f"Frequent Evaluation Loss: {eval_loss:.4f}")
    logger.info(f"Average BLEU Score: {avg_bleu:.4f}")
    logger.info(f"Average ROUGE Scores: Rouge-1: {avg_rouge['rouge1']:.4f}, Rouge-L: {avg_rouge['rougeL']:.4f}")

    return eval_loss, avg_bleu, avg_rouge

def main(args):
    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Paths to the JSON files and image directory
    json_file_path = args.json_file
    image_dir = args.image_dir

    # Load the JSON data
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # Initialize processor and tokenizer
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token

    model = AutoModelForCausalLM.from_pretrained(args.language_model)

    # Move models to GPU
    model.to(device)
    model = DDP(model, device_ids=[local_rank])
    logger.info(f"Moved Language Model to device {device}")

    clip_model = CLIPModel.from_pretrained(args.clip_model)
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.to(device)
    logger.info(f"Moved CLIP model to device {device}")

    # Initialize the projection matrix
    clip_dim = clip_model.config.projection_dim
    gpt2_dim = model.module.config.hidden_size
    projection_matrix = ProjectionMatrix(clip_dim, gpt2_dim).to(device)
    projection_matrix = DDP(projection_matrix, device_ids=[local_rank])
    logger.info(f"Moved Projection Matrix to device {device}")

    # Load pre-trained projection matrix
    load_checkpoint(projection_matrix, 'checkpoints/checkpoint_epoch_2.pt')

    # Optimizer and scheduler setup
    optimizer = AdamW(list(projection_matrix.parameters()) + list(model.parameters()), lr=args.learning_rate, weight_decay=0.01)  # Including weight decay
    num_epochs = args.num_epochs

    train_size = int(0.8 * len(json_data))
    val_size = len(json_data) - train_size
    train_data, val_data = torch.utils.data.random_split(json_data, [train_size, val_size])

    train_dataset = ImageTextDataset(train_data, image_dir, processor, tokenizer)
    val_dataset = ImageTextDataset(val_data, image_dir, processor, tokenizer)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers)

    # Assuming `eval_data_paths` contains the paths to the 100 images for frequent evaluation
    eval_data_paths = random_names  # Placeholder for your evaluation image paths
    eval_data = [{'image': path, 'conversations': [{"from": "human", "value": "<image>\nDescribe this image."}, {"from": "gpt", "value": "Description."}]} for path in eval_data_paths]
    eval_dataset = ImageTextDataset(eval_data, image_dir, processor, tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=int(0.03 * num_training_steps), num_training_steps=num_training_steps)

    scaler = GradScaler()

    accumulation_steps = args.accumulation_steps

    global_step = 0
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        projection_matrix.train()
        model.train()
        for step, batch in enumerate(train_dataloader):
            pixel_values, inputs, labels = batch
            pixel_values = pixel_values.to(device, non_blocking=True)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.no_grad():
                visual_features = clip_model.get_image_features(pixel_values)
            projected_features = projection_matrix(visual_features)
            
            input_embeds = model.module.transformer.wte(inputs)
            input_embeds = torch.cat((projected_features.unsqueeze(1), input_embeds), dim=1)

            visual_attention_mask = torch.ones(projected_features.shape[0], 1, dtype=torch.long).to(inputs.device)
            attention_mask = torch.cat((visual_attention_mask, (inputs != tokenizer.pad_token_id).long()), dim=1)

            labels = torch.cat((torch.full((labels.size(0), 1), -100).to(device), labels), dim=1)

            with autocast():
                outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(projection_matrix.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                lr_scheduler.step()
                gc.collect()

            if step % args.log_interval == 0:
                logger.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item() * accumulation_steps:.4f}")

            global_step += 1
            if global_step % 1000 == 0:
                evaluate_model(projection_matrix, model, clip_model, eval_dataloader, device, tokenizer)

        projection_matrix.eval()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                pixel_values, inputs, labels = batch
                pixel_values = pixel_values.to(device, non_blocking=True)
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                visual_features = clip_model.get_image_features(pixel_values)
                projected_features = projection_matrix(visual_features)
                
                input_embeds = model.module.transformer.wte(inputs)
                input_embeds = torch.cat((projected_features.unsqueeze(1), input_embeds), dim=1)
                
                visual_attention_mask = torch.ones(projected_features.shape[0], 1, dtype=torch.long).to(inputs.device)
                attention_mask = torch.cat((visual_attention_mask, (inputs != tokenizer.pad_token_id).long()), dim=1)
                
                labels = torch.cat((torch.full((labels.size(0), 1), -100).to(device), labels), dim=1)
                
                outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        logger.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        if local_rank == 0:  # Ensure only one process saves the checkpoint
            checkpoint_dir = args.output_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': projection_matrix.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': loss.item()
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    logger.info(f"End of Training")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='llava_instruct_150k.json', help='Path to the JSON file')
    parser.add_argument('--image_dir', type=str, default='train2017', help='Path to the image directory')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14', help='CLIP model name')
    parser.add_argument('--language_model', type=str, default='openai-community/gpt2-xl', help='Language model name')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/llava-llama-2-7b-finetune', help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--local_rank', type=int, help='Local rank for distributed training', default=0)
    
    args = parser.parse_args()
    
    main(args)
