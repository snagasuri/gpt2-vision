import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel
import argparse
import logging

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

def load_checkpoint(projection_matrix, filepath):
    if os.path.isfile(filepath):
        logger.info(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath)
        projection_matrix.load_state_dict({
            'projection.weight': checkpoint['module.projection.weight'],
            'projection.bias': checkpoint['module.projection.bias']
        })
        logger.info("Checkpoint loaded successfully")
    else:
        logger.warning(f"No checkpoint found at {filepath}")
def test_inference(image_path, projection_matrix, clip_model, tokenizer, model, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Extract visual features using CLIP
    with torch.no_grad():
        visual_features = clip_model.get_image_features(pixel_values)

    # Project visual features to language model embedding space
    projected_features = projection_matrix(visual_features)

    # Prepare the input embeddings for the language model
    prompt = "Today is a "
    inputs = tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=512).input_ids.to(device)
    input_embeds = model.transformer.wte(inputs)
    input_embeds = torch.cat((projected_features.unsqueeze(1), input_embeds), dim=1)

    # Create attention mask
    visual_attention_mask = torch.ones(projected_features.shape[0], 1, dtype=torch.long).to(inputs.device)
    attention_mask = torch.cat((visual_attention_mask, (inputs != tokenizer.pad_token_id).long()), dim=1)

    # Generate text using the language model
    with torch.no_grad():
        outputs = model.generate(inputs_embeds=input_embeds, attention_mask=attention_mask, max_length=100)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the models
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token

    model = AutoModelForCausalLM.from_pretrained(args.language_model).to(device)

    # Initialize the projection matrix
    clip_dim = clip_model.config.projection_dim
    gpt2_dim = model.config.hidden_size
    projection_matrix = ProjectionMatrix(clip_dim, gpt2_dim).to(device)

    # Load the pre-trained projection matrix
    load_checkpoint(projection_matrix, args.projection_checkpoint)

    # Set models to evaluation mode
    clip_model.eval()
    model.eval()
    projection_matrix.eval()

    # Test inference
    generated_text = test_inference(args.image_path, projection_matrix, clip_model, tokenizer, model, device)
    logger.info(f"Generated Text: {generated_text}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='train2017/000000001059.jpg')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14', help='CLIP model name')
    parser.add_argument('--language_model', type=str, default='openai-community/gpt2-xl', help='Language model name')
    parser.add_argument('--projection_checkpoint', type=str, default='checkpoints/llava-v1.5-13b-pretrain/checkpoint_epoch_2.pt')
    
    args = parser.parse_args()
    
    main(args)
