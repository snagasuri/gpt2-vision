import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, CLIPProcessor
from PIL import Image
import argparse

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
        for conv in conversations:
            if conv['from'] == 'human':
                history += f"Human: {conv['value']}\n"
            elif conv['from'] == 'gpt':
                history += f"Assistant: {conv['value']}\n"

        question = conversations[0]['value'].replace("<image>\n", "")
        answer = conversations[1]['value']

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        inputs = self.tokenizer(history, return_tensors="pt", padding='max_length', truncation=True, max_length=512).input_ids.squeeze()
        labels = self.tokenizer(answer, return_tensors="pt", padding='max_length', truncation=True, max_length=512).input_ids.squeeze()

        return image_path, pixel_values, history, inputs, answer, labels

def inspect_data(args):
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

    # Create dataset and dataloader
    dataset = ImageTextDataset(json_data, image_dir, processor, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Inspect the data
    for idx, (image_path, pixel_values, history, inputs, answer, labels) in enumerate(dataloader):
        print(f"Example {idx + 1}")
        print(f"Image Path: {image_path[0]}")
        image = Image.open(image_path[0])
        image.show()
        print(f"Pixel Values: {pixel_values.shape}")
        print(f"Conversation History: {history[0]}")
        print(f"Tokenized Inputs: {inputs.shape} - {inputs}")
        print(f"Answer: {answer[0]}")
        print(f"Tokenized Labels: {labels.shape} - {labels}")
        print("\n" + "-"*50 + "\n")
        if idx == args.num_examples - 1:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='llava_instruct_150k.json', help='Path to the JSON file')
    parser.add_argument('--image_dir', type=str, default='train2017', help='Path to the image directory')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14', help='CLIP model name')
    parser.add_argument('--language_model', type=str, default='openai-community/gpt2-xl', help='Language model name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inspection')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of examples to inspect')

    args = parser.parse_args()
    
    inspect_data(args)
