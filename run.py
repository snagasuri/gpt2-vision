import torch
import argparse
import defs

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
    torch.multiprocessing.spawn(defs.main_worker, args=(world_size, args), nprocs=world_size, join=True)