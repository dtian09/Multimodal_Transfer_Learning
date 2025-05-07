'''training and testing the multimodal transformer model

 transformer = CLIP image encoder and text encoder + decoder

train and test decoder on flickr30k Hugging Face dataset
'''
from torch.nn.modules import padding
import transformers
import torch
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPTokenizerFast
from decoder import ImageCaptionDecoder
from datasets import Dataset

# Load model and tokenizer
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_hidden_states=True)
clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

# Initialize wandb
wandb.init(
        project="multimodal-transfer-learning", 
        entity="dtian",
        config={
            "batch_size": 32,
            "hidden_size": 512,
            "num_heads": 8,
            "vocab_size": clip_model.config.text_config.vocab_size,
            "num_decoder_layers": 6,
            "dropout": 0.1,
            "epochs": 10,
            "learning_rate": 0.0001,
            "patience": 5
        })

# Get parameters from wandb config
batch_size = wandb.config.batch_size
hidden_size = wandb.config.hidden_size
num_heads = wandb.config.num_heads
vocab_size = wandb.config.vocab_size
num_decoder_layers = wandb.config.num_decoder_layers
dropout = wandb.config.dropout
num_epochs = wandb.config.epochs
learning_rate = wandb.config.learning_rate
patience = wandb.config.patience

# Load the Flickr30k training dataset from Hugging Face
dataset = load_dataset("nlphuji/flickr30k")
full_data = dataset['test']
train_dataset = full_data.filter(lambda x: x['split'] == 'train')
val_dataset = full_data.filter(lambda x: x['split'] == 'val')
test_dataset = full_data.filter(lambda x: x['split'] == 'test')

# def expand_flickr30k_split(split_dataset):
#     expanded_data = []
#     for example in split_dataset:
#         for caption in example["caption"]:
#             expanded_data.append({
#                 "image": example["image"],
#                 "caption": caption
#             })
#     return Dataset.from_list(expanded_data)

# train_dataset = expand_flickr30k_split(train_dataset)
# val_dataset = expand_flickr30k_split(val_dataset)
# test_dataset = expand_flickr30k_split(test_dataset)

def select_one_caption_per_image(split_dataset):
    selected_data = []
    for example in split_dataset:
        # Just pick the first caption
        caption = example["caption"][0]
        selected_data.append({
            "image": example["image"],
            "caption": caption
        })
    return Dataset.from_list(selected_data)

train_dataset = select_one_caption_per_image(train_dataset)
val_dataset = select_one_caption_per_image(val_dataset)
test_dataset = select_one_caption_per_image(test_dataset)

# Image transform (must match CLIP's expected input)
clip_image_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

# Updated collate function for using CLIP-specific transforms
def collate_fn_clip(batch):
    images = [clip_image_transform(example["image"]) for example in batch]
    captions = [example["caption"] for example in batch]
    return torch.stack(images), captions

# Wrap in a DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn_clip
)

# Validation DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn_clip
)

# Test DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn_clip
)



if clip_tokenizer.pad_token is None:
    clip_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model.to(device)
clip_model.eval()

# Ensure tokenizer has <sos>
if clip_tokenizer.bos_token is None:
    clip_tokenizer.add_special_tokens({'bos_token': '<sos>'})
    clip_model.resize_token_embeddings(len(clip_tokenizer))  # Resize embedding layer if added

sos_id = clip_tokenizer.bos_token_id


import torch.nn as nn
import torch.nn.functional as F

def compute_accuracy(logits, labels, pad_id):
    preds = torch.argmax(logits, dim=-1)
    mask = labels != pad_id
    correct = (preds == labels) & mask
    acc = correct.sum().item() / mask.sum().item()
    return acc

# Initialize decoder
decoder = ImageCaptionDecoder(hidden_size=hidden_size, vocab_size=vocab_size).to(device)

# Loss, optimizer, decoder
criterion = nn.CrossEntropyLoss(ignore_index=clip_tokenizer.pad_token_id)
optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
decoder.to(device)

best_val_acc = 0.0
patience_counter = 0

for epoch in range(num_epochs):
    decoder.train()
    total_loss = 0.0

    for images, captions in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = images.to(device)
  
        # Get image patch embeddings from vision encoder (last hidden state)
        with torch.no_grad():
            vision_outputs = clip_model.vision_model(pixel_values=images, output_hidden_states=True)
            image_patch_embeddings = vision_outputs.last_hidden_state  # [B, N+1, D]

        # Tokenize captions without special tokens
        encoded = clip_tokenizer(
            captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        input_ids = encoded["input_ids"]
        padding_mask = encoded["attention_mask"]

        # Prepend <sos> token to input_ids
        sos_tokens = torch.full((input_ids.size(0), 1), sos_id, dtype=torch.long).to(device)
        input_ids = torch.cat([sos_tokens, input_ids], dim=1).to(device)
        padding_mask = torch.cat([torch.ones_like(sos_tokens), padding_mask], dim=1).to(device)
        targets = input_ids[:, 1:]  # Shifted for teacher forcing

        # Get token embeddings from CLIP text model embedding layer
        with torch.no_grad():
            text_embeddings = clip_model.text_model.embeddings(input_ids=input_ids)  # [B, T+1, D]

        # Concatenate image + text embeddings
        combined_embeddings = torch.cat([image_patch_embeddings, text_embeddings], dim=1).to(device)  # [B, N+1 + T+1, D]

        # Pass combined embeddings to decoder (along with image patch length)
        logits = decoder(combined_embeddings, image_patch_len=image_patch_embeddings.size(1), padding_mask=padding_mask)  # [B, T+1, V]
        
        # logits can now be compared to ground truth input_ids[:, 1:] (excluding <sos>) using CrossEntropyLoss
        # Compute loss: logits -> [B, T, V], targets -> [B, T]
        loss = criterion(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    wandb.log({"train_loss": avg_train_loss, "epoch": epoch})

    # Validation
    decoder.eval()
    total_acc = 0.0
    total_count = 0

    with torch.no_grad():
        for images, captions in val_loader:
            images = images.to(device)

            encoded = clip_tokenizer(
                captions,
                padding=True,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            )
            input_ids = encoded["input_ids"]
            sos_tokens = torch.full((input_ids.size(0), 1), clip_tokenizer.bos_token_id, dtype=torch.long).to(device)
            input_ids = torch.cat([sos_tokens, input_ids], dim=1).to(device)
            targets = input_ids[:, 1:]
            
            image_out = clip_model.vision_model(pixel_values=images, output_hidden_states=True)
            image_patches = image_out.last_hidden_state

            token_embeddings = clip_model.text_model.embeddings(input_ids=input_ids)
            combined = torch.cat([image_patches, token_embeddings], dim=1)

            padding_mask = encoded["attention_mask"]
            padding_mask = torch.cat([torch.ones_like(sos_tokens), padding_mask], dim=1).to(device)
            
            logits = decoder(combined, 
                             image_patch_len=image_patches.size(1),
                             padding_mask=padding_mask)

            acc = compute_accuracy(logits, targets, pad_id=clip_tokenizer.pad_token_id)
            total_acc += acc
            total_count += 1

    val_acc = total_acc / total_count
    wandb.log({"val_accuracy": val_acc}, step=epoch)

    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(decoder.state_dict(), "best_decoder.pt")
        wandb.save("best_decoder.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Test Evaluation
decoder.load_state_dict(torch.load("best_decoder.pt"))
decoder.eval()
test_acc = 0.0
test_batches = 0

with torch.no_grad():
    for images, captions in test_loader:
        images = images.to(device)

        encoded = clip_tokenizer(
            captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        input_ids = encoded["input_ids"]
        sos_tokens = torch.full((input_ids.size(0), 1), clip_tokenizer.bos_token_id, dtype=torch.long).to(device)
        input_ids = torch.cat([sos_tokens, input_ids], dim=1).to(device)
        targets = input_ids[:, 1:]

        image_out = clip_model.vision_model(pixel_values=images, output_hidden_states=True)
        image_patches = image_out.last_hidden_state
        token_embeddings = clip_model.text_model.embeddings(input_ids=input_ids)
        combined = torch.cat([image_patches, token_embeddings], dim=1)

        padding_mask = encoded["attention_mask"]
        padding_mask = torch.cat([torch.ones_like(sos_tokens), padding_mask], dim=1).to(device)
            
        logits = decoder(combined, image_patch_len=image_patches.size(1), padding_mask=padding_mask)
        acc = compute_accuracy(logits, targets, pad_id=clip_tokenizer.pad_token_id)
        test_acc += acc
        test_batches += 1

final_test_acc = test_acc / test_batches
wandb.log({"test_accuracy": final_test_acc})
print(f"Final test accuracy: {final_test_acc:.4f}")
wandb.finish()