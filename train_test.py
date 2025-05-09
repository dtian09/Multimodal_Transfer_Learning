'''training and testing the multimodal transformer model

 transformer = CLIP image encoder and text encoder + decoder

train and test decoder on flickr30k Hugging Face dataset
'''
from math import comb
from torch.nn.modules import padding
from torch import nn
import transformers
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score
#import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
#from PIL import Image
from transformers import CLIPModel, CLIPTokenizerFast
from decoder import ImageCaptionDecoder
from datasets import Dataset
from Flickr30kDataset import Flickr30kDataset
import os

# Load model and tokenizer
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_hidden_states=True)
clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
trainset = Flickr30kDataset(split="train", max_length=31)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Decoder
decoder = ImageCaptionDecoder(
    hidden_size=hidden_size,
    num_heads=num_heads,
    vocab_size=vocab_size,
    num_decoder_layers=num_decoder_layers,
    dropout=dropout
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)

# Special tokens
sos_token_id = clip_tokenizer.convert_tokens_to_ids("<|startoftext|>")
eos_token_id = clip_tokenizer.convert_tokens_to_ids("<|endoftext|>")

# Training
best_val_acc = 0
early_stop_counter = 0

#load valiation data
valset = Flickr30kDataset(split="val", max_length=31)
valloader = DataLoader(valset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    decoder.train()
    total_loss = 0
    batch_num = 0
    for batch in tqdm(trainloader, desc=f"Epoch {epoch+1} Training"):
        # Cache combined input if needed
        combined_path = f"combined_inputs_batch_{batch_num}.pt"

        if os.path.exists(combined_path):
            combined_input = torch.load(combined_path)
        else:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            captions = batch["caption"]

            # Encode images
            with torch.no_grad():
                image_features = clip_model.vision_model(pixel_values=images).last_hidden_state  # [B, P, D]
            image_patch_len = image_features.size(1)

            # Encode text (get token embeddings only)
            with torch.no_grad():
                text_inputs = clip_tokenizer(captions, padding="max_length", truncation=True,
                                            max_length=input_ids.size(1), return_tensors="pt").to(device)
                text_embeddings = clip_model.text_model.embeddings(input_ids=text_inputs["input_ids"])  # [B, L, D]

            # Get <sos> embedding
            sos_embedding = clip_model.text_model.embeddings.token_embedding(
                torch.tensor([sos_token_id], device=device)
            ).expand(images.size(0), 1, -1)

            # Combine image + <sos> + caption token embeddings
            combined_input = torch.cat([image_features, sos_embedding, text_embeddings[:, :-1, :]], dim=1)
            torch.save(combined_input, combined_path)

        batch_num+=1
        
        # Append </sos> token to targets
        eos_column = torch.full((input_ids.size(0), 1), eos_token_id, dtype=torch.long, device=device)
        targets = torch.cat([input_ids, eos_column], dim=1)  # [B, L+1]

        # Create padding mask for the decoder
        pad_mask = (targets == 0)  # [B, L+1]

        # Decoder forward
        logits = decoder(combined_input, image_patch_len=image_patch_len, padding_mask=pad_mask)
        logits = logits[:, -targets.size(1):, :]  # align with caption length

        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(trainloader)
    wandb.log({"loss": avg_loss})
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    # Validation
    decoder.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        batch_num = 0
        for batch in valloader:
            # Cache combined input if needed
            combined_val_path = f"combined_val_inputs_batch_{batch_num}.pt"

            if os.path.exists(combined_val_path):
                combined_val_input = torch.load(combined_val_path)
            else:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                captions = batch["caption"]

                image_features = clip_model.vision_model(pixel_values=images).last_hidden_state
                image_patch_len = image_features.size(1)

                text_inputs = clip_tokenizer(captions, padding="max_length", truncation=True,
                                            max_length=input_ids.size(1), return_tensors="pt").to(device)
                text_embeddings = clip_model.text_model.embeddings(input_ids=text_inputs["input_ids"])

                sos_embedding = clip_model.text_model.embeddings.token_embedding(
                    torch.tensor([sos_token_id], device=device)
                ).expand(images.size(0), 1, -1)

                combined_input = torch.cat([image_features, sos_embedding, text_embeddings[:, :-1, :]], dim=1)
                torch.save(combined_input, combined_val_path)
            
            batch_num+=1

            eos_column = torch.full((input_ids.size(0), 1), eos_token_id, dtype=torch.long, device=device)
            targets = torch.cat([input_ids, eos_column], dim=1)

            pad_mask = (targets == 0)

            logits = decoder(combined_input, image_patch_len=image_patch_len, padding_mask=pad_mask)
            logits = logits[:, -targets.size(1):, :]

            predictions = logits.argmax(dim=-1)

            val_preds.extend(predictions.cpu().numpy().flatten())
            val_labels.extend(targets.cpu().numpy().flatten())

    val_acc = accuracy_score(val_labels, val_preds)
    wandb.log({"val_accuracy": val_acc})
    print(f"Epoch {epoch+1} - Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stop_counter = 0
        torch.save(decoder.state_dict(), "best_decoder.pt")
        print(f"[Epoch {epoch}] New best model with val_acc: {val_acc:.4f}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            # Create artifact and log it
            artifact = wandb.Artifact("best-model", type="model")
            artifact.add_file("best_decoder.pt")
            wandb.log_artifact(artifact)
            break

#load test data
test = Flickr30kDataset(split="test", max_length=31)
testloader = DataLoader(test, batch_size=32, shuffle=True)

decoder.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in testloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        captions = batch["caption"]

        image_features = clip_model.vision_model(pixel_values=images).last_hidden_state
        image_patch_len = image_features.size(1)

        text_inputs = clip_tokenizer(captions, padding="max_length", truncation=True,
                                    max_length=input_ids.size(1), return_tensors="pt").to(device)
        text_embeddings = clip_model.text_model.embeddings(input_ids=text_inputs["input_ids"])

        sos_embedding = clip_model.text_model.embeddings.token_embedding(
            torch.tensor([sos_token_id], device=device)
        ).expand(images.size(0), 1, -1)

        combined_input = torch.cat([image_features, sos_embedding, text_embeddings[:, :-1, :]], dim=1)
        torch.save(combined_input, combined_val_path)       

        eos_column = torch.full((input_ids.size(0), 1), eos_token_id, dtype=torch.long, device=device)
        targets = torch.cat([input_ids, eos_column], dim=1)

        pad_mask = (targets == 0)

        logits = decoder(combined_input, image_patch_len=image_patch_len, padding_mask=pad_mask)
        logits = logits[:, -targets.size(1):, :]

        predictions = logits.argmax(dim=-1)

        test_preds.extend(predictions.cpu().numpy().flatten())
        test_labels.extend(targets.cpu().numpy().flatten())

test_acc = accuracy_score(test_labels, test_preds)
wandb.log({"test_accuracy": test_acc})
print(f"Test Acc: {test_acc:.4f}")
wandb.finish()

###clear up
#delete combined inputs from disk
for epoch in range(num_epochs):
    batch_num = 0
    for batch in trainloader:
        combined_path = f"combined_inputs_batch_{batch_num}.pt"
        if os.path.exists(combined_path):
            os.remove(combined_path)
            print(f"deleted: {combined_path} from disk.")
        batch_num+=1
batch_num = 0
for batch in valloader:
    combined_path = f"combined_val_inputs_batch_{batch_num}.pt"
    if os.path.exists(combined_path):
        os.remove(combined_path)
        print(f"deleted: {combined_path} from disk.")
    batch_num+=1