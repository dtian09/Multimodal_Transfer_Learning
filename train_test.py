#training and testing the multimodal transformer model
from preprocessCLIP import load_dataloaders
import transformers
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score
import numpy as np

# Initialize wandb
wandb.init(
        project="multimodal-transfer-learning", 
        entity="dtian",
        config={
            "batch_size": 64,
            "hidden_size": 512,
            "num_heads": 8,
            "vocab_size": 32000,
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

# Load dataloaders and model
train_loader, test_loader = load_dataloaders()
model = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

# Split training data into train and validation (80-20 split)
train_size = int(0.8 * len(train_loader.dataset))
val_size = len(train_loader.dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])

# Create new dataloaders with batch size from wandb config
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Learnable Positional Encoding
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.randn(max_len, d_model))
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        position_embeddings = self.position_embeddings[:seq_len].unsqueeze(0)  # [1, seq_len, d_model]
        return x + position_embeddings

# Define a single decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def create_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)
    
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        
        # Create attention mask
        attn_mask = self.create_mask(x.size(1), x.device)
        
        # First residual connection: skip masked multi-head attention
        residual1 = x
        
        # Self-attention on sequence
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask
        )
        x = self.norm1(residual1 + self.dropout(attn_output))
        
        # Second residual connection: skip feed-forward network
        residual2 = x
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm2(residual2 + self.dropout(ff_output))
        
        return x

# Define the decoder with residual connections and linear projection

class ImageCaptionDecoder(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, vocab_size=32000, num_decoder_layers=6, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        
        # Positional encoding
        self.pos_encoder = LearnablePositionalEncoding(hidden_size)
        
        # Stack of decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Linear projection to vocabulary size
        self.projection = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, combined_embeddings):
        # combined_embeddings: [batch_size, num_patches+seq_len, hidden_size]
        
        # Add positional encoding
        combined_embeddings = self.pos_encoder(combined_embeddings)
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            combined_embeddings = decoder_layer(combined_embeddings)
        
        # Project only caption part (assumes first token is image context)
        logits = self.projection(combined_embeddings[:, 1:])  # [batch_size, seq_len, vocab_size]
        
        return logits

def get_patch_embeddings_from_encoder(images):
    """Process full images (not patches) using CLIP vision encoder."""
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=images, return_dict=True)
        hidden_state = vision_outputs.last_hidden_state  # [B, num_patches+1, hidden_size]
        return hidden_state[:, 1:]  # Exclude [CLS] token

def compute_patch_embeddings(dataloader, device):
    patch_embeddings = []    
    for batch in tqdm(dataloader, desc="Processing images"):
        images = batch['images'].to(device)
        batch_patch_embeddings = get_patch_embeddings_from_encoder(images)
        patch_embeddings.append(batch_patch_embeddings)
    return patch_embeddings

def compute_text_embeddings(dataloader, device):
    text_embeddings = []
    labels = []

    for batch in tqdm(dataloader, desc="Processing captions"):
        caption_input, caption_label = batch['captions']  # Unpack the full batch
        caption_input = caption_input.to(device)          # Shape: [B, seq_len]
        caption_label = caption_label.to(device)

        with torch.no_grad():
            text_outputs = model.text_model(
                input_ids=caption_input,
                return_dict=True
            )
            caption_embeddings = text_outputs.last_hidden_state  # [B, seq_len, hidden]

        text_embeddings.append(caption_embeddings)
        labels.append(caption_label)

    return text_embeddings, labels


def precompute_decoder_inputs(patch_embeddings_list, text_embeddings_list):
    combined_inputs = []
    for batch_patches, batch_captions in zip(patch_embeddings_list, text_embeddings_list):
        batch_combined = []
        for patch_embed, caption_embed in zip(batch_patches, batch_captions):
            combined = torch.cat([patch_embed.unsqueeze(0), caption_embed], dim=1)  # Add batch dimension to patch_embed
            batch_combined.append(combined)
        combined_inputs.append(batch_combined)
    return combined_inputs

# Pre-compute context vectors for all datasets
#print("Pre-computing context vectors for training data...")
#train_patch_embeddings = compute_patch_embeddings(train_loader, device)

#print("Pre-computing context vectors for validation data...")
#val_patch_embeddings = compute_patch_embeddings(val_loader, device)

#print("Pre-computing context vectors for test data...")
#test_patch_embeddings = compute_patch_embeddings(test_loader, device)

# Pre-compute text embeddings for all datasets
print("Pre-computing text embeddings for training data...")
train_text_embeddings, train_labels = compute_text_embeddings(train_loader, device)

print("Pre-computing text embeddings for validation data...")
val_text_embeddings, val_labels = compute_text_embeddings(val_loader, device)

print("Pre-computing text embeddings for test data...")
test_text_embeddings, test_labels = compute_text_embeddings(test_loader, device)

print("Creating decoder input embeddings...")
train_combined_inputs = precompute_decoder_inputs(train_patch_embeddings, train_text_embeddings)
val_combined_inputs = precompute_decoder_inputs(val_patch_embeddings, val_text_embeddings)
test_combined_inputs = precompute_decoder_inputs(test_patch_embeddings, test_text_embeddings)

# Initialize decoder with parameters from wandb config
decoder = ImageCaptionDecoder(
    hidden_size=hidden_size,
    num_heads=num_heads,
    vocab_size=vocab_size,
    num_decoder_layers=num_decoder_layers,
    dropout=dropout
).to(device)

# Initialize optimizer with learning rate from wandb config
optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Initialize loss tracking
    total_train_loss = 0
    num_train_batches = 0
    
    # Set model to training mode
    decoder.train()
    
    #iterate over the training loader
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Get pre-computed context vectors and text embeddings
        combined_inputs = train_combined_inputs[batch_idx]
        batch_labels = train_labels[batch_idx]
        
        # Process each image-caption pair
        batch_loss = 0
        for i, (combined_input, caption_label) in enumerate(zip(combined_inputs, batch_labels)):
            # Get context vector for this image
            combined_input = combined_input.unsqueeze(0)
            
            # Pass through decoder
            logits = decoder(combined_input)
            
            # Compute cross entropy loss
            logits = logits.view(-1, logits.size(-1))
            labels = caption_label.view(-1)
            
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            loss = criterion(logits, labels)
            
            batch_loss += loss.item()
        
        # Average loss over captions in batch
        batch_loss /= len(combined_inputs)
        total_train_loss += batch_loss
        num_train_batches += 1
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Print batch loss
        print(f"Training Batch {num_train_batches} Loss: {batch_loss:.4f}")
    
    # Calculate average training loss
    avg_train_loss = total_train_loss / num_train_batches
    
    # Evaluate on validation set
    decoder.eval()
    val_predictions = []
    val_true_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")):
            combined_inputs = val_combined_inputs[batch_idx]
            batch_labels = val_labels[batch_idx]
            
            for i, (combined_input, caption_label) in enumerate(zip(combined_inputs, batch_labels)):
                combined_input = combined_input.unsqueeze(0)
                logits = decoder(combined_input)
                predictions = torch.argmax(logits, dim=-1)
                
                # Remove padding tokens (0) for accuracy calculation
                valid_predictions = predictions[predictions != 0]
                valid_labels = caption_label[caption_label != 0]
                
                val_predictions.extend(valid_predictions.cpu().numpy())
                val_true_labels.extend(valid_labels.cpu().numpy())
    
    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_accuracy": val_accuracy
    })
    
    print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save the trained model
torch.save(decoder.state_dict(), 'best_model.pth')

# Evaluation on test set
print("\nEvaluating on test set...")
test_predictions = []
test_true_labels = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Test")):
        combined_inputs = test_combined_inputs[batch_idx]
        batch_labels = test_labels[batch_idx]
        
        for i, (combined_input, caption_label) in enumerate(zip(combined_inputs, batch_labels)):
            combined_input = combined_input.unsqueeze(0)
            logits = decoder(combined_input)
            predictions = torch.argmax(logits, dim=-1)
            
            # Remove padding tokens (0) for accuracy calculation
            valid_predictions = predictions[predictions != 0]
            valid_labels = caption_label[caption_label != 0]
            
            test_predictions.extend(valid_predictions.cpu().numpy())
            test_true_labels.extend(valid_labels.cpu().numpy())

test_accuracy = accuracy_score(test_true_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Log final test accuracy to wandb
wandb.log({
    "test_accuracy": test_accuracy
})

# Close wandb
wandb.finish()











