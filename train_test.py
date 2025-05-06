#training and testing the multimodal transformer model
from preprocess import load_dataloaders
import transformers
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataloaders and model
train_loader, test_loader = load_dataloaders()
model = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

# Split training data into train and validation (80-20 split)
train_size = int(0.8 * len(train_loader.dataset))
val_size = len(train_loader.dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])

# Create new dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_loader.batch_size, shuffle=False)

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
        
    def forward(self, tgt, memory):
        # tgt: caption input sequence [batch_size, seq_len, hidden_size]
        # memory: context vectors [batch_size, 1, hidden_size]
        
        # Add positional encoding to caption embeddings
        tgt = self.pos_encoder(tgt)
        
        # Concatenate context vector with caption embeddings
        # memory: [batch_size, 1, hidden_size]
        # tgt: [batch_size, seq_len, hidden_size]
        combined = torch.cat([memory, tgt], dim=1)  # [batch_size, seq_len+1, hidden_size]
        
        # Pass through each decoder layer
        for decoder_layer in self.decoder_layers:
            combined = decoder_layer(combined)
        
        # Project to vocabulary size for caption logits
        # Only project the caption part (excluding the context vector)
        logits = self.projection(combined[:, 1:])  # [batch_size, seq_len, vocab_size]
        
        return logits

# Process image patches through CLIP's vision encoder
def get_context_vectors_from_encoder(image_patches):
    # Process each image in the batch
    context_vectors = []
    
    for patches in image_patches:
        # Process image patches through CLIP's vision encoder
        vision_outputs = model.vision_model(
            pixel_values=patches.unsqueeze(0),  # Add batch dimension
            return_dict=True
        )
        
        # Get the hidden state output instead of pooled output
        # This contains spatial information from the image patches
        # Shape: [1, num_patches + 1, hidden_size]
        # The +1 is for the [CLS] token
        hidden_state = vision_outputs.last_hidden_state
        
        # Remove the [CLS] token and use the patch embeddings as context
        # Shape: [1, num_patches, hidden_size]
        context_vector = hidden_state[:, 1:]
        
        context_vectors.append(context_vector)
    
    return torch.cat(context_vectors, dim=0)  # Shape: [batch_size, num_patches, hidden_size]

# Initialize wandb

wandb.init(
        project="multimodal-transfer-learning", 
        entity="dtian",
        config={
            "batch_size": batch_size,
            "embed_dim": embed_dim,
            "ff_dim": ff_dim, 
            "num_layers": num_layers,
            "epochs": epochs,
            "learning_rate": lr,
            "patience": patience
        })

# Initialize decoder
decoder = ImageCaptionDecoder().to(device)

# Initialize optimizer
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.0001)

num_epochs = 10

# Pre-compute context vectors for all training data
print("Pre-computing context vectors for training data...")
train_context_vectors = []
for batch in tqdm(train_loader, desc="Processing training images"):
    image_patches = batch['image_patches'].to(device)
    context_vectors = get_context_vectors_from_encoder(image_patches)
    train_context_vectors.append(context_vectors)

# Pre-compute context vectors for all validation data
print("Pre-computing context vectors for validation data...")
val_context_vectors = []
for batch in tqdm(val_loader, desc="Processing validation images"):
    image_patches = batch['image_patches'].to(device)
    context_vectors = get_context_vectors_from_encoder(image_patches)
    val_context_vectors.append(context_vectors)

# Pre-compute context vectors for all test data
print("Pre-computing context vectors for test data...")
test_context_vectors = []
for batch in tqdm(test_loader, desc="Processing test images"):
    image_patches = batch['image_patches'].to(device)
    context_vectors = get_context_vectors_from_encoder(image_patches)
    test_context_vectors.append(context_vectors)

# Pre-compute text embeddings for all training data
print("Pre-computing text embeddings for training data...")
train_text_embeddings = []
train_labels = []
for batch in tqdm(train_loader, desc="Processing training captions"):
    captions = batch['captions']
    batch_embeddings = []
    batch_labels = []
    for caption_input, caption_label in captions[0]:
        caption_input = caption_input.to(device)
        caption_label = caption_label.to(device)
        text_outputs = model.text_model(
            input_ids=caption_input.unsqueeze(0),
            return_dict=True
        )
        caption_embeddings = text_outputs.last_hidden_state
        batch_embeddings.append(caption_embeddings)
        batch_labels.append(caption_label)
    train_text_embeddings.append(batch_embeddings)
    train_labels.append(batch_labels)

# Pre-compute text embeddings for all validation data
print("Pre-computing text embeddings for validation data...")
val_text_embeddings = []
val_labels = []
for batch in tqdm(val_loader, desc="Processing validation captions"):
    captions = batch['captions']
    batch_embeddings = []
    batch_labels = []
    for caption_input, caption_label in captions[0]:
        caption_input = caption_input.to(device)
        caption_label = caption_label.to(device)
        text_outputs = model.text_model(
            input_ids=caption_input.unsqueeze(0),
            return_dict=True
        )
        caption_embeddings = text_outputs.last_hidden_state
        batch_embeddings.append(caption_embeddings)
        batch_labels.append(caption_label)
    val_text_embeddings.append(batch_embeddings)
    val_labels.append(batch_labels)

# Pre-compute text embeddings for all test data
print("Pre-computing text embeddings for test data...")
test_text_embeddings = []
test_labels = []
for batch in tqdm(test_loader, desc="Processing test captions"):
    captions = batch['captions']
    batch_embeddings = []
    batch_labels = []
    for caption_input, caption_label in captions[0]:
        caption_input = caption_input.to(device)
        caption_label = caption_label.to(device)
        text_outputs = model.text_model(
            input_ids=caption_input.unsqueeze(0),
            return_dict=True
        )
        caption_embeddings = text_outputs.last_hidden_state
        batch_embeddings.append(caption_embeddings)
        batch_labels.append(caption_label)
    test_text_embeddings.append(batch_embeddings)
    test_labels.append(batch_labels)

# Training loop
for epoch in range(num_epochs):
    # Initialize loss tracking
    total_train_loss = 0
    num_train_batches = 0
    
    # Set model to training mode
    decoder.train()
    
    # Training loop
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Get pre-computed context vectors and text embeddings
        context_vectors = train_context_vectors[batch_idx]
        batch_embeddings = train_text_embeddings[batch_idx]
        batch_labels = train_labels[batch_idx]
        
        # Process each image-caption pair
        batch_loss = 0
        for i, (caption_embeddings, caption_label) in enumerate(zip(batch_embeddings, batch_labels)):
            # Get context vector for this image
            context_vector = context_vectors[i].unsqueeze(0)
            
            # Pass through decoder
            logits = decoder(caption_embeddings, context_vector)
            
            # Compute cross entropy loss
            logits = logits.view(-1, logits.size(-1))
            labels = caption_label.view(-1)
            
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            loss = criterion(logits, labels)
            
            batch_loss += loss.item()
        
        # Average loss over captions in batch
        batch_loss /= len(batch_embeddings)
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
            context_vectors = val_context_vectors[batch_idx]
            batch_embeddings = val_text_embeddings[batch_idx]
            batch_labels = val_labels[batch_idx]
            
            for i, (caption_embeddings, caption_label) in enumerate(zip(batch_embeddings, batch_labels)):
                context_vector = context_vectors[i].unsqueeze(0)
                logits = decoder(caption_embeddings, context_vector)
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
        context_vectors = test_context_vectors[batch_idx]
        batch_embeddings = test_text_embeddings[batch_idx]
        batch_labels = test_labels[batch_idx]
        
        for i, (caption_embeddings, caption_label) in enumerate(zip(batch_embeddings, batch_labels)):
            context_vector = context_vectors[i].unsqueeze(0)
            logits = decoder(caption_embeddings, context_vector)
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











