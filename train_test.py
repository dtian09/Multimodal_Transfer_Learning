#training and testing the multimodal transformer model
from preprocess import load_dataloaders
import transformers
import torch
import torch.nn as nn
from tqdm import tqdm

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

# Define the decoder with residual connections and linear projection
class ImageCaptionDecoder(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, vocab_size=32000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Positional encoding
        self.pos_encoder = LearnablePositionalEncoding(hidden_size)
        
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
        self.dropout = nn.Dropout(0.1)
        
        # Linear projection to vocabulary size
        self.projection = nn.Linear(hidden_size, vocab_size)
        
    def create_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)
    
    def forward(self, tgt, memory):
        # tgt: caption input sequence [batch_size, seq_len, hidden_size]
        # memory: context vectors [batch_size, 1, hidden_size]
        
        # Add positional encoding to caption embeddings
        tgt = self.pos_encoder(tgt)
        
        # Concatenate context vector with caption embeddings
        # memory: [batch_size, 1, hidden_size]
        # tgt: [batch_size, seq_len, hidden_size]
        combined = torch.cat([memory, tgt], dim=1)  # [batch_size, seq_len+1, hidden_size]
        
        # Create attention mask for the combined sequence
        attn_mask = self.create_mask(combined.size(1))
        
        # First residual connection: skip masked multi-head attention
        residual1 = combined
        
        # Self-attention on combined sequence
        attn_output, _ = self.self_attn(
            query=combined,
            key=combined,
            value=combined,
            attn_mask=attn_mask
        )
        combined = self.norm1(residual1 + self.dropout(attn_output))
        
        # Second residual connection: skip feed-forward network
        residual2 = combined
        
        # Feed-forward network
        ff_output = self.feed_forward(combined)
        combined = self.norm2(residual2 + self.dropout(ff_output))
        
        # Project to vocabulary size for caption logits
        # Only project the caption part (excluding the context vector)
        logits = self.projection(combined[:, 1:])  # [batch_size, seq_len, vocab_size]
        
        return logits

# Process image patches through CLIP's vision encoder
def process_image_patches_using_encoder(image_patches):
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

# Initialize decoder
decoder = ImageCaptionDecoder().to(device)

# Initialize optimizer
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.0001)

# Early stopping parameters
patience = 3
min_delta = 0.001
best_val_loss = float('inf')
counter = 0

#for each epoch
for epoch in range(num_epochs):
    # Initialize loss tracking
    total_train_loss = 0
    num_train_batches = 0
    
    # Set model to training mode
    decoder.train()
    
    # Training loop
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Get image patches and captions
        image_patches = batch['image_patches'].to(device)
        captions = batch['captions']
        
        # Process patches through CLIP's vision encoder
        context_vectors = process_image_patches_using_encoder(image_patches)
        
        # Process each image-caption pair
        batch_loss = 0
        for i, (caption_input, caption_label) in enumerate(captions[0]):
            # Move caption to device
            caption_input = caption_input.to(device)
            caption_label = caption_label.to(device)
            
            # Get text embeddings for caption input
            text_outputs = model.text_model(
                input_ids=caption_input.unsqueeze(0),
                return_dict=True
            )
            caption_embeddings = text_outputs.last_hidden_state
            
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
        batch_loss /= len(captions[0])
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
    
    # Validation loop
    decoder.eval()
    total_val_loss = 0
    num_val_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            # Get image patches and captions
            image_patches = batch['image_patches'].to(device)
            captions = batch['captions']
            
            # Process patches through CLIP's vision encoder
            context_vectors = process_image_patches_using_encoder(image_patches)
            
            # Process each image-caption pair
            batch_loss = 0
            for i, (caption_input, caption_label) in enumerate(captions[0]):
                # Move caption to device
                caption_input = caption_input.to(device)
                caption_label = caption_label.to(device)
                
                # Get text embeddings for caption input
                text_outputs = model.text_model(
                    input_ids=caption_input.unsqueeze(0),
                    return_dict=True
                )
                caption_embeddings = text_outputs.last_hidden_state
                
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
            batch_loss /= len(captions[0])
            total_val_loss += batch_loss
            num_val_batches += 1
    
    # Calculate average validation loss
    avg_val_loss = total_val_loss / num_val_batches
    
    # Print epoch losses
    print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
    # Early stopping check
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        counter = 0
        # Save best model
        torch.save(decoder.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # You can now use logits for caption generation or other tasks











