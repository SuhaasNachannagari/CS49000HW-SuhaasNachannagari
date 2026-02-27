import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import List, Dict, Union
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# --- Utility: Data Loader ---
def get_data(path: str) -> List[Dict[str, Union[str, int]]]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# --- Dataset Class ---
class SarcasmDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, max_length: int = 128):
        """
        Args:
            data: List of dictionaries (from get_data)
            tokenizer: Instance of BertTokenizer
            max_length: Maximum sequence length for truncation/padding
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a dictionary containing:
            - input_ids: (max_length,) Tensor of token IDs 
            - attention_mask: (max_length,) Tensor for attention masking
            - label: (1,) Tensor containing the label
        """
        # TODO: - Retrieve text and label from self.data[index]
        #       - Use self.tokenizer to tokenize the text and
        #         ensure you handle padding, truncation, and 
        #         return a flattened tensor for input_ids and attention_mask.
        
        raise NotImplementedError("Dataset __getitem__ not implemented")

# --- Model Class ---
class SarcasmBERT(nn.Module):
    def __init__(self):
        super(SarcasmBERT, self).__init__()
        
        # TODO: Load pre-trained BERT model (bert-base-uncased)
        
        # TODO: Define a linear classification layer
        
        raise NotImplementedError("SarcasmBERT initialization not implemented")

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            logits: (batch_size, 2)
        """
        # TODO: Pass inputs through BERT

        # TODO: Extract the representation of the [CLS] token

        # TODO: Pass [CLS] representation through the classifier layer
        
        raise NotImplementedError("SarcasmBERT forward pass not implemented")

# --- Training Loop ---
def train_loop(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    lr: float, 
    epochs: int,
    **kwargs
) -> List[float]:
    
    # We use AdamW as the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # TODO: Move batch components to device
            
            # TODO: Forward pass, Loss calculation, Backward pass, Optimizer step
            pass
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Feel free to pass additional arguments (like validation dataloader) 
        # through kwargs and perform validation here.
        
    return loss_history

# --- Main Execution ---
if __name__ == "__main__":
    # Check for GPU/MPS
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    try:
        train_data = get_data('train.jsonl')
        valid_data = get_data('valid.jsonl')
    except NotImplementedError:
        print("Error: Implement get_data first.")
        exit(1)
    except FileNotFoundError:
        print("Error: Data files not found.")
        exit(1)

    # 2. Tokenizer & Dataset
    print("Initializing Tokenizer...")
    # TODO: Initialize BertTokenizer from 'bert-base-uncased'
    tokenizer = ... 
    
    try:
        train_dataset = SarcasmDataset(train_data, tokenizer)
        valid_dataset = SarcasmDataset(valid_data, tokenizer)
        
        # Create DataLoaders
        batch_size = 8  # Change batch size if needed

        # DO NOT CHANGE THE FOLLOWING LINES
        if int(os.environ.get("GS_TESTING_BATCH_SIZE", "0")) > 0:
            batch_size = int(os.environ["GS_TESTING_BATCH_SIZE"])
        # END OF DO NOT CHANGE

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Error initializing datasets: {e}")
        exit(1)

    # 3. Model
    try:
        model = SarcasmBERT()
    except NotImplementedError:
        print("Error: Implement SarcasmBERT first.")
        exit(1)

    # 4. Training
    # DO NOT CHANGE THE FOLLOWING LINES
    is_testing = os.environ.get("GS_TESTING", "0") == "1"
    checkpoint_path = "checkpoint.pt"
    if is_testing:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
    else:
        print("Starting Training...")
        # END OF DO NOT CHANGE
        try:
            # TODO: Set learning rate and number of epochs
            lr = ...
            epochs = ...
            train_loop(model, train_loader, device, lr, epochs)
        except NotImplementedError:
            print("Error: Implement train_loop.")
            exit(1)

        # DO NOT MODIFY THE FOLLOWING LINES
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")
        # END OF DO NOT CHANGE

    """
        YOUR ADDITIONAL CODE BELOW (DO NOT DELETE THIS COMMENT)
    """