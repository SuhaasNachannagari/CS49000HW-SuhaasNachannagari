import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import json
import os
from collections import Counter
from typing import List, Dict, Tuple, Any, Union
import gensim.models


# --- Utility: Data Loader ---
def get_data(path: str) -> List[Dict[str, Union[str, int]]]:
    """
    Reads a JSONL file and returns a list of dictionaries.
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class TextFeaturizer:
    def __init__(self, corpus: List[str], w2v_path: str = "GoogleNews-vectors-negative300.bin"):
        # The pre-trained Word2Vec model is available at:
        # https://app.box.com/s/tpmoeke56fimcbpcrdm4lvbphsjxzlc9

        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        
        # 1. Build the vocabulary from the corpus
        self.build_vocab(corpus)
        
        # 2. Load Pre-trained Word2Vec
        print(f"Loading Word2Vec from {w2v_path}...")
        try:
            self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                w2v_path, binary=True
            )
            self.emb_dim = self.w2v_model.vector_size
            print("Word2Vec loaded successfully.")
        except FileNotFoundError:
            print(f"Unable to find {w2v_path}.\nPlease ensure the file exists, raise a ticket with course staff if needed.")
        
    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenizer, you may also use nltk for tokenization
        return re.findall(r'\w+', text.lower())

    def build_vocab(self, corpus: List[str]) -> None:
        """
        Constructs the vocabulary from the corpus.
        """
        # Reserve 0 for <UNK>
        self.word_to_idx['<UNK>'] = 0
        self.idx_to_word[0] = '<UNK>'

        all_tokens = set()
        for text in corpus:
            tokens = self._tokenize(text)
            all_tokens.update(tokens)
        
        # Sort to ensure deterministic order
        sorted_tokens = sorted(list(all_tokens))
        
        for i, word in enumerate(sorted_tokens, start=1):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word

    def to_one_hot(self, text: str) -> np.ndarray:
        """
        Convert text to a Binary Vector. Shape: (actual_vocab_size,)
        """
        tokens = self._tokenize(text)
        vocab_size = len(self.word_to_idx)
        vec = np.zeros(vocab_size)
        
        for token in tokens:
            idx = self.word_to_idx.get(token, 0) # Default to 0 (<UNK>)
            vec[idx] = 1
        return vec

    def to_bow(self, text: str) -> np.ndarray:
        """
        Convert text to Count Vector. Shape: (actual_vocab_size,)
        """
        tokens = self._tokenize(text)
        vocab_size = len(self.word_to_idx)
        vec = np.zeros(vocab_size)
        
        for token in tokens:
            idx = self.word_to_idx.get(token, 0) # Default to 0 (<UNK>)
            vec[idx] += 1
        return vec

    def to_word2vec(self, text: str) -> np.ndarray:
        """
        Convert text to a vector by Averaging the word embeddings.
        Shape: (emb_dim,)
        """
        tokens = self._tokenize(text)
        vectors = []
        for token in tokens:
            if token in self.w2v_model:
                vectors.append(self.w2v_model[token])
        
        if not vectors:
            return np.zeros(self.emb_dim)
        
        return np.mean(vectors, axis=0)


class SarcasmMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int) -> None:
        super(SarcasmMLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_loop(
    model: nn.Module, 
    X: np.ndarray, 
    y: np.ndarray, 
    lr: float, 
    epochs: int
) -> Tuple[List[float], List[float]]:
    """
    Returns: loss_history, acc_history
    """
    x_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_history: List[float] = []
    acc_history: List[float] = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 1. Forward pass
        logits = model(x_tensor)
        
        # 2. Calculate loss
        loss = criterion(logits, y_tensor)
        
        # 3. Backward pass & Optimizer step
        loss.backward()
        optimizer.step()
        
        # 4. Calculate Accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct = (preds == y_tensor).sum().item()
            accuracy = correct / y_tensor.size(0)
        
        loss_history.append(loss.item())
        acc_history.append(accuracy)
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%")
    
    return loss_history, acc_history


if __name__ == "__main__":
    print("--- Starting Assignment Execution ---")
    
    # 1. Load Data
    try:
        print("Loading data...")
        # Ensure these files exist in your directory
        base_path = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(base_path, 'train.jsonl')
        valid_path = os.path.join(base_path, 'valid.jsonl')
        
        train_data = get_data(train_path)
        valid_data = get_data(valid_path)
        
        train_corpus = [str(d['headline']) for d in train_data]
        train_labels = [int(d['is_sarcastic']) for d in train_data]
        print(f"Loaded {len(train_data)} training samples.")      
    except NotImplementedError:
        print("Error: You must implement get_data first.")
        exit(1)
    except FileNotFoundError:
        print("Error: Data files not found.")
        exit(1)

    # 2. Setup Features
    try:
        # Note: DO NOT change the w2v_path. The .bin file must be placed at the top-level directory.
        featurizer = TextFeaturizer(train_corpus, w2v_path="GoogleNews-vectors-negative300.bin")
        
        # --- SELECT FEATURE MODE HERE ---
        feature_mode = featurizer.to_bow  # Change this to to_one_hot or to_word2vec as needed
        
        print("Vectorizing corpus...")
        x_train = np.array([feature_mode(text) for text in train_corpus])
        y_train = np.array(train_labels)
        
    except NotImplementedError as e:
        print(f"\nError: {e}")
        exit(1)

    # 3. Initialize Model
    input_dim = x_train.shape[1]
    hidden_dims: List[int] = [128, 64]
    output_dim: int = 2 # Sarcastic vs Not Sarcastic
    try:
        model = SarcasmMLP(input_dim, hidden_dims, output_dim)
    except NotImplementedError as e:
        print(f"\nError: {e}")
        exit(1)

    # 4. Train
    print("\n--- Training Start ---")
    learning_rate: float = 0.001
    num_epochs: int = 10
    try:
        losses, accs = train_loop(model, x_train, y_train, lr=learning_rate, epochs=num_epochs)
        print(f"Final Training Loss: {losses[-1]:.4f}")
        print(f"Final Training Accuracy: {accs[-1]*100:.2f}%")
    except NotImplementedError as e:
        print(f"\nError: {e}")
        exit(1)

    # 5. Prediction (DO NOT MODIFY)
    print("\n--- Generating Predictions ---")
    try:
        valid_corpus = [str(d['headline']) for d in valid_data]
        x_valid = np.array([feature_mode(text) for text in valid_corpus])
        x_valid_tensor = torch.FloatTensor(x_valid)
        
        model.eval()
        with torch.no_grad():
            logits = model(x_valid_tensor)
            predictions = torch.argmax(logits, dim=1).numpy()

        output_path = os.path.join(base_path, "prediction.jsonl")
        with open(output_path, "w") as f:
            for i, pred in enumerate(predictions):
                record = {
                    "headline": valid_data[i]['headline'],
                    "prediction": int(pred)
                }
                f.write(json.dumps(record) + "\n")
        print(f"Predictions saved to {output_path}")
    except Exception as e:
        print(f"Error during prediction generation: {e}")

    print("\n--- Execution Complete ---")
