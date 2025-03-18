# Summary report  for [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)

<!-- # Building GPT from Scratch: A Summary Report

This report summarizes the techniques and tutorial code presented in Andrej Karpathy's YouTube video "Let's build GPT: from scratch, in code, spelled out." and its companion Colab notebook "gpt-dev.ipynb", detailing the process of creating a Generative Pre-trained Transformer (GPT) model from the ground up. -->

## 1. Introduction to Building GPT from Scratch

The video provides a step-by-step guide to understanding and implementing the core components of a GPT model, starting with basic language modeling concepts and progressively building towards a multi-layered Transformer network. The goal is to demystify the inner workings of large language models like ChatGPT by constructing a similar, albeit much smaller, model trained on the Tiny Shakespeare dataset.

## 2. Tokenization

The first crucial step in processing text data for a language model is **tokenization**, which involves converting raw text into a sequence of integers based on a defined vocabulary. The tutorial explores two main approaches:

*   **Character-Level Tokenization**: This simple method treats each individual character as a token. The Colab notebook demonstrates this by:
    *   Reading the `input.txt` file containing the Tiny Shakespeare dataset.
    *   Identifying all unique characters in the text to form the vocabulary.
    *   Creating encoder and decoder mappings (dictionaries) to translate between characters and their corresponding integer IDs.
    *   Encoding the entire text dataset into a PyTorch tensor of integers.

    ```python
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    ```

*   **Subword Tokenization**: While the tutorial focuses on character-level tokenization for simplicity, it acknowledges that in practice, more sophisticated techniques like SentencePiece (used by Google) and Byte Pair Encoding (BPE, used by OpenAI's `tiktoken`) are common. These methods break words into smaller subword units, allowing for a more efficient representation of the vocabulary and handling of unseen words. The video briefly mentions GPT uses a byte pair encode tokenizer.

## 3. Building the GPT Model

The process of building the GPT model involves several key architectural components:

### 3.1. Data Preparation

Before training, the encoded text data is split into training and validation sets. The tutorial also introduces the concept of creating **batches** of data, where small, random chunks of the training data with a fixed `block_size` (maximum context length) are sampled. Each chunk contains multiple input-target pairs, where the model learns to predict the next token given a sequence of preceding tokens.

```python
block_size = 8 # maximum context length for predictions
batch_size = 4 # how many independent sequences will we process in parallel?

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
```

### 3.2. Bigram Language Model

The tutorial starts with a simple **Bigram Language Model**, which predicts the next character based solely on the current character. This model uses an embedding table to directly map each token to logits (scores) for the next token.

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C)
        # ... (loss calculation) ...
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # ... (generation logic) ...
        return idx
```

### 3.3. Self-Attention

The core innovation of the Transformer architecture is the **self-attention** mechanism. This allows the model to weigh the importance of different tokens in the input sequence when making predictions for each position. The tutorial explains and implements a single head of self-attention:

*   Each token produces three vectors: **query**, **key**, and **value**.
*   The **attention scores** (affinities) between tokens are calculated by taking the dot product of their query and key vectors.
*   A **triangular mask** is applied to the attention scores to ensure that a token can only attend to preceding tokens (including itself), maintaining the autoregressive nature for language modeling. This mask is implemented using `torch.tril` and masking with negative infinity before applying the softmax function.
*   The attention scores are then normalized using **softmax** to obtain weights.
*   The **output** of the self-attention head is a weighted sum of the value vectors, where the weights are the attention scores.
*   **Scaled attention** is introduced, where the attention scores are scaled by $1/\sqrt{head\_size}$ before the softmax to prevent the scores from becoming too large and the softmax from becoming too peaky, especially at initialization.

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
```

### 3.4. Multi-Head Attention

To capture different types of relationships between tokens, the tutorial introduces **Multi-Head Attention**. This involves running multiple independent self-attention heads in parallel, each with its own set of query, key, and value projections. The outputs of these heads are then concatenated and linearly projected back to the original embedding dimension.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

### 3.5. Feedforward Network

After the attention mechanism, each token's representation is passed through a **Feedforward Network**. This is typically a simple multi-layer perceptron (MLP) with a non-linear activation function (ReLU or GELU). This allows each token to process the information it has gathered from other tokens through attention.

```python
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

### 3.6. Transformer Block

The self-attention and feedforward layers are combined into a **Transformer Block**. The tutorial also incorporates two crucial techniques for training deep networks:

*   **Residual Connections (Skip Connections)**: The input to each sub-layer (self-attention or feedforward) is added to the output of that sub-layer. This helps to mitigate the vanishing gradient problem and allows for easier optimization of deeper networks.
*   **Layer Normalization**: Layer normalization is applied before each sub-layer. Unlike batch normalization, which normalizes across the batch dimension, layer normalization normalizes across the features of each individual token.

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

### 3.7. The Full GPT Model

The complete GPT model consists of multiple stacked Transformer blocks. It also includes:

*   **Token Embedding Table**: Maps input token IDs to dense embedding vectors.
*   **Positional Embedding Table**: Adds information about the position of each token in the sequence.
*   **Final Layer Normalization**: Applied to the output of the last Transformer block.
*   **Language Modeling Head**: A linear layer that maps the final token representations back to logits over the vocabulary, predicting the next token.

```python
class BigramLanguageModel(nn.Module): # Renamed to reflect Transformer architecture
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # ... (loss calculation) ...
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # ... (generation logic with context cropping) ...
        return idx
```

### 3.8. Training the Model

The tutorial demonstrates a standard training loop using the AdamW optimizer. It includes:

*   Sampling batches of training data.
*   Performing the forward pass to get logits and calculate the loss using cross-entropy.
*   Performing backpropagation to compute gradients.
*   Updating model parameters using the optimizer.
*   Periodically evaluating the loss on the validation set to monitor overfitting.

### 3.9. Generating Text

The `generate` function takes a starting sequence of tokens and iteratively predicts the next token, sampling from the probability distribution output by the model. The newly predicted token is appended to the sequence, and the process repeats until the desired number of new tokens is generated.

## 4. Scaling Up to a Larger Model

The video briefly discusses scaling the model by increasing the number of parameters (embedding dimension, number of heads, number of layers), the context length (`block_size`), and the size of the training dataset. The example of GPT-3 with 175 billion parameters trained on 300 billion tokens highlights the vast difference in scale between the tutorial model (around 10 million parameters trained on ~300,000 tokens in OpenAI's vocabulary equivalent) and state-of-the-art large language models. Scaling significantly improves the model's ability to learn complex patterns and generate more coherent and relevant text.

## 5. Decoder-Only Transformer Architecture

The implemented model is a **decoder-only Transformer**. This is because it only uses the decoder part of the original Transformer architecture (as described in the "Attention is All You Need" paper), characterized by the **causal (triangular) mask** in the self-attention mechanism, which prevents tokens from attending to future tokens. Decoder-only Transformers are well-suited for generative tasks like language modeling.

In contrast, the original Transformer architecture for machine translation is an **encoder-decoder** model. The encoder processes the input sequence (e.g., French sentence), and the decoder generates the output sequence (e.g., English translation) conditioned on the encoder's output through a **cross-attention** mechanism. Since the tutorial focuses on unconditional text generation, the encoder part and cross-attention are not implemented.

## 6. Relationship to Larger Models (like ChatGPT)

While the fundamental architecture of the tutorial model is similar to that of larger models like ChatGPT (which also uses a decoder-only Transformer architecture), there are significant differences:

*   **Scale**: ChatGPT has vastly more parameters and is trained on much larger datasets.
*   **Tokenization**: ChatGPT uses a more efficient subword tokenization method.
*   **Fine-tuning**: ChatGPT undergoes extensive **fine-tuning** after the initial pre-training to align it with human instructions and preferences. This involves techniques like supervised fine-tuning on question-answer pairs and reinforcement learning from human feedback (RLHF) using a reward model. The tutorial primarily covers the **pre-training** stage of a language model.

## 7. NanogGPT

The video mentions a GitHub repository called **nanogpt**. This repository contains a more polished and feature-complete implementation of a GPT model, including training scripts with options for GPU usage, saving and loading checkpoints, learning rate scheduling, and distributed training. The `model.py` file in nanogpt contains a modular implementation of the Transformer blocks, multi-head attention, and the GPT model, which is architecturally similar to the code developed in the tutorial but with some optimizations and more detailed implementation (e.g., handling multiple attention heads within a single `CasualSelfAttention` module).

## 8. Conclusion

This tutorial provides a valuable hands-on experience in understanding the fundamental principles behind large language models. By building a GPT model from scratch, the video and accompanying materials demystify the complex architecture and offer a solid foundation for further exploration in the field of natural language processing and deep learning.
