# Summary report  for [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

<!-- # Tokenization in Large Language Models: A Summary Report -->

## 1. Introduction to Tokenization in LLMs

**Tokenization is a fundamental pre-processing step in Natural Language Processing (NLP), particularly crucial for Large Language Models (LLMs)**. It involves breaking down raw text into smaller units called **tokens**, which are then used as the basic building blocks for the model. Andrej Karpathy's video, "Let's build the GPT Tokenizer," underscores the significance of understanding tokenization, highlighting it as a common root of various complexities, unexpected behaviors, and even potential issues in LLMs. This report summarizes the techniques discussed, provides relevant code insights, and outlines key concepts related to tokenization in the context of modern LLMs.

## 2. Naive Character-Level Tokenization

In the initial stages of developing LLMs, a simple approach called **character-level tokenization** was employed. As demonstrated in Karpathy's previous video "Let's Build GPT from scratch," this method involves creating a vocabulary of all unique characters present in the training dataset. Each character is then mapped to a unique integer ID (token).

For example, the string "High there" could be tokenized into a sequence of integer representations of 'H', 'i', 'g', 'h', ' ', 't', 'h', 'e', 'r', 'e'. While straightforward, this approach leads to long sequences, especially for languages with extensive character sets, and may not capture semantic relationships effectively.

## 3. Advanced Tokenization with Byte Pair Encoding (BPE)

Modern LLMs predominantly utilize more sophisticated techniques, with **Byte Pair Encoding (BPE)** being a widely adopted algorithm. BPE operates at the level of **character chunks** rather than individual characters and aims to learn a vocabulary of frequently occurring subword units.

### 3.1. The BPE Algorithm

The BPE algorithm starts by treating each individual byte (from the UTF-8 encoding of the text) as a token. It then iteratively performs the following steps:

1.  **Identify the most frequent pair of adjacent tokens in the training data.**
2.  **Merge this pair into a new, single token.**
3.  **Add the new token to the vocabulary.**
4.  **Repeat steps 1-3 until a desired vocabulary size is reached.**

This process allows the tokenizer to learn common word fragments and even whole words as single tokens, leading to shorter and more semantically meaningful token sequences compared to character-level tokenization. The final vocabulary consists of the initial byte tokens and the newly merged tokens.

### 3.2. Byte-Level BPE

The GPT-2 paper popularized the use of **byte-level BPE**, where the algorithm is applied directly to the UTF-8 encoded bytes of the text. This approach has several advantages:

*   It can handle any Unicode character as UTF-8 can represent virtually all characters.
*   It avoids the need for explicit handling of unknown characters, as any sequence of bytes can be tokenized.

### 3.3. Iterative Merging and Vocabulary Creation

The training of a BPE tokenizer involves iteratively finding the most frequent byte pairs or existing token pairs and merging them. Each merge creates a new token and expands the vocabulary. The order of merges is crucial and is stored to enable the encoding of new text. The final vocabulary size is a hyperparameter that influences the tokenizer's performance. Models like GPT-4 use significantly larger vocabularies (around 100,000 tokens) compared to earlier models like GPT-2 (around 50,000 tokens).

## 4. Implementation Details and Code Examples

The video provides Python code snippets illustrating the core steps of the BPE algorithm. Below are some key functions and their functionalities:

### 4.1. Finding the Most Common Pair (`get_stats`)

This function takes a list of integer tokens as input and returns a dictionary where keys are tuples representing consecutive token pairs, and values are their frequencies.

```python
def get_stats(ids):
    counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

### 4.2. Merging Pairs (`merge`)

This function takes a list of tokens, a pair of tokens to be merged, and the new token ID. It iterates through the token list and replaces every occurrence of the specified pair with the new token ID.

```python
def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids
```

### 4.3. Training the Tokenizer

The tokenizer training involves initializing the vocabulary with individual byte values (0-255) and then iteratively applying the `get_stats` and `merge` functions for a predefined number of merges or until a target vocabulary size is reached. The merge operations are recorded in a dictionary (`merges`).

```python
num_merges = 20
ids = list(text.encode("utf-8")) # Initial tokens are bytes
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx
```

### 4.4. Encoding Text to Tokens (`encode`)

The encoding process takes a string, converts it to UTF-8 bytes, and then iteratively applies the learned merges to produce a sequence of tokens. It prioritizes earlier merges (those added to the `merges` dictionary first).

```python
def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # Find the best pair to merge based on the learned merges
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens
```

### 4.5. Decoding Tokens to Text (`decode`)

Decoding involves reversing the tokenization process. A vocabulary mapping from token IDs to their byte representations is created. The token sequence is then converted back to a byte sequence, which is finally decoded into a UTF-8 string. Error handling for invalid UTF-8 byte sequences (using `errors='replace'`) is crucial.

```python
def decode(ids):
    vocab = {i: bytes([i]) for i in range(256)}
    for (p1, p2), idx in merges.items():
        vocab[idx] = vocab[p1] + vocab[p2]

    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text
```

## 5. Complexities and Variations in State-of-the-Art Tokenizers

While the basic BPE algorithm is fundamental, state-of-the-art tokenizers incorporate additional complexities and variations.

### 5.1. GPT-2 Tokenizer

The GPT-2 tokenizer introduces a **regex-based pre-splitting** step before applying BPE. This regex pattern aims to split the text into chunks based on categories like letters, numbers, punctuation, and whitespace. The BPE algorithm is then applied within these chunks, preventing merges across category boundaries. This pre-splitting helps in handling different types of text components consistently. However, the specific regex used in GPT-2 has been noted for some inconsistencies, particularly with Unicode characters and case sensitivity.

### 5.2. GPT-4 Tokenizer

The GPT-4 tokenizer, as used by models like `cl100k_base` in the `tiktoken` library, employs a different regex pattern for pre-splitting. Key changes include **case-insensitive matching** for certain patterns (like contractions) and limitations on merging numbers beyond three digits. Furthermore, GPT-4's tokenizer exhibits improved whitespace handling, merging multiple spaces into single tokens, unlike GPT-2. The vocabulary size of GPT-4's tokenizer is also significantly larger (around 100,000).

### 5.3. Tiktoken Library

**Tiktoken** is OpenAI's official library for tokenization inference. It provides efficient implementations of the tokenizers used in GPT models, allowing users to encode and decode text using pre-trained vocabularies. Importantly, `tiktoken` is primarily for inference and does not include functionality for training new tokenizers from scratch.

### 5.4. Sentencepiece Library

**Sentencepiece** is another widely used tokenization library that supports both **training and inference** of BPE tokenizers, along with other algorithms. A key difference from `tiktoken` is that Sentencepiece directly applies BPE on **Unicode code points** rather than UTF-8 bytes. It also includes features like **byte fallback** for rare code points, encoding them as UTF-8 bytes and then as individual byte tokens. Sentencepiece offers extensive configuration options for training, including normalization rules and handling of rare words and whitespace. It introduces the concept of sentences during training, which can be a notable distinction. The video notes that while powerful, Sentencepiece can be complex to configure and its documentation is not always straightforward.

## 6. Special Tokens

**Special tokens** are added to the tokenizer's vocabulary to represent specific semantic or structural elements. Examples include `<|endoftext|>` to mark the end of a document, and tokens used in chat models to delineate turns and roles in conversations (e.g., `<|im_start|>`, `<|im_end|>`).

When a tokenizer encounters a sequence that matches a special token, it directly maps it to the corresponding special token ID, bypassing the regular BPE merging process. Adding special tokens often requires modifications to the LLM's architecture, such as resizing the embedding layer and the final prediction layer. The `tiktoken` library and `minbpe` allow for handling and registering special tokens.

## 7. Vocabulary Size

The **vocabulary size** is a crucial hyperparameter in tokenization. A larger vocabulary can lead to more efficient encoding of frequent words and shorter token sequences, but it also increases the size of the model's embedding layer and output layer, potentially increasing computational costs. The optimal vocabulary size depends on the size and nature of the training data.

Extending the vocabulary of a pre-trained model by adding new tokens is possible but requires "model surgery," specifically resizing the embedding and prediction layers and initializing the new token embeddings. Techniques like "gist tokens" aim to introduce new tokens for compressing long prompts while keeping most of the model frozen.

## 8. Impact of Tokenization on LLM Behavior

The choice and characteristics of the tokenizer significantly influence the behavior and capabilities of LLMs. The video illustrates several ways tokenization can manifest in LLM performance:

*   **Spelling and String Manipulation:** LLMs might struggle with character-level tasks like spelling or reversing strings if these units are often part of larger tokens.
*   **Non-English Languages:** Tokenizers trained primarily on English data can be less efficient for other languages, resulting in longer token sequences and potentially poorer performance.
*   **Whitespace Handling:** Inconsistent or inefficient handling of whitespace can affect the encoding of code (like Python, where whitespace is semantic) and lead to unexpected behavior.
*   **"Solid Gold Magikarp" Phenomenon:** Rare strings that are overrepresented in the tokenizer's training data but not the LLM's training data can become single tokens with poorly learned embeddings, leading to nonsensical or even harmful model outputs.
*   **Efficiency of Data Formats:** Different data formats like JSON and YAML can have varying token densities, impacting the cost and context length utilization when processing structured data with LLMs.
*   **Trailing Whitespace Warning:** This warning arises because tokenizers often treat whitespace as part of a token, and a trailing space can disrupt the expected token boundaries, leading to suboptimal performance.
*   **Partial Tokens and Unstable Tokens:** Issues arise when the input text ends in a fragment of a token that the model has rarely or never seen in isolation, leading to unpredictable completions.

## 9. Conclusion

Tokenization, despite being often overlooked, is a critical component underpinning the functionality and limitations of LLMs. Understanding the principles of BPE, the nuances of different tokenizer implementations (like those in `tiktoken` and Sentencepiece), and the impact of special tokens and vocabulary size is essential for anyone working with these models. The video emphasizes the "hairy" and "gnarly" nature of tokenization, highlighting its role in various observed LLM behaviors and the importance of not brushing it off. While research into tokenization-free approaches is ongoing, tokenization remains a necessary step with significant implications for LLM performance, efficiency, and even safety.

## 10. Recommendations

Based on the video's insights and discussion, the following recommendations can be made:

*   **Leverage Existing Tokenizers:** If possible, consider reusing the GPT-4 tokenizer and vocabulary via the `tiktoken` library for its efficiency and robustness.
*   **Exercise Caution with Custom Training:** If training a custom vocabulary is necessary, BPE with Sentencepiece is a viable option, but be extremely careful with its numerous settings and ensure thorough understanding to avoid misconfigurations. Consider the algorithm used by `tiktoken` (byte-level BPE with regex pre-splitting) as a potentially superior approach.
*   **Explore `minbpe`:** For a more transparent and understandable implementation of BPE (including training), the `minbpe` repository is a valuable resource. Keep an eye on its development for potential efficiency improvements.
*   **Be Mindful of Special Tokens:** Understand the purpose and handling of special tokens, especially when dealing with user inputs, to avoid potential security vulnerabilities or unexpected behavior.
*   **Optimize Data Formats:** When working with structured data, prefer more token-efficient formats like YAML over JSON to minimize token usage and costs.
*   **Investigate Tokenization Effects:** Be aware that many seemingly model-related issues can trace back to tokenization. Utilize tools like the Tiktokenizer web app to analyze how text is being tokenized.
