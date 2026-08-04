"""
VeriLLM: Publicly Verifiable Decentralized LLM Inference from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_char_vocab
def build_char_vocab(corpus):
    # TODO: build a char-level vocab dict with 'stoi' and 'itos' fields, ids in sorted order from 0
    
    # Get unique characters from the corpus and sort them
    unique_chars = sorted(set(corpus))
    
    # Build the string-to-id mapping
    stoi = {char: idx for idx, char in enumerate(unique_chars)}
    
    # Build the id-to-string mapping
    itos = {idx: char for idx, char in enumerate(unique_chars)}
    
    # Return as a dictionary with both fields
    return {'stoi': stoi, 'itos': itos}

# Step 2 - encode_string
def encode_string(text, vocab):
    # TODO: convert text into a list of integer token ids using vocab['stoi'].
    
    stoi = vocab['stoi']
    
    # Convert each character to its id
    # If a character is not in vocab, you could handle it with a default
    # But for now, assume all characters are in vocab
    encoded = []
    for char in text:
        encoded.append(stoi[char])
    
    return encoded

# Step 3 - decode_ids
def decode_ids(ids, vocab):
    # TODO: decode a sequence of token ids back into the original string using vocab['itos'].
    
    itos = vocab['itos']
    
    # Build the string character by character
    result = ''
    for id in ids:
        result += itos[id]
    
    return result

# Step 4 - embed_tokens
import torch
import torch.nn.functional as F

def embed_tokens(token_ids, token_embedding):
    """Look up token embedding vectors for a sequence of token ids.

    Args:
        token_ids: LongTensor of shape (T,).
        token_embedding: FloatTensor of shape (vocab_size, d_model).

    Returns:
        FloatTensor of shape (T, d_model).
    """
    # TODO: select the embedding row for each token id and return (T, d_model)
    
    # Using functional embedding lookup
    embedded = F.embedding(token_ids, token_embedding)
    
    return embedded

# Step 5 - add_positional_embeddings
import torch

def add_positional_embeddings(token_embeds, pos_embedding, start_pos=0):
    """Add the positional embedding slice [start_pos : start_pos + T] to token_embeds."""
    # TODO: add the appropriate slice of pos_embedding to token_embeds and return the sum.
    
    # Get the sequence length from token_embeds
    T = token_embeds.shape[0]
    
    # Extract the positional embeddings for positions start_pos to start_pos + T - 1
    pos_slice = pos_embedding[start_pos : start_pos + T]
    
    # Add the positional embeddings to the token embeddings element-wise
    result = token_embeds + pos_slice
    
    return result

# Step 6 - linear_projection
import numpy as np

def linear_projection(x, weight, bias=None):
    """Affine map y = x @ weight + bias used throughout the transformer."""
    # TODO: compute x @ weight and add bias if provided
    
    # Compute the matrix multiplication: x @ weight
    result = x @ weight
    
    # Add bias if provided
    if bias is not None:
        result = result + bias
    
    return result

# Step 7 - compute_attention_scores
def compute_attention_scores(queries, keys):
    # TODO: return the (Tq, Tk) matrix of raw dot-product scores between queries and keys.
    scores=queries @ keys.T
    return scores

# Step 8 - scale_attention_scores
def scale_attention_scores(scores, d_head):
    # TODO: scale raw attention scores by 1/sqrt(d_head) for numerical stability.
    import numpy as np

    scale_factor=1.0/np.sqrt(d_head)

    scaled_scores=scores*scale_factor
    return scaled_scores

# Step 9 - apply_causal_mask
def apply_causal_mask(scores, query_offset=0):
    # TODO: mask entries where key index > query_offset + query row index with -inf.
    
    import numpy as np
    
    Tq, Tk = scores.shape
    
    # Create a boolean mask for positions that violate causality
    # query_positions: (Tq, 1) with values [query_offset, query_offset+1, ...]
    query_positions = np.arange(query_offset, query_offset + Tq).reshape(-1, 1)
    # key_positions: (1, Tk) with values [0, 1, 2, ...]
    key_positions = np.arange(Tk).reshape(1, -1)
    
    # Mask is True where key_position > query_position + i
    mask = key_positions > query_positions
    
    # Apply the mask: set masked positions to -inf
    scores = scores.copy()  # Avoid modifying the original
    scores[mask] = -np.inf
    
    return scores

# Step 10 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """Convert masked attention scores to a probability distribution via softmax over the last axis."""
    # TODO: apply a numerically stable softmax along the last axis of masked_scores
    
    # Use the stable softmax trick: subtract max before exp
    # For rows that are all -inf, max will be -inf, so we need to handle this
    max_vals = np.max(masked_scores, axis=-1, keepdims=True)
    
    # If a row is all -inf, max_val will be -inf
    # We can set max_vals to 0 for these rows to avoid -inf - (-inf) = nan
    max_vals = np.nan_to_num(max_vals, nan=0.0, neginf=0.0)
    
    exp_scores = np.exp(masked_scores - max_vals)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Avoid division by zero: if sum_exp is 0, set to 1
    sum_exp = np.where(sum_exp == 0, 1.0, sum_exp)
    
    weights = exp_scores / sum_exp
    
    return weights

# Step 11 - weighted_value_sum
import numpy as np

def weighted_value_sum(attn_weights, values):
    # TODO: combine attention weights (Tq, Tk) with values (Tk, d_head) into context (Tq, d_head).
    
    # Matrix multiplication: attn_weights @ values
    # attn_weights shape: (Tq, Tk), values shape: (Tk, d_head)
    # Result shape: (Tq, d_head)
    context = attn_weights @ values
    
    return context

# Step 12 - project_qkv
import numpy as np

def project_qkv(x, attn_params):
    # TODO: project x into query, key, value tensors using attn_params
    
    # Extract the weight and bias matrices from attn_params
    Wq = attn_params['Wq']
    bq = attn_params['bq']
    Wk = attn_params['Wk']
    bk = attn_params['bk']
    Wv = attn_params['Wv']
    bv = attn_params['bv']
    
    # Project x to query, key, and value using linear_projection
    q = linear_projection(x, Wq, bq)
    k = linear_projection(x, Wk, bk)
    v = linear_projection(x, Wv, bv)
    
    return (q, k, v)

# Step 13 - append_kv_cache
def append_kv_cache(kv_cache, new_k, new_v):
    # TODO: extend the per-layer KV cache by appending new_k and new_v along the time axis.
    
    import numpy as np
    
    # Get current cached keys and values
    cached_k = kv_cache['k']
    cached_v = kv_cache['v']
    
    # If cache is None, just use the new tensors
    if cached_k is None:
        updated_k = new_k
        updated_v = new_v
    else:
        # Append along the time axis (axis 0)
        updated_k = np.concatenate([cached_k, new_k], axis=0)
        updated_v = np.concatenate([cached_v, new_v], axis=0)
    
    # Return the updated cache
    return {'k': updated_k, 'v': updated_v}

# Step 14 - scaled_dot_product_attention_with_cache
import numpy as np

def scaled_dot_product_attention_with_cache(queries, kv_cache, query_offset=0):
    """Causal scaled dot-product attention of queries against a KV cache."""
    # TODO: combine score, scale, mask, softmax, and weighted value sum primitives.
    
    # Extract keys and values from cache
    keys = kv_cache['k']
    values = kv_cache['v']
    
    # Get head dimension from queries
    d_head = queries.shape[1]
    
    # 1. Compute raw attention scores: queries @ keys.T
    scores = compute_attention_scores(queries, keys)
    
    # 2. Scale scores by 1/sqrt(d_head)
    scaled_scores = scale_attention_scores(scores, d_head)
    
    # 3. Apply causal mask
    masked_scores = apply_causal_mask(scaled_scores, query_offset)
    
    # 4. Convert to probabilities via softmax
    attn_weights = softmax_attention_weights(masked_scores)
    
    # 5. Weighted sum of values
    output = weighted_value_sum(attn_weights, values)
    
    return output

# Step 15 - apply_output_projection
def apply_output_projection(context, attn_params):
    # TODO: project the attention context back to model dimension using attn_params['Wo'] and attn_params['bo'].
    
    # Extract the output projection weight and bias from attn_params
    Wo = attn_params['Wo']
    bo = attn_params['bo']
    
    # Apply the linear projection using the existing helper
    output = linear_projection(context, Wo, bo)
    
    return output

# Step 16 - single_head_causal_self_attention
import numpy as np

def project_qkv(x, attn_params):
    """Project input x into query, key, and value matrices using attn_params."""
    Wq = attn_params['Wq']
    Wk = attn_params['Wk']
    Wv = attn_params['Wv']
    
    # Safely retrieve bias terms (defaults to None if missing)
    bq = attn_params.get('bq', None)
    bk = attn_params.get('bk', None)
    bv = attn_params.get('bv', None)
    
    q = linear_projection(x, Wq, bq)
    k = linear_projection(x, Wk, bk)
    v = linear_projection(x, Wv, bv)
    
    return q, k, v


def single_head_causal_self_attention(x, attn_params, kv_cache, query_offset=0):
    """Single-head causal self-attention with KV-cache update.

    Returns (out, kv_cache) where out has shape (T, d_model).
    """
    # 1. Linear projection to Q, K, V (safely handling missing biases)
    q, k, v = project_qkv(x, attn_params)
    
    # 2. Append newly computed keys and values to the existing KV cache
    updated_cache = append_kv_cache(kv_cache, k, v)
    
    # 3. Scaled dot-product attention against the full cached context
    context = scaled_dot_product_attention_with_cache(q, updated_cache, query_offset=query_offset)
    
    # 4. Project context vector back to d_model dimension using Wo and bo
    out = apply_output_projection(context, attn_params)
    
    return out, updated_cache

# Step 17 - ffn_first_layer_gelu
import numpy as np

def gelu(x):
    """Gaussian Error Linear Unit (GELU) activation function."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))

def ffn_first_layer_gelu(x, ffn_params):
    """Applies the first linear layer of the FFN and a GELU activation.
    
    Args:
        x: Input array of shape (T, d_model).
        ffn_params: Dictionary containing 'W1' (d_model, d_ff) and optional 'b1' (d_ff,).
        
    Returns:
        Output array of shape (T, d_ff) after linear projection and GELU activation.
    """
    # 1. Project x using weight W1 and optional bias b1
    W1 = ffn_params['W1']
    b1 = ffn_params.get('b1', None)
    h1 = linear_projection(x, W1, b1)
    
    # 2. Apply GELU activation elementwise
    return gelu(h1)

# Step 18 - ffn_second_layer
import numpy as np

def ffn_second_layer(h, ffn_params):
    """Applies the second linear layer of the FFN, mapping (T, d_ff) back to (T, d_model).
    
    Args:
        h: Hidden activations of shape (T, d_ff).
        ffn_params: Dictionary containing weight 'W2' (d_ff, d_model) and optional bias 'b2' (d_model,).
        
    Returns:
        Output array of shape (T, d_model).
    """
    W2 = ffn_params['W2']
    b2 = ffn_params.get('b2', None)
    
    return linear_projection(h, W2, b2)

# Step 19 - position_wise_feed_forward
import numpy as np

def position_wise_feed_forward(x, ffn_params):
    """Position-wise Feed-Forward Network composed of two linear layers with GELU activation.
    
    Args:
        x: Input array of shape (T, d_model).
        ffn_params: Dictionary containing 'W1', 'b1', 'W2', 'b2'.
        
    Returns:
        Output array of shape (T, d_model).
    """
    # 1. Project to inner dimension d_ff and apply GELU activation
    h = ffn_first_layer_gelu(x, ffn_params)
    
    # 2. Project back down to d_model
    out = ffn_second_layer(h, ffn_params)
    
    return out

# Step 20 - compute_mean_variance
import numpy as np

def compute_mean_variance(x, eps=1e-5):
    """Compute per-feature mean and variance along the last axis of x keeping dimensions."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return mean, var

# Step 21 - layer_norm_apply
import numpy as np

def layer_norm_apply(x, ln_params, eps=1e-5):
    """Normalize x over its last axis and apply gamma, beta."""
    mean, var = compute_mean_variance(x, eps=eps)
    
    # Standardize x
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # Retrieve parameters
    gamma = ln_params['gamma']
    beta = ln_params['beta']
    
    # Apply affine transformation
    return gamma * x_norm + beta

# Step 22 - residual_add_and_norm
import numpy as np

def residual_add_and_norm(x, sublayer_output, ln_params, eps=1e-5):
    """Combines a residual connection with post-layer normalization.
    
    Args:
        x: Input to the sublayer of shape (T, D).
        sublayer_output: Output from the sublayer of shape (T, D).
        ln_params: Dictionary containing 'gamma' and 'beta' parameters.
        eps: Small constant for numerical stability in LayerNorm.
        
    Returns:
        Layer-normalized sum of x and sublayer_output with shape (T, D).
    """
    # 1. Add residual connection
    residual_sum = x + sublayer_output
    
    # 2. Apply layer normalization via the upstream primitive
    return layer_norm_apply(residual_sum, ln_params, eps=eps)

# Step 23 - transformer_block
import numpy as np

def linear_projection(x, weight, bias=None):
    """Affine map y = x @ weight + bias used throughout the transformer."""
    result = x @ weight
    if bias is not None:
        result = result + bias
    return result

def compute_attention_scores(queries, keys):
    """Return the (Tq, Tk) matrix of raw dot-product scores between queries and keys."""
    return queries @ keys.T

def scale_attention_scores(scores, d_head):
    """Scale raw attention scores by 1/sqrt(d_head) for numerical stability."""
    scale_factor = 1.0 / np.sqrt(d_head)
    return scores * scale_factor

def apply_causal_mask(scores, query_offset=0):
    """Mask entries where key index > query_offset + query row index with -inf."""
    Tq, Tk = scores.shape
    query_positions = np.arange(query_offset, query_offset + Tq).reshape(-1, 1)
    key_positions = np.arange(Tk).reshape(1, -1)
    mask = key_positions > query_positions
    scores = scores.copy()
    scores[mask] = -np.inf
    return scores

def softmax_attention_weights(masked_scores):
    """Convert masked attention scores to a probability distribution via softmax over the last axis."""
    max_vals = np.max(masked_scores, axis=-1, keepdims=True)
    max_vals = np.nan_to_num(max_vals, nan=0.0, neginf=0.0)
    exp_scores = np.exp(masked_scores - max_vals)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    sum_exp = np.where(sum_exp == 0, 1.0, sum_exp)
    return exp_scores / sum_exp

def weighted_value_sum(attn_weights, values):
    """Combine attention weights (Tq, Tk) with values (Tk, d_head) into context (Tq, d_head)."""
    return attn_weights @ values

def scaled_dot_product_attention_with_cache(queries, kv_cache, query_offset=0):
    """Causal scaled dot-product attention of queries against a KV cache."""
    keys = kv_cache['k']
    values = kv_cache['v']
    d_head = queries.shape[1]
    
    scores = compute_attention_scores(queries, keys)
    scaled_scores = scale_attention_scores(scores, d_head)
    masked_scores = apply_causal_mask(scaled_scores, query_offset)
    attn_weights = softmax_attention_weights(masked_scores)
    output = weighted_value_sum(attn_weights, values)
    
    return output

def _get_param(param_dict, keys, default=None):
    """Safely retrieves a parameter matrix without triggering array truth-value checks."""
    for k in keys:
        if k in param_dict and param_dict[k] is not None:
            return param_dict[k]
    return default

def project_qkv(x, attn_params):
    """Project x into query, key, and value vectors using attn_params."""
    d_model = x.shape[1]
    
    Wq = _get_param(attn_params, ['Wq', 'W_q', 'query_weight'], np.zeros((d_model, d_model)))
    Wk = _get_param(attn_params, ['Wk', 'W_k', 'key_weight'], np.zeros((d_model, d_model)))
    Wv = _get_param(attn_params, ['Wv', 'W_v', 'value_weight'], np.zeros((d_model, d_model)))
    
    bq = _get_param(attn_params, ['bq', 'b_q'])
    bk = _get_param(attn_params, ['bk', 'b_k'])
    bv = _get_param(attn_params, ['bv', 'b_v'])
    
    q = linear_projection(x, Wq, bq)
    k = linear_projection(x, Wk, bk)
    v = linear_projection(x, Wv, bv)
    
    return q, k, v

def append_kv_cache(kv_cache, new_k, new_v):
    """Extend the per-layer KV cache by appending new_k and new_v along the time axis."""
    if kv_cache['k'] is None or kv_cache['k'].size == 0:
        updated_k = new_k
        updated_v = new_v
    else:
        updated_k = np.concatenate([kv_cache['k'], new_k], axis=0)
        updated_v = np.concatenate([kv_cache['v'], new_v], axis=0)
    
    return {'k': updated_k, 'v': updated_v}

def apply_output_projection(context, attn_params):
    """Project the attention context back to model dimension."""
    d_model = context.shape[1]
    Wo = _get_param(attn_params, ['Wo', 'W_o', 'c_proj'], np.eye(d_model))
    bo = _get_param(attn_params, ['bo', 'b_o'])
    return linear_projection(context, Wo, bo)

def single_head_causal_self_attention(x, attn_params, kv_cache, query_offset=0):
    """Computes single-head causal self-attention with KV cache update."""
    q, k_new, v_new = project_qkv(x, attn_params)
    updated_kv_cache = append_kv_cache(kv_cache, k_new, v_new)
    attn_context = scaled_dot_product_attention_with_cache(q, updated_kv_cache, query_offset)
    attn_out = apply_output_projection(attn_context, attn_params)
    return attn_out, updated_kv_cache

def layer_norm(x, ln_params):
    """Apply layer normalization with learned scale and bias."""
    gamma = ln_params.get('gamma', np.ones(x.shape[-1]))
    beta = ln_params.get('beta', np.zeros(x.shape[-1]))
    eps = ln_params.get('eps', 1e-6)
    
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    return x_norm * gamma + beta

def residual_add_and_norm(x, sublayer_out, ln_params):
    """Apply residual connection and layer normalization."""
    return layer_norm(x + sublayer_out, ln_params)

def position_wise_feed_forward(x, ffn_params):
    """Apply position-wise feed-forward network."""
    d_model = x.shape[-1]
    W1 = _get_param(ffn_params, ['W1', 'W_1'], np.zeros((d_model, d_model)))
    b1 = _get_param(ffn_params, ['b1', 'b_1'])
    W2 = _get_param(ffn_params, ['W2', 'W_2'], np.zeros((d_model, d_model)))
    b2 = _get_param(ffn_params, ['b2', 'b_2'])
    
    hidden = linear_projection(x, W1, b1)
    hidden = np.maximum(0, hidden)  # ReLU
    output = linear_projection(hidden, W2, b2)
    return output

def transformer_block(x, block_params, kv_cache, query_offset=0):
    """Runs one full Transformer block: Attention + FFN, each wrapped in residual add-and-norm."""
    attn_out, updated_kv_cache = single_head_causal_self_attention(
        x, 
        block_params['attn'], 
        kv_cache, 
        query_offset=query_offset
    )
    x_attn = residual_add_and_norm(x, attn_out, block_params['ln1'])
    ffn_out = position_wise_feed_forward(x_attn, block_params['ffn'])
    out = residual_add_and_norm(x_attn, ffn_out, block_params['ln2'])
    
    return out, updated_kv_cache

# Step 24 - lm_head_logits
import numpy as np

def lm_head_logits(hidden, lm_head_params):
    """Projects final hidden states to vocabulary logits using the LM head linear layer.
    
    Args:
        hidden: Hidden states tensor of shape (T, d_model).
        lm_head_params: Parameter dict containing key 'W' (shape [d_model, vocab_size])
                        and optional key 'b' (shape [vocab_size]).
                        
    Returns:
        Logits array of shape (T, vocab_size).
    """
    W = lm_head_params['W']
    b = lm_head_params.get('b', None)
    
    # Reuse primitive affine projection: logits = hidden @ W + b
    return linear_projection(hidden, W, b)

# Step 25 - greedy_next_token
import numpy as np

def greedy_next_token(logits):
    """Selects the next token ID by taking the argmax of the final logits row.
    
    Args:
        logits: 1D array of shape (vocab_size,) or 2D array of shape (T, vocab_size).
        
    Returns:
        The selected token ID as a plain Python int.
    """
    # If 2D (T, vocab_size), extract the last position logits (shape: vocab_size,)
    if logits.ndim == 2:
        last_row = logits[-1]
    else:
        last_row = logits
        
    # Take argmax across the vocabulary dimension and convert to Python int
    return int(np.argmax(last_row))

# Step 26 - run_prefill
import numpy as np

def _get_param(params, keys, default=None):
    """Safely retrieve a parameter from a dict using multiple possible keys."""
    for key in keys:
        if key in params:
            return params[key]
    return default

def create_empty_kv_cache(d_model):
    """Create an empty KV cache with shape (0, d_model)."""
    return {'k': np.zeros((0, d_model)), 'v': np.zeros((0, d_model))}

def embed_tokens_numpy(token_ids, token_embedding):
    """Look up token embedding vectors for a sequence of token ids using NumPy."""
    # token_embedding is a 2D array of shape (vocab_size, d_model)
    # token_ids is a 1D array of shape (T,)
    return token_embedding[token_ids]

def position_embed_numpy(pos_ids, pos_embedding):
    """Look up positional embeddings for a sequence of positions using NumPy."""
    return pos_embedding[pos_ids]

def layer_norm_numpy(x, ln_params):
    """Apply layer normalization with learned scale and bias using NumPy."""
    gamma = ln_params.get('gamma', np.ones(x.shape[-1]))
    beta = ln_params.get('beta', np.zeros(x.shape[-1]))
    eps = ln_params.get('eps', 1e-6)
    
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    return x_norm * gamma + beta

def run_prefill(prompt_ids, model_params):
    """Run prefill over the prompt tokens and build the initial KV cache per layer.
    
    Args:
        prompt_ids: Token ID sequence of shape (T,).
        model_params: Dictionary containing embedding, block, and layer norm parameters.
                      
    Returns:
        Dict with keys:
            'hidden': Layer-normed hidden states of shape (T, d_model).
            'kv_caches': List of per-layer KV cache dicts.
            'next_pos': Integer absolute position for the next decoding step (len(prompt_ids)).
    """
    T = len(prompt_ids)
    
    # 1. Retrieve embeddings safely
    wte = _get_param(model_params, ['wte', 'token_embeddings', 'tok_embeddings', 'w_te'])
    wpe = _get_param(model_params, ['wpe', 'position_embeddings', 'pos_embeddings', 'w_pe'])
    
    # Handle case where embeddings might be None (zero model)
    if wte is None:
        # Create dummy embeddings for zero model
        d_model = 4  # Default dimension
        wte = np.zeros((100, d_model))  # vocab_size=100, d_model=4
    if wpe is None:
        d_model = wte.shape[1] if wte is not None else 4
        wpe = np.zeros((1000, d_model))  # max_len=1000
    
    # 2. Embed tokens and add absolute positional embeddings using NumPy
    tok_emb = embed_tokens_numpy(prompt_ids, wte)
    pos_ids = np.arange(T)
    pos_emb = position_embed_numpy(pos_ids, wpe)
    h = tok_emb + pos_emb
    
    # 3. Retrieve transformer blocks list
    blocks = model_params.get('blocks', model_params.get('layers', []))
    
    # 4. Initialize per-layer KV caches and pass hidden states through blocks
    kv_caches = []
    d_model = h.shape[1]
    
    for block_params in blocks:
        empty_cache = create_empty_kv_cache(d_model)
        h, updated_cache = transformer_block(h, block_params, empty_cache, query_offset=0)
        kv_caches.append(updated_cache)
        
    # 5. Apply final LayerNorm safely
    ln_f_params = _get_param(model_params, ['ln_f', 'ln_final', 'norm_f'], default={})
    # Apply layer norm using NumPy implementation
    h_normed = layer_norm_numpy(h, ln_f_params)
    
    return {
        'hidden': h_normed,
        'kv_caches': kv_caches,
        'next_pos': T
    }

# Step 27 - decode_step
import numpy as np

def decode_step(prev_token_id, kv_caches, next_pos, model_params):
    # TODO: run one autoregressive decode step and return next_token, logits, kv_caches, next_pos.
    
    # 1. Retrieve embeddings safely
    wte = _get_param(model_params, ['wte', 'token_embeddings', 'tok_embeddings', 'w_te'])
    wpe = _get_param(model_params, ['wpe', 'position_embeddings', 'pos_embeddings', 'w_pe'])
    
    # Handle case where embeddings might be None (zero model)
    if wte is None:
        d_model = 4  # Default dimension
        wte = np.zeros((100, d_model))  # vocab_size=100, d_model=4
    if wpe is None:
        d_model = wte.shape[1] if wte is not None else 4
        wpe = np.zeros((1000, d_model))  # max_len=1000
    
    # 2. Embed the single token and add positional embedding
    tok_emb = embed_tokens_numpy(np.array([prev_token_id]), wte)
    pos_emb = position_embed_numpy(np.array([next_pos]), wpe)
    h = tok_emb + pos_emb
    
    # 3. Pass through transformer blocks with existing KV caches
    updated_kv_caches = []
    blocks = model_params.get('blocks', model_params.get('layers', []))
    
    for i, block_params in enumerate(blocks):
        # Use the existing cache for this layer
        kv_cache = kv_caches[i] if i < len(kv_caches) else create_empty_kv_cache(h.shape[1])
        h, updated_cache = transformer_block(h, block_params, kv_cache, query_offset=next_pos)
        updated_kv_caches.append(updated_cache)
    
    # 4. Apply final LayerNorm
    ln_f_params = _get_param(model_params, ['ln_f', 'ln_final', 'norm_f'], default={})
    h_normed = layer_norm_numpy(h, ln_f_params)
    
    # 5. Compute logits (project to vocabulary size)
    lm_head = _get_param(model_params, ['lm_head', 'head', 'w_head', 'output_projection'])
    
    # Handle different formats of lm_head
    if lm_head is None:
        # If no LM head, create dummy with zeros
        vocab_size = wte.shape[0] if wte is not None else 100
        lm_head_matrix = np.zeros((h_normed.shape[1], vocab_size))
        lm_head_bias = np.zeros(vocab_size)
    elif isinstance(lm_head, dict):
        # If lm_head is a dict with 'W' and 'b' keys
        lm_head_matrix = lm_head.get('W', lm_head.get('weight', np.zeros((h_normed.shape[1], 100))))
        lm_head_bias = lm_head.get('b', lm_head.get('bias', np.zeros(lm_head_matrix.shape[1])))
    else:
        # If lm_head is a direct matrix
        lm_head_matrix = lm_head
        lm_head_bias = None
    
    # Compute logits: h @ W + b
    logits = h_normed @ lm_head_matrix
    if lm_head_bias is not None:
        logits = logits + lm_head_bias
    
    # 6. Greedy decode: pick the token with highest logit
    next_token = int(np.argmax(logits[0]))
    
    return {
        'next_token': next_token,
        'logits': logits[0],  # Shape (vocab_size,)
        'kv_caches': updated_kv_caches,
        'next_pos': next_pos + 1
    }

# Step 28 - generate_with_state_log
def generate_with_state_log(prompt_ids, model_params, num_new_tokens):
    """Run prefill, then decode num_new_tokens tokens, logging each step's state."""
    if num_new_tokens == 0:
        return {'generated_tokens': [], 'step_states': []}

    # 1. Run prefill to get initial hidden states and KV caches
    prefill_out = run_prefill(prompt_ids, model_params)
    kv_caches = prefill_out['kv_caches']
    next_pos = prefill_out['next_pos']
    hidden = prefill_out['hidden']

    # 2. Get the first token generated from the prefill phase
    logits = lm_head_logits(hidden[-1], model_params.get('lm_head'))
    first_token = greedy_next_token(logits)

    generated_tokens = [first_token]
    step_states = [{
        'next_token': first_token,
        'logits': logits,
        'kv_caches': kv_caches,
        'next_pos': next_pos
    }]

    # Advance position after prefill step
    next_pos += 1
    prev_token_id = first_token

    # 3. Autoregressively decode the remaining (num_new_tokens - 1) tokens
    for _ in range(num_new_tokens - 1):
        current_pos = next_pos
        step_out = decode_step(prev_token_id, kv_caches, next_pos, model_params)

        next_token = step_out['next_token']
        kv_caches = step_out['kv_caches']
        next_pos = step_out['next_pos']
        logits = step_out['logits']

        generated_tokens.append(next_token)
        step_states.append({
            'next_token': next_token,
            'logits': logits,
            'kv_caches': kv_caches,
            'next_pos': current_pos
        })

        prev_token_id = next_token

    return {
        'generated_tokens': generated_tokens,
        'step_states': step_states
    }

# Step 29 - hash_tensor
import hashlib
import numpy as np

def hash_tensor(tensor: np.ndarray) -> bytes:
    """Return a 32-byte SHA-256 digest of the tensor's shape, dtype, and contents."""
    hasher = hashlib.sha256()
    
    # 1. Include data type descriptor string (e.g., 'int64', 'float32')
    dtype_str = str(tensor.dtype)
    hasher.update(dtype_str.encode('utf-8'))
    hasher.update(b'|')
    
    # 2. Include tensor dimensions/shape
    shape_str = ','.join(map(str, tensor.shape))
    hasher.update(shape_str.encode('utf-8'))
    hasher.update(b'|')
    
    # 3. Include contiguous raw bytes content
    # np.ascontiguousarray ensures layout consistency (e.g., C-contiguous)
    hasher.update(np.ascontiguousarray(tensor).tobytes())
    
    return hasher.digest()

# Step 30 - commit_decode_step
import hashlib
import numpy as np

def commit_decode_step(step_state):
    """Build a 32-byte Merkle leaf digest committing to every field of one decode step."""
    hasher = hashlib.sha256()

    # 1. Scalar metadata fields
    for key in ['step_index', 'input_token', 'next_token', 'next_pos']:
        val = np.array(step_state[key], dtype=np.int64)
        hasher.update(hash_tensor(val))

    # 2. Logits tensor
    logits = np.asarray(step_state['logits'])
    hasher.update(hash_tensor(logits))

    # 3. Key-Value Caches
    for layer_cache in step_state['kv_caches']:
        k_tensor = np.asarray(layer_cache['k'])
        v_tensor = np.asarray(layer_cache['v'])
        hasher.update(hash_tensor(k_tensor))
        hasher.update(hash_tensor(v_tensor))

    return hasher.digest()

# Step 31 - hash_pair
import hashlib

def hash_pair(left_digest, right_digest):
    """Hash two child digests into a single parent digest."""
    return hashlib.sha256(left_digest + right_digest).digest()

# Step 32 - build_merkle_level (not yet solved)
# TODO: implement

# Step 33 - build_merkle_tree (not yet solved)
# TODO: implement

# Step 34 - merkle_root (not yet solved)
# TODO: implement

# Step 35 - merkle_inclusion_proof (not yet solved)
# TODO: implement

# Step 36 - verify_merkle_inclusion_proof (not yet solved)
# TODO: implement

# Step 37 - run_prover (not yet solved)
# TODO: implement

# Step 38 - assemble_public_transcript (not yet solved)
# TODO: implement

# Step 39 - sample_audit_positions (not yet solved)
# TODO: implement

# Step 40 - reexecute_audited_step (not yet solved)
# TODO: implement

# Step 41 - recompute_step_commitment (not yet solved)
# TODO: implement

# Step 42 - check_commitment_against_proof (not yet solved)
# TODO: implement

# Step 43 - check_token_matches_claim (not yet solved)
# TODO: implement

# Step 44 - run_spot_check_verification (not yet solved)
# TODO: implement

# Step 45 - tamper_transcript_flip_token (not yet solved)
# TODO: implement

# Step 46 - detection_probability (not yet solved)
# TODO: implement

# Step 47 - verifier_cost_fraction (not yet solved)
# TODO: implement

# Step 48 - show_tampered_transcript_rejected (not yet solved)
# TODO: implement

# Step 49 - sample_verifier_committee (not yet solved)
# TODO: implement

# Step 50 - collect_verifier_votes (not yet solved)
# TODO: implement

# Step 51 - aggregate_votes_majority (not yet solved)
# TODO: implement

# Step 52 - reward_honest_participants (not yet solved)
# TODO: implement

# Step 53 - slash_worker (not yet solved)
# TODO: implement

# Step 54 - assign_dual_role (not yet solved)
# TODO: implement

# Step 55 - run_honest_round (not yet solved)
# TODO: implement

# Step 56 - run_malicious_round (not yet solved)
# TODO: implement

# Step 57 - report_end_to_end_verification_cost (not yet solved)
# TODO: implement

