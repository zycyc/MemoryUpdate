"""Qwen3-Embedding-0.6B: direct text-embedding model (1024 dims).

This implementation uses the dedicated embedding model instead of a base LLM.
Performs attention-mask aware mean pooling and L2-normalization.
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List

class QwenEmbeddings:
    """Qwen3-Embedding-0.6B text embeddings (1024-dim)."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", normalize: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.normalize = normalize
        print(f"Loading {model_name} on {self.device}...")

        # Use the embedding model directly
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Some embedding repos require remote code for custom forward helpers; safe fallback to standard forward
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(self.device)

        hidden = getattr(self.model.config, "hidden_size", None)
        print(f"✅ Embedding model loaded. hidden_size={hidden}")
    
    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: [B, T, H], attention_mask: [B, T]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
        summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-6)                        # [B, 1]
        return summed / counts

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using the Qwen embedding model."""
        if not texts:
            return []

        # Tokenize as a batch for efficiency
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Prefer pooled output if provided, else attention-aware mean pooling
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output  # [B, H]
            else:
                emb = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])  # [B, H]

            if self.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        return emb.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
