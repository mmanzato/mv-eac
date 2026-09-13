"""
NRMS -- Neural News Recommendation with Multi-Head Self-Attention.

Reference: Wu et al. (2019), "Neural News Recommendation with Multi-Head
Self-Attention", EMNLP-IJCNLP 2019.

This is a *user-encoder-focused* reimplementation: the full NRMS news encoder
(a CNN or attention layer over raw article text) is replaced with the
pre-trained, frozen 300-d Word2Vec document embeddings already shipped with
EB-NeRD, and only the user encoder (multi-head self-attention over the
embeddings of a user's history, pooled by additive attention) is trained.
This keeps training tractable on a single machine (CPU, CUDA, or Apple
Silicon MPS) while preserving NRMS's central architectural idea: a user is
represented as an attention-weighted function of their reading history, not
a single fixed embedding.

A candidate article's score is the dot product between its (linearly
projected) embedding and the user vector -- standard NRMS click prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when torch is missing
    _TORCH_AVAILABLE = False
    log.warning("PyTorch is not installed; NRMS cannot be used until it is.")


# ---------------------------------------------------------------------------
# Training dataset: one (history, candidates, labels) triple per positive click
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class ImpressionDataset(Dataset):
        """Builds (history embeddings, candidate embeddings, one-hot label) training triples.

        For every clicked article in an impression, one training example is
        created: the candidate set is the clicked article plus ``neg_ratio``
        negatives sampled from the impression's non-clicked articles (with
        replacement if there are fewer negatives available than requested).
        The clicked article is always placed at candidate position 0, so
        training reduces to a ``neg_ratio + 1``-way categorical cross-entropy
        classification of "which candidate was clicked".
        """

        def __init__(
            self,
            behaviors: pd.DataFrame,
            user_history: dict[int, list[int]],
            embedding_matrix: np.ndarray,
            id2idx: dict[int, int],
            max_history: int,
            neg_ratio: int,
        ) -> None:
            self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            dim = embedding_matrix.shape[1]
            zero_vec = np.zeros(dim, dtype=np.float32)

            for row in behaviors.itertuples(index=False):
                uid = int(row.user_id)
                clicked = set(int(x) for x in row.article_ids_clicked)
                inview = [int(x) for x in row.article_ids_inview]
                positives = [a for a in inview if a in clicked]
                negatives = [a for a in inview if a not in clicked]
                if not positives:
                    continue

                hist = user_history.get(uid, [])[-max_history:]
                hist_vecs = np.zeros((max_history, dim), dtype=np.float32)
                for i, aid in enumerate(hist):
                    idx = id2idx.get(aid)
                    if idx is not None:
                        hist_vecs[i] = embedding_matrix[idx]

                for pos in positives:
                    pos_idx = id2idx.get(pos)
                    if pos_idx is None:
                        continue
                    neg_sample = negatives[:neg_ratio]
                    if negatives and len(neg_sample) < neg_ratio:
                        neg_sample = (negatives * (neg_ratio // len(negatives) + 1))[:neg_ratio]
                    neg_vecs = np.array(
                        [embedding_matrix[id2idx[n]] if n in id2idx else zero_vec for n in neg_sample],
                        dtype=np.float32,
                    )
                    if len(neg_vecs) == 0:
                        continue
                    cand_vecs = np.vstack([embedding_matrix[pos_idx][None], neg_vecs])
                    labels = np.zeros(len(cand_vecs), dtype=np.float32)
                    labels[0] = 1.0
                    self.samples.append((hist_vecs, cand_vecs, labels))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            h, c, y = self.samples[idx]
            return torch.FloatTensor(h), torch.FloatTensor(c), torch.FloatTensor(y)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class AdditiveAttention(nn.Module):
        """Pools a sequence of vectors into one via a learned attention query."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.proj = nn.Linear(dim, dim)
            self.query = nn.Linear(dim, 1, bias=False)

        def forward(self, x: "torch.Tensor", mask: "torch.Tensor | None" = None) -> "torch.Tensor":
            e = self.query(torch.tanh(self.proj(x))).squeeze(-1)  # (batch, seq)
            if mask is not None:
                e = e.masked_fill(~mask, float("-inf"))
            alpha = torch.softmax(e, dim=-1)
            return (alpha.unsqueeze(-1) * x).sum(dim=1)

    class NRMSUserEncoder(nn.Module):
        """History -> multi-head self-attention -> additive-attention pooling -> user vector."""

        def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, dropout: float) -> None:
            super().__init__()
            self.proj = nn.Linear(input_dim, hidden_dim)
            self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.pool = AdditiveAttention(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, hist_vecs: "torch.Tensor") -> "torch.Tensor":
            mask = hist_vecs.abs().sum(dim=-1) > 0  # True where a history slot is populated
            x = self.dropout(F.relu(self.proj(hist_vecs)))
            x, _ = self.self_attn(x, x, x, key_padding_mask=~mask)
            x = self.dropout(x)
            return self.pool(x, mask)

    class NRMSModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, dropout: float) -> None:
            super().__init__()
            self.article_proj = nn.Linear(input_dim, hidden_dim)
            self.user_encoder = NRMSUserEncoder(input_dim, hidden_dim, num_heads, dropout)

        def forward(self, hist_vecs: "torch.Tensor", cand_vecs: "torch.Tensor") -> "torch.Tensor":
            """``hist_vecs``: (batch, hist_len, dim); ``cand_vecs``: (batch, n_cand, dim).

            Returns click logits of shape (batch, n_cand).
            """
            user_vec = self.user_encoder(hist_vecs)               # (batch, hidden)
            cand_proj = F.relu(self.article_proj(cand_vecs))       # (batch, n_cand, hidden)
            return torch.bmm(cand_proj, user_vec.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Trainer / recommender wrapper
# ---------------------------------------------------------------------------

class NRMS:
    """Fit/score wrapper around :class:`NRMSModel`, matching the other models' interface."""

    name = "nrms"

    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
        epochs: int = 5,
        batch_size: int = 128,
        lr: float = 1e-3,
        neg_ratio: int = 4,
        max_history: int = 50,
        seed: int = 0,
        device: str | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NRMS. Install it with `pip install torch`.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.neg_ratio = neg_ratio
        self.max_history = max_history
        self.seed = seed
        self.checkpoint_path = checkpoint_path

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self._model: "NRMSModel | None" = None
        self._embeddings: np.ndarray | None = None
        self._id2idx: dict[int, int] = {}

    def fit(
        self,
        behaviors: pd.DataFrame,
        user_history: dict[int, list[int]],
        embedding_matrix: np.ndarray,
        id2idx: dict[int, int],
    ) -> "NRMS":
        """Train from scratch, or load a cached checkpoint if one already exists at
        ``checkpoint_path`` -- reruns of the pipeline should not repeat a multi-minute
        training run.
        """
        torch.manual_seed(self.seed)
        self._embeddings, self._id2idx = embedding_matrix, id2idx

        if self.checkpoint_path and self.checkpoint_path.exists():
            log.info("Loading NRMS checkpoint from %s", self.checkpoint_path)
            self._model = NRMSModel(self.input_dim, self.hidden_dim, self.num_heads, self.dropout)
            self._model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            self._model.to(self.device).eval()
            return self

        log.info("Building training dataset ...")
        dataset = ImpressionDataset(
            behaviors, user_history, embedding_matrix, id2idx, self.max_history, self.neg_ratio
        )
        if len(dataset) == 0:
            raise ValueError("Training dataset is empty -- check `behaviors` and the embeddings.")
        log.info("Training examples: %d", len(dataset))

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=(self.device.type == "cuda"),
        )
        self._model = NRMSModel(self.input_dim, self.hidden_dim, self.num_heads, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            total_loss, n_batches = 0.0, 0
            for hist_vecs, cand_vecs, labels in loader:
                hist_vecs, cand_vecs, labels = (
                    hist_vecs.to(self.device), cand_vecs.to(self.device), labels.to(self.device)
                )
                logits = self._model(hist_vecs, cand_vecs)
                # Position 0 is always the clicked article (see ImpressionDataset).
                loss = F.cross_entropy(logits, labels.argmax(dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            log.info("epoch %d/%d | loss=%.4f", epoch, self.epochs, total_loss / max(n_batches, 1))

        self._model.eval()
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), self.checkpoint_path)
            log.info("Checkpoint saved to %s", self.checkpoint_path)
        return self

    def _score_one(self, history: list[int], article_ids: list[int]) -> list[tuple[int, float]]:
        dim = self._embeddings.shape[1]
        if not any(self._id2idx.get(aid) is not None for aid in history[-self.max_history:]):
            # Cold-start fallback (users with no usable history, 16.9% of the paper's test
            # impressions): an all-masked attention would return NaN logits. Return tied
            # scores instead, which keeps the logged candidate order for the "Original"
            # ranking and makes every reranker treat relevance as constant -- exactly
            # what the paper's (NaN-producing) run did, now explicit. See paper Sec. 4.2.
            return [(aid, 0.0) for aid in article_ids]
        hist = history[-self.max_history:]
        hist_arr = np.zeros((1, self.max_history, dim), dtype=np.float32)
        for i, aid in enumerate(hist):
            idx = self._id2idx.get(aid)
            if idx is not None:
                hist_arr[0, i] = self._embeddings[idx]

        cand_arr = np.zeros((1, len(article_ids), dim), dtype=np.float32)
        for j, aid in enumerate(article_ids):
            idx = self._id2idx.get(aid)
            if idx is not None:
                cand_arr[0, j] = self._embeddings[idx]

        with torch.no_grad():
            logits = self._model(
                torch.FloatTensor(hist_arr).to(self.device),
                torch.FloatTensor(cand_arr).to(self.device),
            )[0].cpu().numpy()

        scored = list(zip(article_ids, (float(x) for x in logits)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def score_all_impressions(
        self, behaviors: pd.DataFrame, user_history: dict[int, list[int]]
    ) -> dict[int, list[tuple[int, float]]]:
        results: dict[int, list[tuple[int, float]]] = {}
        for row in behaviors.itertuples(index=False):
            results[int(row.impression_id)] = self._score_one(
                user_history.get(int(row.user_id), []), list(row.article_ids_inview)
            )
        log.info("NRMS scored %d impressions", len(results))
        return results
