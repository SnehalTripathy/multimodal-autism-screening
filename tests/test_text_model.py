"""Focused tests for the Phase 2 clinical text sub-network.

These tests use synthetic 768-dimensional DistilBERT-style embeddings so they
can run without downloading or storing the AV-ASD dataset.
"""

import torch

from src.models import ClinicalTextMLP


def test_text_feature_and_logit_shapes():
    model = ClinicalTextMLP().eval()
    x = torch.randn(4, 768)

    with torch.no_grad():
        features = model(x, return_features=True)
        logits = model(x, return_features=False)

    assert features.shape == (4, 64)
    assert logits.shape == (4, 1)


def test_text_model_supports_single_sample():
    model = ClinicalTextMLP().eval()
    x = torch.randn(1, 768)

    with torch.no_grad():
        features = model(x, return_features=True)
        logits = model(x, return_features=False)

    assert features.shape == (1, 64)
    assert logits.shape == (1, 1)


def test_text_model_accepts_singleton_sequence_dimension():
    model = ClinicalTextMLP().eval()
    x = torch.randn(4, 1, 768)

    with torch.no_grad():
        features = model(x, return_features=True)

    assert features.shape == (4, 64)