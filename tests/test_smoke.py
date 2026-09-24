import torch

import jevmark


def test_tiny_model_forward(tokenizer, tiny_model):
    assert jevmark.__version__
    ids = tokenizer("### State\nhello\n\n### Question: Is this a greeting?\nAnswer:", return_tensors="pt").input_ids
    with torch.no_grad():
        logits = tiny_model(input_ids=ids).logits
    assert logits.shape == (1, ids.shape[1], len(tokenizer))
    assert torch.isfinite(logits).all()
