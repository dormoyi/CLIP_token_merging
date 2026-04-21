import pytest

torch = pytest.importorskip("torch")

from clip.model import VisionTransformer


def test_merge_topk_pairs_reduces_sequence_and_averages():
    vit = VisionTransformer(
        input_resolution=8,
        patch_size=4,
        width=4,
        layers=1,
        heads=1,
        output_dim=4,
        tome_pairs_per_layer=2,
    )

    # [cls, t0, t1, t2, t3, t4, t5]
    x = torch.tensor(
        [
            [
                [9.0, 9.0, 9.0, 9.0],  # cls
                [1.0, 0.0, 0.0, 0.0],  # t0
                [1.0, 0.0, 0.0, 0.0],  # t1 (identical to t0)
                [0.0, 1.0, 0.0, 0.0],  # t2
                [0.0, 1.0, 0.0, 0.0],  # t3 (identical to t2)
                [0.0, 0.0, 1.0, 0.0],  # t4
                [0.0, 0.0, 0.0, 1.0],  # t5
            ]
        ]
    )

    merged = vit._merge_topk_token_pairs(x, num_pairs=2)

    # 6 patch tokens -> merge 2 pairs -> 4 patch tokens (+1 cls).
    assert merged.shape == (1, 5, 4)
    assert torch.allclose(merged[:, 0, :], x[:, 0, :])  # cls untouched

    # Merged representatives are means of each selected pair.
    expected_pair_a = (x[:, 1, :] + x[:, 2, :]) * 0.5
    expected_pair_b = (x[:, 3, :] + x[:, 4, :]) * 0.5
    patch_tokens = merged[:, 1:, :]
    assert (torch.isclose(patch_tokens, expected_pair_a.unsqueeze(1)).all(dim=-1).any(dim=1)).all()
    assert (torch.isclose(patch_tokens, expected_pair_b.unsqueeze(1)).all(dim=-1).any(dim=1)).all()
