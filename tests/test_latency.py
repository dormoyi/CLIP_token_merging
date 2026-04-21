import time

import pytest
from PIL import Image

import clip

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("model_name", ["ViT-B/32"])
def test_encode_image_latency_dummy_image(model_name):
    """Smoke + timing for one image forward; uses a solid-color PIL image (no assets)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    dummy = Image.new("RGB", (512, 512), color=(42, 128, 200))
    image = preprocess(dummy).unsqueeze(0).to(device)

    def forward():
        with torch.no_grad():
            return model.encode_image(image)

    warmup = 5
    runs = 20
    for _ in range(warmup):
        forward()
    if device == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        feats = forward()
        if device == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)

    assert feats.shape[0] == 1
    assert torch.isfinite(feats).all()

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    # Loose bound: catches hangs / pathological regressions without flaky CI thresholds.
    assert median_ms < 120_000.0

    print(f"encode_image ({model_name}, {device}) median latency: {median_ms:.2f} ms")
