"""Export the MiniFASNet anti-spoofing weights to ONNX.

Run this once, in a throwaway container that has torch; the runtime image never
needs it. See tools/README.md for the exact command.

The upstream model definitions and weights are imported from the cloned repo
rather than reimplemented here - the architecture has enough small details
(kernel sizes derived from the input size, DataParallel key prefixes) that
retyping them is a good way to produce a model that loads but is subtly wrong.
"""
import os
import sys

import torch

REPO = os.environ.get("MINIFAS_REPO", "/repo")
OUT = os.environ.get("MINIFAS_OUT", "/out")

sys.path.insert(0, REPO)
from src.model_lib.MiniFASNet import (  # noqa: E402
    MiniFASNetV1,
    MiniFASNetV1SE,
    MiniFASNetV2,
    MiniFASNetV2SE,
)
from src.utility import get_kernel, parse_model_name  # noqa: E402

MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}

weights_dir = os.path.join(REPO, "resources", "anti_spoof_models")
os.makedirs(OUT, exist_ok=True)

for name in sorted(os.listdir(weights_dir)):
    if not name.endswith(".pth"):
        continue
    h_input, w_input, model_type, scale = parse_model_name(name)
    kernel_size = get_kernel(h_input, w_input)
    model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size)

    state = torch.load(os.path.join(weights_dir, name), map_location="cpu")
    # Saved from DataParallel, so every key carries a "module." prefix.
    first_key = next(iter(state))
    if first_key.startswith("module."):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, h_input, w_input)
    with torch.no_grad():
        reference = model(dummy)
        # Two different inputs must give different outputs.  Comparing the
        # export against the torch model cannot catch a model that ignores its
        # input - both come out identically constant and the check passes.  An
        # earlier version of this script shipped exactly that: weights that
        # returned the same logits for zeros, ones and noise alike.
        other = model(torch.zeros(1, 3, h_input, w_input))
        spread = float((reference - other).abs().max())

    out_path = os.path.join(OUT, name.replace(".pth", ".onnx"))
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )

    # Prove the export matches the torch model before trusting the file.
    import numpy as np
    import onnxruntime

    session = onnxruntime.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = session.run(None, {"input": dummy.numpy()})[0]
    delta = float(np.abs(got - reference.numpy()).max())

    print(f"{name}")
    print(f"  type {model_type}  input {h_input}x{w_input}  crop scale {scale}")
    print(f"  output shape {tuple(reference.shape)}")
    print(f"  responds to input: max |noise - zeros| = {spread:.4f}  "
          f"{'OK' if spread > 0.5 else 'DEAD - weights did not take effect'}")
    print(f"  onnx vs torch max abs diff {delta:.3e}  {'OK' if delta < 1e-4 else 'MISMATCH'}")
    print(f"  wrote {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
