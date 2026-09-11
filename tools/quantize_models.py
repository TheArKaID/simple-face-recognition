"""Quantise the ArcFace ONNX models to INT8 and report what it actually bought.

Run in a throwaway container that has the `onnx` package; the runtime image
deliberately does not carry it. See tools/README.md.

Two modes, and the difference matters:

  dynamic   weight-only.  No calibration data, weights stored as int8 and
            dequantised on the fly.  Safe, but on a convolution-heavy network
            ONNX Runtime only quantises MatMul/Gemm by default, so a ResNet may
            barely shrink and may not speed up at all.
  static    weights and activations, calibrated on real aligned face crops.
            Bigger win where the CPU has VNNI, but every activation scale is
            fitted to the calibration set, so it can distort embeddings in ways
            weight-only does not.

Neither is adopted on the strength of size alone. What decides it is the
embedding drift measured by tools/compare_quantized.py against the FP32
baseline, weighed against the impostor margin the deployment actually runs on.
"""
import os
import sys

import glob

import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

MODEL_DIR = os.getenv("FACE_ARCFACE_MODEL_DIR", "/models")
CALIB_DIR = os.getenv("QUANT_CALIB_DIR", os.path.join(MODEL_DIR, "calib"))
TARGETS = [t for t in os.getenv("QUANT_TARGETS", "w600k_r50.onnx,det_2.5g.onnx").split(",") if t]
MODE = os.getenv("QUANT_MODE", "static").strip().lower()

# Which calibration tensors belong to which model.
CALIB_PREFIX = {"w600k_r50.onnx": "rec_", "w600k_mbf.onnx": "rec_", "det_2.5g.onnx": "det_"}


class NpyReader(CalibrationDataReader):
    """Feeds the real tensors the pipeline produces, not raw images.

    Activation scales are fitted to whatever this yields, so calibrating on
    anything other than genuinely aligned crops and letterboxed detector blobs
    would fit the quantiser to a distribution the service never sees.
    """

    def __init__(self, input_name, files):
        self.input_name = input_name
        self.files = list(files)
        self.i = 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        arr = np.load(self.files[self.i]).astype(np.float32)
        self.i += 1
        return {self.input_name: arr}

    def rewind(self):
        self.i = 0


def op_histogram(path):
    model = onnx.load(path)
    counts = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


for name in TARGETS:
    src = os.path.join(MODEL_DIR, name)
    if not os.path.exists(src):
        print(f"  {name}: tidak ada di {MODEL_DIR}, dilewati")
        continue
    dst = os.path.join(MODEL_DIR, name.replace(".onnx", "_int8.onnx"))

    before = os.path.getsize(src)
    ops = op_histogram(src)
    top = sorted(ops.items(), key=lambda kv: -kv[1])[:5]

    if MODE == "dynamic":
        # Kept for the record, and it does not work here.  Including Conv emits
        # ConvInteger nodes, which the ONNX Runtime CPU provider does not
        # implement - the file shrinks 75% and then refuses to load.  Excluding
        # Conv leaves a 53-convolution ResNet essentially unquantised.  Neither
        # is a usable outcome, which is why static is the default.
        quantize_dynamic(
            model_input=src, model_output=dst, weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        )
    else:
        prefix = CALIB_PREFIX.get(name, "rec_")
        files = sorted(glob.glob(os.path.join(CALIB_DIR, prefix + "*.npy")))
        if not files:
            print(f"  {name}: tidak ada tensor kalibrasi {prefix}*.npy di {CALIB_DIR}")
            continue
        input_name = onnxruntime.InferenceSession(
            src, providers=["CPUExecutionProvider"]).get_inputs()[0].name

        # Per-channel weights need opset 13: below that, DequantizeLinear has no
        # `axis` attribute and the quantiser emits a graph onnxruntime refuses to
        # load ("Unrecognized attribute: axis").  The insightface models ship at
        # opset 11, so upgrade first and fall back to per-tensor if the converter
        # cannot manage it.
        model = onnx.load(src)
        current = next((o.version for o in model.opset_import
                        if o.domain in ("", "ai.onnx")), 0)
        per_channel = True
        quant_src = src
        if current < 13:
            try:
                upgraded = onnx.version_converter.convert_version(model, 13)
                quant_src = os.path.join(MODEL_DIR, name.replace(".onnx", "_op13.onnx"))
                onnx.save(upgraded, quant_src)
                print(f"    opset            : {current} -> 13")
            except Exception as exc:
                per_channel = False
                print(f"    opset            : {current}, konversi ke 13 gagal "
                      f"({type(exc).__name__}), turun ke per-tensor")

        quantize_static(
            model_input=quant_src,
            model_output=dst,
            calibration_data_reader=NpyReader(input_name, files),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
        )
        if quant_src != src and os.path.exists(quant_src):
            os.remove(quant_src)
        print(f"    kalibrasi        : {len(files)} tensor {prefix}*.npy")

    after = os.path.getsize(dst)
    ops_after = op_histogram(dst)
    quantised = sum(v for k, v in ops_after.items()
                    if k.startswith("QLinear") or k in ("QuantizeLinear", "ConvInteger"))

    print(f"  {name}")
    print(f"    op terbanyak     : {', '.join(f'{k}x{v}' for k, v in top)}")
    print(f"    ukuran           : {before/1048576:.1f} MB -> {after/1048576:.1f} MB "
          f"({(1 - after/before)*100:.0f}% lebih kecil)")
    print(f"    node terkuantisasi: {quantised}")
    # Load it before claiming anything.  A file that shrank 75% and cannot be
    # instantiated is not a smaller model, it is a broken one - and the size
    # figure alone reads like success.
    try:
        onnxruntime.InferenceSession(dst, providers=["CPUExecutionProvider"])
        print(f"    dapat dimuat     : ya")
    except Exception as exc:
        print(f"    dapat dimuat     : TIDAK - {type(exc).__name__}: {str(exc)[:120]}")
    print(f"    ditulis          : {dst}")
