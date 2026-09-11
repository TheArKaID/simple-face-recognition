# Third-party notices

This service ships two sets of pretrained model weights that it did not train.
Their terms are not the same, and one of them needs a decision from whoever
owns this deployment.

## MiniFASNet - liveness / anti-spoofing

* Source: [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing), MiniVision (minivision-ai)
* Licence: Apache License 2.0
* Shipped as: `models/liveness/*.onnx`, converted from upstream `.pth` by `tools/convert_minifas.py`
* Attribution: `models/liveness/NOTICE`, licence text in `models/liveness/LICENSE`

These weights are committed to this repository, so the repository redistributes
them. Apache 2.0 allows that and asks for the notice and licence copy that are
now in place.

## InsightFace buffalo_m - detection and recognition

* Source: [insightface](https://github.com/deepinsight/insightface) release `v0.7`, `buffalo_m.zip`
* Shipped as: `models/arcface/det_2.5g.onnx` and `models/arcface/w600k_r50.onnx`, fetched during the Docker build (see `Dockerfile`), not committed
* Code licence: MIT, no restriction on commercial use

**The model weights are not under the MIT licence, and this needs attention.**
Upstream states that the training data and the models trained on it "are
available for non-commercial research purposes only", and says explicitly that
this applies to models downloaded manually from their GitHub repository - which
is exactly how the Dockerfile obtains them.

This service performs employee attendance for a company, which is not
non-commercial research. So the recognition weights this system depends on
appear to fall outside the terms they were published under. That is a licensing
question, not an engineering one, and it is recorded here rather than resolved:

* Verify the current terms at the source before any production deployment.
* Upstream directs commercial enquiries to the InsightFace team for licensing.
* If the terms cannot be met, the engine is replaceable without touching the
  rest of the system - `engine.py` documents the backend contract, and
  `config._vector_space_id` keeps a different model's templates isolated from
  this one's. What is not cheap is the calibration: thresholds are specific to
  the vector space that produced them, so a replacement needs `tests/calibrate.py`
  re-run from scratch and every employee re-enrolled.

Note that a model licence restriction is not visible in any test result. The
system works exactly as well either way, which is why this file exists.

## Python dependencies

Runtime dependencies (onnxruntime, opencv-python-headless, fastapi, uvicorn,
numpy, pillow) carry their own licences in their distribution metadata. None of
their weights or data are redistributed here.
