# Instavar Breeze TTS 2 fine-tuning

An independent, source-only companion for LoRA and full supervised fine-tuning
of [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts).

This repository adds the training and evaluation surface that is not included in
the upstream inference release. It does not contain model weights, adapters,
checkpoints, training audio, or generated speech.

Instavar is not affiliated with or endorsed by BreezeBlue or RESONIA, INC.

## Published research derivatives

- [Instavar SG Narration LoRA R8](https://huggingface.co/instavar/sg-narration-lora-r8)
- [Instavar SG Narration Full SFT](https://huggingface.co/instavar/sg-narration-full-sft)
- [Breeze TTS 2 fine-tuning collection](https://huggingface.co/collections/instavar/breeze-tts-2-fine-tuning-by-instavar-6a9a9a9d15c0cacc893b3c55)
- [Matched LoRA and full-SFT experiment report](https://instavar.com/research/tts/breeze-tts-2-lora-full-sft-singapore-english)

Both Hugging Face repositories are gated, non-commercial research releases.
Their model cards document the exact base revision, selected checkpoint,
training scope, checksums, evaluation limits, licence, and required notices.

## What is implemented

| Capability | Implementation |
| --- | --- |
| Supervised labels | Reconstructs text and 16-codebook audio targets from an exact transcript and target WAV |
| Feasibility gate | One BF16 forward and backward step, finite losses, nonzero family gradients, changed weights, and fresh-process reload |
| LoRA | Backbone, depth decoder, text projection, depth-input projection, and output projection targets |
| LoRA lifecycle | Trainable-parameter receipt, checkpoint, resume, adapter export, merge, and fresh-process merged reload |
| Adapter release runtime | Manifest-driven adapter loading with base revision, base-file, and adapter checksum validation |
| Full SFT | BF16 synthesis model with frozen codec and text encoder, FP32-master SGD or Adafactor, family learning-rate multipliers, and bounded checkpoints |
| Full-SFT sweeps | JSON plans, multiple seeds, independent schedule horizons, validation-only trials, and lightweight finalist exports |
| Evaluation | Matched prompts and seeds, ASR, speaker similarity, acoustic diagnostics, confidence intervals, and an opaque blind-listening pack |
| Safety and reproducibility | Refuses output overwrite, hashes source audio and artifacts, records Git state, writes atomic receipts, and handles SIGTERM at checkpoint boundaries |

These tools were exercised on an NVIDIA RTX 3090 Ti. That establishes the
documented code paths on one 24 GB CUDA system, not general convergence or model
quality on every dataset or GPU.

## Install

Use Windows PowerShell, Python 3.12, `uv`, and an NVIDIA CUDA GPU. `uv` creates
the repository-local `.venv`; its download cache remains in the normal user
cache instead of being copied into this repository.

```powershell
git clone https://github.com/instavar/breeze-tts2-finetuning.git
Set-Location breeze-tts2-finetuning
uv sync --extra evaluation --group dev
.\.venv\Scripts\Activate.ps1
```

The PowerShell launchers select physical GPU 0 by default. On the documented
machine that is the RTX 5090; set `$env:CUDA_VISIBLE_DEVICES` explicitly to
override it.

Obtain Breeze TTS 2 separately from its official distribution after reading and
accepting its model agreement. Do not add the checkpoint to this repository.

## Low-latency web inference

The streaming API includes a dark browser client at `/`. It plays raw PCM as it
arrives, reports time to first audio (TTFA) and real-time factor (RTF), and can
switch between validated LoRA releases without restarting the server.

The server enables the warmed `--fast-all` path by default. This follows the
[official Breeze TTS 2 inference guidance](https://github.com/breezeblue-ai/breeze-tts#%EF%B8%8F-fast-inference-options):
CUDA graphs are used for the text encoder, backbone prefill/decode, depth
decoder, and one-frame streaming codec. The upstream documentation reports
about 14.4 GiB of GPU memory for this mode. Use `--no-fast-all` when cold-start
time or memory matters more than request latency.

```powershell
.\scripts\start_ui.ps1 `
  -ModelRoot 'D:\models\Breeze-TTS-2' `
  -AdaptersDir 'D:\models\breeze-adapters' `
  -ListenAddress '0.0.0.0' `
  -Port 7860
```

Then open <http://127.0.0.1:7860/>. Each immediate child (or nested directory)
under `--adapters-dir` that contains an `adapter_config.json` release manifest
appears in the LoRA dropdown. Weight changes release the current runtime, load
and validate the selected adapter, rebuild the streaming runtime, warm its CUDA
graphs, and report phase progress plus an estimated remaining time.

The headline TTFA is deliberately not HTTP time to first byte. The client uses
the [BreezeBlue latency benchmark](https://github.com/breezeblue-ai/TTS-Latency-Benchmark#measurement-protocol)
definition and its relative dual-energy/hysteresis onset detector: TTFA is the
later of the voice-onset audio clock and the time the onset samples became
available to the browser. RTF is request-to-stream-EOF wall time divided by the
decoded audio duration. Both measurements therefore include local HTTP delivery
over the actual client path.

## Dataset manifest

Create separate train and validation JSONL files. Each row has an absolute audio
path and its exact transcript:

```json
{"audio":"D:\\data\\speaker\\0001.wav","text":"The exact words spoken in this file."}
```

Use clean, single-speaker recordings and only material for which you have all
necessary rights and consent. Keep the validation recordings disjoint from the
training recordings.

For the included ignored `p003` working data, the only supported source folder
is `p003\tail-sigh-cliping-removed`. The converter reads WAVs and transcript
metadata directly from that folder; no `full`, `stt`, `seam`, or `24` copy is
required.

## Prepare deterministic targets

```powershell
$env:BREEZE_MODEL_ROOT = 'D:\models\Breeze-TTS-2'
$env:BREEZE_TRAIN_MANIFEST = 'D:\data\train.jsonl'
$env:BREEZE_VALIDATION_MANIFEST = 'D:\data\validation.jsonl'
$env:BREEZE_CACHE_ROOT = 'D:\runs\cache-v1'

.\scripts\prepare_cache.ps1 --train-limit 1024 --validation-limit 128
```

The cache stores model-ready tensors and a receipt containing hashes and source
revision information. Treat it as sensitive if the source dataset is sensitive.

## Run LoRA

```powershell
$env:BREEZE_RUN_ROOT = 'D:\runs\lora-r8'
.\scripts\run_lora.ps1 `
  --rank 8 `
  --alpha 16 `
  --max-steps 1000 `
  --gradient-accumulation 4 `
  --learning-rate 2e-4 `
  --save-every 250
```

Resume by using a new output root and passing the prior checkpoint explicitly:

```powershell
$env:BREEZE_RUN_ROOT = 'D:\runs\lora-r8-resumed'
.\scripts\run_lora.ps1 `
  --resume-checkpoint 'D:\runs\lora-r8\checkpoint-step-000250' `
  --max-steps 1000
```

Adapter verification and merge are separate so a failed verification cannot
overwrite the training run:

```powershell
uv run python -m training.real_lora_verify `
  --model-root $env:BREEZE_MODEL_ROOT `
  --cache-root $env:BREEZE_CACHE_ROOT `
  --training-root 'D:\runs\lora-r8' `
  --merged-output 'D:\runs\lora-r8-merged'

uv run python -m training.real_lora_merged_smoke `
  --cache-root $env:BREEZE_CACHE_ROOT `
  --training-root 'D:\runs\lora-r8' `
  --merged-model 'D:\runs\lora-r8-merged'
```

Published adapters can also be loaded without creating a merged checkpoint:

```powershell
.\scripts\run_inference.ps1 `
  -ModelRoot $env:BREEZE_MODEL_ROOT `
  -Adapter 'D:\models\sg-narration-lora-r8' `
  -Text 'The train arrives in five minutes.' `
  -Output 'output.wav' `
  --base-revision '<exact-base-revision>'
```

The adapter manifest pins the base revision and required base-model file hashes.
Loading fails closed if any pinned identity check differs.

## Run full SFT

Start with the one-example feasibility gate before committing to a real run:

```powershell
$env:BREEZE_TRAIN_AUDIO = 'D:\data\speaker\0001.wav'
$env:BREEZE_TRAIN_TRANSCRIPT = 'The exact words spoken in this file.'
$env:BREEZE_RUN_ROOT = 'D:\runs\full-sft-smoke'
.\scripts\run_full_sft_smoke.ps1
```

Then run multi-example full SFT:

```powershell
$env:BREEZE_RUN_ROOT = 'D:\runs\full-sft'
.\scripts\run_full_sft.ps1 `
  --optimizer fp32_master_sgd `
  --max-steps 1000 `
  --gradient-accumulation 4 `
  --learning-rate 2e-5 `
  --save-every 250
```

See [training details](docs/TRAINING.md) before choosing an optimizer or copying
one of the included sweep plans.

## Evaluate before selecting

Do not choose a checkpoint from training loss alone. The repository supports:

- validation-based checkpoint selection;
- matched base, LoRA, and full-SFT generation;
- ASR word-error diagnostics;
- ECAPA speaker-similarity diagnostics;
- pitch, energy, pause, and duration diagnostics;
- bootstrap confidence intervals; and
- blind listening packs with a mode-`0600` private key.

See [evaluation](docs/EVALUATION.md) for the evaluation sequence and what each
measurement does not prove.

## License boundary

Repository source is provided under Apache License 2.0. Breeze TTS 2 model
materials, derivative models, adapters, checkpoints, and self-hosted outputs are
governed separately by the BreezeBlue Research and Non-Commercial License.

The source license does not grant a right to download, use, distribute, or use
Breeze model materials commercially. See [model license boundary](MODEL_LICENSE.md)
and [release boundaries](docs/RELEASE_BOUNDARIES.md).

The repository also includes a fail-closed release-bundle builder. See
[model release operations](docs/RELEASING_MODELS.md).

## Acknowledgments

- [BreezeBlue](https://breezeblue.ai/) for Breeze TTS 2 and its upstream PyTorch runtime
- [Qwen](https://github.com/QwenLM/Qwen3-TTS) for the Qwen3-TTS audio tokenizer used by Breeze
- The open-source libraries listed in `requirements.txt`

## Citation

See [`CITATION.cff`](CITATION.cff).
