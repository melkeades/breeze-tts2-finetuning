# Training guide

## Training surface

The upstream release exposes inference. This companion reconstructs supervised
training targets from an exact transcript and target audio:

1. tokenize the transcript with the checkpoint tokenizer;
2. encode the audio through the released audio tokenizer;
3. place ignore labels on text positions;
4. place the audio marker on audio placeholder positions; and
5. let the released model expand those positions into the 16 codebook targets
   and the backbone end-of-sequence target.

The codec and text encoder remain frozen. LoRA targets the synthesis backbone,
depth decoder, and the text, depth-input, and output projections. Full SFT trains
the released synthesis parameters while keeping those same two components
frozen.

## Admission sequence

Use these gates in order:

1. `scripts/run_full_sft_smoke.ps1` proves one finite BF16 update and a fresh
   reload on one example.
2. `scripts/prepare_cache.ps1` creates deterministic train and validation tensor
   caches from disjoint manifests.
3. `scripts/run_lora.ps1` or `scripts/run_full_sft.ps1` runs the selected path.
4. Select a checkpoint from held-out validation history.
5. Run the corresponding fresh-process verification and export.
6. Generate matched audio and freeze blind ratings before decoding identities.

Passing the feasibility gate proves that labels, forward loss, gradients,
updates, serialization, and reload are connected. It does not prove useful
adaptation, speaker similarity, naturalness, or convergence.

## LoRA lifecycle

`training.real_lora` writes atomic five-role checkpoints containing the adapter,
optimizer, scheduler, random-number state, and receipt. SIGTERM requests a clean
stop at a checkpoint boundary. Resume requires an explicit checkpoint and a new
output root.

The released checkpoint prefers FlashAttention 2 only for its frozen T5Gemma2
text encoder. To retain that scoped preference during LoRA training, install the
locked Windows environment and pass:

```powershell
.\scripts\run_lora.ps1 `
  --attention-implementation eager `
  --text-encoder-attention-implementation flash_attention_2
```

Do not apply FlashAttention 2 globally to the causal backbone or depth decoder.
The Windows RTX 5090 admission run produced finite forward losses but non-finite
backward gradients in those paths. The CLI therefore exposes FA2 only as a
text-encoder override; `eager` and `sdpa` remain the supported global training
backends.

Experimental `torch.compile` support can compile only the synthesis backbone's
repeated transformer blocks:

```powershell
.\scripts\run_lora.ps1 `
  --compile-regions backbone `
  --compile-mode default
```

The first step performs a cold compile. The Inductor cache makes later launches
cheaper, and the trainer initializes the installed Visual Studio x64 build
environment automatically when it is started from a normal PowerShell session.
Do not use `reduce-overhead` for these regional blocks: its CUDA Graph output
lifetime is incompatible with one compiled wrapper per transformer layer.

Compilation and the text-encoder FlashAttention override remain opt-in because
their BF16 results are not bit-identical to eager training. A paired 250-step RTX
5090 experiment evaluated 80 generated WAVs per checkpoint: 20 neutral, 20 with
style instructions, 20 sound-tag controls, and 20 sound-tagged prompts. Relative
to native eager training, backbone compilation changed mean NISQA by +0.0187 and
compilation plus text-encoder FA2 changed it by +0.0184; both paired 95% bootstrap
intervals included zero. P808 and WavLM identity likewise showed no material
aggregate regression. These automatic proxies support using the options for
experiments, but a blind listening test is still required for a perceptual claim.

After validation selection, `training.real_lora_verify` reloads the adapter in a
fresh process, evaluates held-out examples, merges the adapted linear layers,
and exports a separate merged package. `training.real_lora_merged_smoke` then
reloads that package independently.

## Full-SFT lifecycle

`training.real_full_sft` supports:

- FP32-master SGD for a low-state optimizer on a 24 GB GPU;
- Adafactor as an alternative low-memory optimizer;
- separate learning-rate multipliers for the backbone, depth decoder, text
  projection, and remaining synthesis parameters;
- an independent cosine schedule horizon;
- validation without checkpoint export;
- interval, final, model-only, or no-save policies; and
- explicit resume from a complete checkpoint.

Full-SFT exports are large. Check local disk capacity before enabling interval
checkpoints and never point two runs at the same output root.

## Sweep plans

`training.full_sft_sweep` reads a JSON object with `defaults` and `trials`.
Every trial receives its own directory and command receipt. The plans in
`training/plans/` are examples from one bounded experiment, not universal
recommendations. Re-test learning rates when the dataset, batch construction,
optimizer, schedule horizon, or model revision changes.

## Data and consent

The tools accept arbitrary manifest paths but do not grant rights to any audio,
speaker, transcript, model, or output. Dataset availability is not proof of
permission to create or distribute a reusable voice model. Keep the rights and
consent record outside Git and bind it to the dataset version used for training.
