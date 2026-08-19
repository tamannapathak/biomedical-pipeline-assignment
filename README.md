# Biomedical Image-Analysis Pipeline — Nuclei / Fluorescence Microscopy

Assignment 3 (Data Science & AI in Practice): a hybrid biomedical image-analysis pipeline combining a
local multimodal LLM, classical image processing, and a U-Net segmentation network, applied to a
synthetic fluorescence-microscopy nuclei dataset.

`raw image -> segmentation -> quantitative region features -> structured JSON record -> short narrative`

## Repository layout

```
biomedical-pipeline-assignment/
├── README.md                  <- you are here
├── requirements.txt
├── notebooks/
│   └── biomedical_pipeline.ipynb   <- MAIN DELIVERABLE: the full, SELF-CONTAINED pipeline
│                                       (Tasks 1-4 + extensions -- everything in one notebook)
├── src/
│   └── biomedical_pipeline.py    <- a plain .py mirror of the notebook above (auto-exported via
│                                     `jupyter nbconvert --to script`), included purely so the code
│                                     is easy to browse/diff on GitHub. The notebook never imports
│                                     from this file -- it is not a package, just a read-only copy.
├── data/
│   └── nuclei_dataset/          <- dataset (train/val/test/test_corrupted + metadata.csv)
├── outputs/
│   ├── figures/                 <- all generated figures (PNG)
│   ├── csv/                     <- all generated tables (feature tables, comparisons, histories)
│   └── models/                  <- trained U-Net checkpoints (.pt)
└── report/
    └── report_template.docx    <- 4-page report skeleton aligned to the marking rubric (fill in yourself)
```

**This project is intentionally a single notebook, not a package split across many files.** Every
function/class used anywhere in the pipeline (configuration, data loading, the classical Otsu pipeline,
the U-Net model, the Ollama/LLM wrapper, the Task 4 hybrid pipeline) is defined directly inside the
notebook's "Core Functions" cells, grouped by pipeline stage — see the notebook's own "Core Functions"
overview cell for the full list (Configuration; Data loading & EDA; Classical Otsu pipeline; U-Net model,
losses & metrics; train-or-load caching; LLM/VLM utilities; hybrid pipeline & robustness helpers). There
is nothing to hunt for in a separate file: edit a function in the notebook, re-run that cell and
everything below it.

All paths in the notebook are computed relative to its own location (`PROJECT_ROOT = ../` from
`notebooks/`) — there are no machine-specific absolute paths anywhere, so this runs unmodified on any
computer after `git clone`.

## Dataset

`data/nuclei_dataset/` (synthetic, CC0) — see `data/nuclei_dataset/README.md` for full details. Train=80,
val=20, test=12 image/mask/instance-label triples, plus `test_corrupted/` (pre-corrupted variants of two
test images, used by the robustness extension). Source assignment dataset:
<https://github.com/Nickolay-K/Assingnment-3-dataset>.

## Setup

### 1. Python environment

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 2. Ollama (required for the LLM/VLM steps in Tasks 1, 2, and 4)

```bash
curl -fsSL https://ollama.com/install.sh | sh     # macOS/Linux; Windows: download from ollama.com/download
ollama serve &                                     # start the local server (or launch the Ollama app)
ollama pull llava:7b                               # ~4.7 GB — Task 1 (direct VLM description; must be vision-capable)
ollama pull qwen2.5:3b                             # ~2 GB   — Tasks 2 and 4 (numbers-first / hybrid record, text-only)
```

**Why `llava:7b` and not `llama3.2-vision`:** the assignment brief names `llama3.2-vision`, and
`CONFIG.vlm_model` (set in the notebook's Configuration cell) can be switched back to it if your Ollama
install runs it cleanly. On the machine this was tested on, `llama3.2-vision` (which uses the newer
`mllama` architecture, requiring Ollama v0.4.0+) failed to load with `unknown model architecture:
'mllama'` even on a fully updated, freshly-reinstalled Ollama with a freshly re-pulled model — confirmed
with a raw `curl http://localhost:11434/api/generate` call that bypassed Python entirely and reproduced
the exact same error, isolating it to the local Ollama/model install rather than anything in this
codebase. `llava:7b` is a smaller, more mature vision architecture also listed as a supported option in
the course's own Lab 2 materials. If you hit the same `mllama` error and want to keep trying
`llama3.2-vision` specifically: confirm the *serving* version with
`curl -s http://localhost:11434/api/version` (not just `ollama --version`, in case a stale background
process is still holding port 11434 — `pkill -9 -f ollama` and relaunch to be sure), and try
`ollama rm llama3.2-vision && ollama pull llama3.2-vision` for a fully clean re-download.

**Ollama must be running before you open the notebook.** See "Error-handling philosophy" below for what
happens if it is not.

### 3. Run the notebook

```bash
jupyter notebook notebooks/biomedical_pipeline.ipynb
```

Run cells top to bottom ("Restart & Run All"). The notebook is fully self-contained — the "Core
Functions" section near the top defines everything the pipeline needs, including the train-or-load
caching helper, which trains from scratch (a few minutes on CPU, no GPU required) if no cached checkpoint
exists yet, or loads the cached checkpoint instantly if one does — see "Reproducibility & caching" below.

**If you edit a Core Functions cell after already running the notebook once, use Kernel → Restart Kernel,
then Run All** — Jupyter keeps old function/class definitions in memory until the kernel is restarted, so
simply re-running a single edited cell (without restarting) can leave you looking at stale behaviour.

## Configuration

Every hyperparameter used anywhere in this project — image size, epochs, batch size, learning rate, the
two Ollama model names, and the random seed — is defined exactly once, in the notebook's "Core Functions
→ 1. Configuration" cell (`CONFIG`). The notebook prints it again near the top of the main flow. To
change an experiment (e.g. run fewer epochs for a quick smoke test), edit that cell, or override a field
in a later cell before training (e.g. `CONFIG.epochs = 3`).

## Reproducibility & caching

`set_global_seed()` (Core Functions, section 1) seeds Python's `random`, NumPy, and PyTorch (CPU + CUDA)
with `CONFIG.seed`, and is called once after Setup and again immediately before every training run inside
`train_or_load` (Core Functions, section 5), so U-Net training is deterministic given the same code and
seed.

`train_or_load(tag, loss_name, ...)` is used for every U-Net trained in this notebook (Task 3's main
model and both loss-ablation models). On a **fresh clone with no `outputs/` committed**, it trains from
scratch and saves the checkpoint + history. On a **repeat run** (checkpoint + history already present),
it loads them instead of retraining. Either way, every number was genuinely produced by this exact code
— caching only decides *when* the computation happened. Pass `force=True` in a notebook cell to always
retrain.

## Error-handling philosophy (no fallback results)

Every LLM/VLM call goes through `chat_json` (Core Functions, section 6), which:

1. Checks Ollama is reachable and that the call to the requested model succeeds. If not, it raises
   `OllamaUnavailableError` immediately, with an actionable fix in the message (install/start Ollama,
   `ollama pull <model>`).
2. Extracts the JSON object from the response, renames any near-miss key spelling back onto the
   exact schema name (`normalize_record_keys` — e.g. a model replying `"density"` instead of
   `"density_class"` is accepted; this only renames a key the model already provided, it never
   invents a missing value), and validates the result against a schema (`validate_vlm_record` /
   `validate_numbers_first_record` / `validate_hybrid_record`) — checking every required field is
   present and every enum-like field (e.g. `density_class`, `quality_flag`) has an allowed value. If
   invalid, it re-prompts the model with the specific problem(s) found and retries
   (`CONFIG.llm_max_retries` times, default 4) before raising `LLMResponseError`. Corrective retries
   also nudge the sampling temperature up slightly and use a concrete filled-in example in the Task
   2/4 prompts rather than an abstract type schema, since a small 3B text model was observed to
   otherwise repeat the same incomplete JSON (e.g. missing one field) on every retry. Task 4's prompt
   additionally asks for the free-form `narrative` sentence *separately* from the JSON object (like
   Task 2's prose-then-JSON pattern), because mixing a long free-text field into the same JSON object
   as short categorical fields was observed to make the small model much less reliable at including
   every categorical field — see `hybrid_record_prompt()`'s docstring for the full story.

(Task 1's naive prompt uses the simpler `chat_text` instead, since that prompt is deliberately free-form
and was never asking for JSON in the first place — see the "Naive prompt vs engineered prompt" note in
the notebook.)

**There is no code path that substitutes a rule-based or placeholder value for a missing/invalid LLM
response.** `run_hybrid_pipeline` (Core Functions, section 7) calls `require_ollama()` before processing
a single test image, so Task 4 either produces a CSV where every `density_class` / `quality_flag` /
`narrative` came from a real, schema-validated model response, or it raises and writes nothing — never a
mix of real and fabricated rows. If you run the notebook without Ollama installed/running, the Task 1/2/4
LLM cells will raise a clear, loud error and stop there — this is intentional, not a bug, and it is
exactly what "the notebook must actually call live LLMs for the final submission" requires: it is now
structurally impossible for those cells to succeed *without* producing genuine output.

### A known limitation of the environment this was authored in

This particular notebook was written and (for every non-LLM step) executed inside a sandboxed cloud
environment with no GPU and no access to `ollama.com`, so the Task 1/2/4 LLM cells could not be executed
there — they show the `OllamaUnavailableError` traceback described above. Every other cell (data prep,
EDA, classical features, U-Net training/evaluation, the Otsu-vs-U-Net comparison, the loss ablation, the
robustness trace) executed fully in that environment, from a genuinely empty `outputs/` folder (a true
"fresh clone" run, not loaded from a pre-existing cache), and the committed notebook contains real
outputs from that run. **Before final submission**, install Ollama locally per the Setup section above
and re-run the Task 1/2/4 cells so the notebook (and the report, which must quote these outputs) contains
live, validated model responses for every image.

## Code organisation

Everything below is a cell inside `notebooks/biomedical_pipeline.ipynb`'s "Core Functions" section (also
mirrored, read-only, in `src/biomedical_pipeline.py`):

- **Preprocessing** — "2. Data loading & EDA utilities" (grayscale/resize, EDA plots)
- **Segmentation** — "3. Classical image processing (Otsu)" and "4. U-Net model, losses & metrics"
  (`unet_predict_mask` for running the trained U-Net lives in section 7)
- **LLM calls** — "6. LLM / VLM utilities (Ollama)" (prompts, validation, retry/raise logic — all models
  call through here)
- **Evaluation** — section 4 (`dice_score`, `iou_score`, `per_image_scores`) — always computed by
  function call, never typed in by hand
- **Exporting results** — each notebook section writes its own figure(s) to `outputs/figures/` and
  table(s) to `outputs/csv/`; `run_hybrid_pipeline` (section 7) returns the aggregated Task 4 DataFrame
  that gets saved to `outputs/csv/hybrid_pipeline_results.csv`

## Final consistency check

Every number, figure, and table referenced in the report should trace back to a cell in
`notebooks/biomedical_pipeline.ipynb` that is actually in this repository — nothing in the report should
be hand-typed or copied from elsewhere. Before submitting: re-run the notebook top to bottom with Ollama
running, confirm no unexpected errors, and spot-check that the figures/tables you pasted into the report
match what's currently in `outputs/`.

## Educational-use disclaimer

None of the models used here (`llava:7b` / `llama3.2-vision`, `qwen2.5:3b`, or the U-Net trained in this
repository) are validated for clinical use. All outputs are for educational purposes only.
