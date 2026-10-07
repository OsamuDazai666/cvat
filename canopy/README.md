# Canopy CVAT (local test)

This is stock CVAT v2.78.0 with our auto-labelling built in through Nuclio functions. It covers vehicles (SAM 3 + Qwen) and plates (detector + Qwen). The Text OCR project has no auto-annotation button; `labeler ocr-fill` can read text boxes you draw (GLM-OCR).

Everything is one disposable Docker Compose project, `canopy-cvat`. Nothing is installed on Windows or WSL. Frames are read-only from `D:\canopy-evidence\training`, and the stack never writes to `D:\canopy-evidence`. Results go back only when you run `apply_pull.py` yourself.

This folder lives inside a fork of CVAT ([OsamuDazai666/cvat](https://github.com/OsamuDazai666/cvat), branch `canopy`). CVAT's own files are unmodified; everything canopy-specific is here, and every command below runs from this folder (`D:\canopy-cvat\cvat\canopy`). Local-only files (`.env` with the admin password, `logs/`, `exports/`) are gitignored.

```
../                   CVAT v2.78.0 (unmodified; its docker-compose files are used as-is)
compose.canopy.yml    overrides: share mount, canopy_models, vllm / vllm_ocr, tools
models/               canopy_models: the one GPU process holding SAM 3 + plate_det_v1
functions/            Nuclio functions (thin CPU adapters) + specs.py (labels/attributes) + gen.py + deploy.sh
labeler/              batch CLI (cvat-sdk) + apply_pull.py (host-side write to index.sqlite)
tests/                parity and smoke tests from the local test
```

**On another machine:**
1. `git clone -b canopy git@github.com:OsamuDazai666/cvat.git`
2. Create `canopy/.env` with `CVAT_USER` and `CVAT_PASSWORD`.
3. Adjust the absolute host paths in `compose.canopy.yml` and `functions/gen.py`: `D:/canopy-evidence/...`, the SAM 3 weights file, and `D:/hf-cache`.
4. Run `.\up.ps1 -Build`, then `.\run.ps1 deploy`.

## Start / stop

```powershell
.\up.ps1 -Build            # first time: builds images, starts CVAT + Nuclio + canopy_models + Qwen3.8
.\run.ps1 deploy           # build + deploy the 3 functions (again after editing functions/ or classes.json)
.\down.ps1                 # stop, keep data
.\down.ps1 -Purge          # remove containers, volumes, network and canopy-cvat images
```

- **CVAT:** http://localhost:9080. Log in with the user in `.env` (`CVAT_USER`, `CVAT_PASSWORD`).
- **Nuclio dashboard:** http://localhost:9070
- **Model server health:** http://localhost:9090/health

## GPU (RTX 5090, 32 GB)

Everything runs at once, with no swapping (measured 2026-10-07: 29.5 of 32.6 GB):
- **Qwen3.8:** 72% of the GPU
- **SAM 3:** the click model plus the proposal model
- **Plate detector:** on the CPU

SAM 3's proposal pass scores 2 text prompts per forward pass instead of 8, with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. That cuts its peak from 13.3 GB to 4.0 GB and still reproduces
the `auto_v1` proposals 97/97.

| Button | Time per frame |
|---|---|
| Canopy vehicles (warm) | 6–9 s |
| Canopy vehicles (first click after a restart, SAM 3 loads) | about 35 s |
| Canopy plates | about 1.5 s per plate |

Measured on task 4, the vehicles button matched 93% of `auto_v1` boxes, and 91% of those have the same class.

`labeler pre-annotate` still swaps whole containers for long batch runs, and always ends back in this mode.

## Functions (CVAT → Models / Automatic annotation)

| Function | Kind | Output |
|---|---|---|
| SAM 3 (canopy) | interactor | mask/polygon from clicks or a box. Text prompts work through the API only. |
| Canopy vehicles | detector | 24 classes from Qwen (SAM 3 boxes), plus `box_quality`, `auto_conf`, `needs_review`, `auto_source` |
| Canopy plates | detector | `plate` with `number`, `code`, `origin`, `category`, `legible`, `plate_colour`, `layout`, `ocr_status`, all read by Qwen3.8 |

## Labelling workflow

```powershell
.\run.ps1 labeler init-projects
.\run.ps1 labeler create-tasks --project vehicles --gate gate1 --date 2026-09-26 --limit 200
.\run.ps1 labeler seed --task 1 --source auto_v1          # existing auto labels as a starting point
.\run.ps1 labeler pre-annotate --task 2 --mode plates     # or run the full pipeline
.\run.ps1 labeler ocr-fill --task 2                       # read plates (Qwen) / text (GLM) you drew by hand
.\run.ps1 labeler pull --task 1 --task 2                  # -> exports/cvat_v1_<ts>.jsonl
python labeler\apply_pull.py exports\cvat_v1_<ts>.jsonl --dry-run   # then without --dry-run
```

### OCR review in CVAT

Each plate or text box carries its reading as attributes, which you edit in the sidebar.

- Plates: filter the job with `attr["ocr_status"] == "low_confidence"` first. On the gold set, `confident` plates are
  fully right 96% of the time (number, code, origin and category: 91/95); `low_confidence` ones only 50%.
  `disputed` / `single` / `agreed` are the older two-reader statuses (`plates_v1` seeds).
- When a reading is correct, set `ocr_status` to `verified`. `ocr-fill` never overwrites `verified` boxes.

## Notes from the local test (2026-10-07)

- **Ports:** CVAT uses 90xx ports because Windows reserves TCP 7981–8680 (Hyper-V/WinNAT), which blocks 8080/8070. All published ports are bound to 127.0.0.1. Nuclio does publish each function on 0.0.0.0 at a random port (for example 7602); Windows Firewall normally blocks inbound traffic to these.
- **`auto_conf`** is an integer percent, because CVAT number attributes must be integers. `apply_pull.py` converts it back to 0–1.
- **SAM 3** runs eager (`TORCHDYNAMO_DISABLE=1`), like the Windows run that produced `auto_v1`.
- **The plate detector** (ONNX) runs on the CPU. onnxruntime-gpu ≥ 1.24 needs CUDA 13, while the torch wheels bring CUDA 12.
- **Text OCR** (`ocr-fill` only): regions under 24×12 px are skipped. GLM output that can't fit its box is dropped and the box is marked `ocr_status=unchecked`.
- **Measured:**

| Check | Result |
|---|---|
| SAM 3 proposals vs stored `auto_v1` proposals | 97/97 identical (IoU ≥ 0.9) |
| `pre-annotate --mode vehicles` vs `auto_v1` | 92% of boxes matched, 92% of those same class |
| GLM-OCR digits on the 250 gold plates | 98.7% |
| Qwen3.8 on gold plates: code / origin / category | 92.0% / 92.8% / 96.2% |
| Qwen3.8 alone on gold plates: digits / full plate / colour | 96.9% / 87.5% / 84.3% (GLM digits + Qwen gave 90.6%; GLM was dropped from plates on 2026-10-07) |
| plate_det_v1 on 20 `plates_v1` boxes | 20/20 found |
| Qwen + SAM 3 click model loaded together | 31.1 of 32.6 GB |

### Label mapping is pre-configured

All three projects (Vehicles, Plates, Text OCR) carry the same full label set: the vehicle classes, `plate`, and
`text` / `sign` / `vehicle_marking`. Each project lists its own labels first. CVAT pairs model and task fields with
equal names, so the "mapping between labels and attributes" dialog is filled in for every canopy model in every task.
Just press **Annotate**.

`labeler init-projects` keeps this in sync. It only ever ADDS labels and attribute values, so existing annotations are
never touched. On export, a task replaces a source for a frame only when the source belongs to its project or when it
has boxes of that kind. So a frame that sits in a Vehicles task and a Text task keeps both sets of labels.
