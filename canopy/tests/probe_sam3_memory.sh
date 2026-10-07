#!/usr/bin/env bash
# Measure SAM 3 proposal GPU memory for given prompt-chunk / allocator settings in a throw-away canopy_models
# container (port 9095), with Qwen parked meanwhile. Usage: bash tests/probe_sam3_memory.sh "2" "2 expandable_segments:True" ...
cd /d/canopy-cvat/cvat/canopy || exit 1
curl -s -X POST "localhost:9011/sleep?level=1" -o /dev/null -w "qwen sleep: %{http_code}\n"
for cfg in "$@"; do
  set -- $cfg
  k=$1; alloc=${2:-}
  docker run -d --rm --name canopy_models_probe --gpus all --network canopy-cvat_cvat \
    -e PROPOSE_TEXT_CHUNK="$k" -e PYTORCH_CUDA_ALLOC_CONF="$alloc" -e TORCHDYNAMO_DISABLE=1 -e PORT=8095 \
    -p 127.0.0.1:9095:8095 -v "D:/canopy-evidence/tools:/canopy/tools:ro" \
    -v "C:/Users/Canopy Intelligence/.cache/huggingface/hub/models--facebook--sam3/blobs/9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e:/models/sam3/sam3.pt:ro" \
    canopy-cvat/models:latest >/dev/null
  until curl -sf localhost:9095/health >/dev/null; do sleep 1; done
  python - "$k ${alloc:-default}" <<'EOF'
import base64, json, sqlite3, sys, urllib.request
con = sqlite3.connect("file:D:/canopy-evidence/curated/index.sqlite?mode=ro", uri=True)
for d in [json.loads(l) for l in open("D:/canopy-evidence/curated/autolabel/proposals.jsonl", encoding="utf-8").readlines()[:3]]:
    rel = con.execute("SELECT img_path FROM items WHERE id=?", (d["item"],)).fetchone()[0]
    img = base64.b64encode(open("D:/canopy-evidence/training/" + rel, "rb").read()).decode()
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:9095/propose", data=json.dumps({"image": img}).encode(),
                                                  headers={"Content-Type": "application/json"}), timeout=600).read()
h = json.loads(urllib.request.urlopen("http://127.0.0.1:9095/health").read())
print(f"chunk/alloc {sys.argv[1]}: reserved peak {h['gpu_peak_gb']} GB, allocated peak {h['gpu_peak_alloc_gb']} GB")
EOF
  docker stop canopy_models_probe >/dev/null
done
curl -s -X POST localhost:9011/wake_up -o /dev/null -w "qwen wake: %{http_code}\n"
