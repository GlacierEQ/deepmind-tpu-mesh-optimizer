#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR=".verification-artifacts"
PLAN="${ARTIFACT_DIR}/mesh-frontier.json"
RECEIPT="${ARTIFACT_DIR}/mesh-frontier.receipt.json"
mkdir -p "${ARTIFACT_DIR}"

python -m compileall -q src tests scripts
python -m unittest discover -s tests -v | tee "${ARTIFACT_DIR}/unittest.txt"
python scripts/mesh_plan.py \
  --chips 16 \
  --reserve-chips 2 \
  --tokens 8192 \
  --hidden-dim 4096 \
  --layers 32 \
  --batch-size 2 \
  --preference balanced \
  --output "${PLAN}" \
  --receipt "${RECEIPT}" \
  | tee "${ARTIFACT_DIR}/mesh-plan.txt"

python - <<'PY'
import hashlib
import json
from pathlib import Path

plan_path = Path('.verification-artifacts/mesh-frontier.json')
receipt_path = Path('.verification-artifacts/mesh-frontier.receipt.json')
plan = json.loads(plan_path.read_text(encoding='utf-8'))
receipt = json.loads(receipt_path.read_text(encoding='utf-8'))

assert plan['evidence_state'] == 'MODELED_MESH_FRONTIER_NOT_TPU_MEASUREMENT'
assert plan['candidate_count'] > plan['frontier_count'] >= 1
assert plan['hardware_model']['reserve_chips'] == 2
assert plan['selected']['active_chips'] == 14
assert len(plan['selected_fingerprint']) == 64
actual = hashlib.sha256(plan_path.read_bytes()).hexdigest()
assert receipt['artifact_sha256'] == actual
assert receipt['selected_fingerprint'] == plan['selected_fingerprint']
assert receipt['verified_state'] == 'DETERMINISTIC_MESH_FRONTIER_EXECUTED'
print(json.dumps({
    'elite_core': 'PASS',
    'candidate_count': plan['candidate_count'],
    'frontier_count': plan['frontier_count'],
    'selected': plan['selected'],
    'artifact_sha256': actual,
}, indent=2))
PY
