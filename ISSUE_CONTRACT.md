# Issue Contract — `deepmind-tpu-mesh-optimizer`

## Pain
Long-context mesh steps stall when ICI transfer is exposed.

## Claim
Ring optimizer returns transfer/compute/exposed metrics and status.

## Proof
```bash
python3 job-app/helix/proofs/proof_tpu_mesh.py
```

## Done when
Proof exits 0. Architecture (strand/integrity/helix) is **not** a substitute for this proof.

## Anti-claim
Not production TPU runtime.
