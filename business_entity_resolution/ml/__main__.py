"""Single runner for the ML stage (CATBOOST_PLAN.md stage A, GEMMA_PLAN.md stage B, stage C):

    python -m business_entity_resolution.ml gbm-features build --artifacts-dir artifacts                  # WP-2 fit matrix
    python -m business_entity_resolution.ml gbm fit     --artifacts-dir artifacts                          # WP-3 (GPU)
    python -m business_entity_resolution.ml gbm predict --artifacts-dir artifacts --split valid|test       # WP-3
    python -m business_entity_resolution.ml gbm report  --artifacts-dir artifacts                          # WP-4

    python -m business_entity_resolution.ml blocks  --artifacts-dir artifacts --n-train 400000   # WP-B (CPU)
    python -m business_entity_resolution.ml train   --train-dir ... --dev-dir ... [--load-4bit]  # WP-C (GPU; torchrun for DDP)
    python -m business_entity_resolution.ml score   --model-dir ... --blocks-root ...            # WP-D (GPU)
    python -m business_entity_resolution.ml decide  --artifacts-dir artifacts                    # stage C rule on valid_llm
    python -m business_entity_resolution.ml export  --artifacts-dir artifacts --test-dir dataset/test
"""
import importlib
import sys

COMMANDS = {"gbm-features": "gbm_features", "gbm": "gbm", "blocks": "blocks", "blocks-aug": "blocks_aug", "uniq-flags": "postprocess", "dense-delta": "dense_delta", "train": "llm.train", "score": "llm.score", "decide": "decide", "export": "export_results"}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        raise SystemExit(2)
    mod = importlib.import_module(f"business_entity_resolution.ml.{COMMANDS[sys.argv[1]]}")
    mod.main(sys.argv[2:])


if __name__ == "__main__":
    main()
