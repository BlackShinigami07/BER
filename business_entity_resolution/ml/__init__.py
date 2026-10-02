"""ML stage on top of blocking v1: stage-A CatBoost pair classifier (CATBOOST_PLAN.md), stage-B LLM block re-ranker
(GEMMA_PLAN.md) and the stage-C decision rule.

    python -m business_entity_resolution.ml {gbm-features,gbm,blocks,train,score,decide,export} [args]
"""
