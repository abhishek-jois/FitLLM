# Dependency Graph Report

- Files (nodes): **30**
- Dependencies (edges): **77**

## Most connected files

- `fitllm/forward.py` — 16 links — Python module defining logger, _make_resolved_future, ForwardEngine (+1 more).
- `fitllm/model.py` — 16 links — Data model defining logger, shard_model, _estimate_shard_size_gb (+3 more).
- `fitllm/scheduler.py` — 14 links — Python module defining logger, save_shard_with_checksum, load_shard_with_checksum (+2 more).
- `fitllm/probe.py` — 13 links — Python module defining AdaptiveShardProbe.
- `fitllm/backward.py` — 10 links — Python module defining logger, BackwardEngine.
- `fitllm/inference.py` — 9 links — Python module defining logger, validate_draft_tokenizer, merge_lora_into_shards (+5 more).
- `tests/test_integration_comprehensive.py` — 9 links — Comprehensive integration test for FitLLM covering all major components.
- `fitllm/config.py` — 8 links — Configuration / constants defining ShardConfig, TrainingConfig, InferenceConfig.
- `eval/gradient_equivalence.py` — 5 links — FitLLM gradient equivalence evaluation.
- `fitllm/__main__.py` — 5 links — FitLLM CLI entry point.
