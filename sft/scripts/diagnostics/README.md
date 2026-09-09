# SFT diagnostics

Scripts in this directory are exploratory audits, probes, and historical
checkpoint sweeps. The production SFT entrypoints remain one level above:

- prepare_sft.py
- train_protocol_selective.py
- evaluate_sft_agent.py
- build_protocol_dataset*.py

Diagnostic scripts may reference archived runs and the local
experiments/sft/ output store.
