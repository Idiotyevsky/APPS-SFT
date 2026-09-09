# Minimal Protocol SFT v3 — transition-clean final 7B experiment

- 238 train prefix samples; no quota padding.
- Retains protocol and single-bug repairs from v2.
- Removes one-shot multi-bug direct/reference transitions.
- Adds 23 mutation-grounded real partial trajectories (3 prefixes each).
- Weighted-repair loss parameters remain unchanged from run10b.
