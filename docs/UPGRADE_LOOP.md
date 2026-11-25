# The upgrade loop — operating instructions for automated rounds

This repository is advanced by an automated loop (one round per hour). Each
round MUST end with a `git push` to `origin`
(https://github.com/yylsszx/thehumanbox-super).

## The mission

Implement, deepen and tune the **bottom-up multi-layer Q-learning
architecture** described in [SUPER_UPGRADE.md](SUPER_UPGRADE.md) and
[ARCHITECTURE.md](ARCHITECTURE.md#the-learning-hierarchy):

1. **Bottom layer** (individuals): vector states
   `[hunger, thirst, enemy_proximity, ally_count, kin_density,
   leader_present, squad_cohesion, threat_level]` + a linear macro-intent Q
   (`Forage / Drink / Flee / Fight / Follow`) that generalises beyond the
   string-keyed table.
2. **Middle layer** (family/squad Q learner): kin clusters with their own Q
   tables over (shared food, size, threat, rival) × (Gather 聚拢, Disperse
   分散寻食, Migrate 集体迁移). They never control members directly — they
   dispatch intents and shape member rewards (reward shaping), scaled by
   learned confidence.
3. **Top layer** (lineage/tribe Q learner): postures (War 宣战, Ally 结盟,
   Missionary 传教, Research 研发) chosen from (population, technology, war,
   carrying-capacity stress), applied by rewriting the tribe's reward
   baseline and multiplying fight-like choice weights.
4. **Action-space reduction**: members execute macro intents dispatched from
   the middle layer; concrete actions are refined beneath them.
5. **Reward backpropagation**: squad Q evaluates "is the band safe", the
   bottom Q learns "did obeying pay"; obedience that improved the band's
   fate is rewarded back down.

## How to run one round

1. `cd /home/z/my-project/thehumanbox-super` (clone of this repo; remotes:
   `origin` = this repo, `sibling` = yylsszx/thehumanbox (the older fork),
   `upstream` = 1homsi/thehumanbox).
2. `git pull --ff-only origin main` (someone may have pushed).
3. Read the "Round log" at the end of SUPER_UPGRADE.md and the open items
   there. Pick the next open item (or the next sensible increment); keep
   rounds small, compilable and tested.
4. Implement it following the code conventions (see ARCHITECTURE.md):
   modules under `crates/sim-core/src/sim/hierarchy/`, files under ~600
   lines, tests next to the code, deterministic iteration, hash-derived
   rolls, no wall clock, no threads.
5. Verify:
   - `cargo test -p sim-core` (full suite must pass),
   - `cargo clippy -p sim-core --all-targets` (no new warnings),
   - `cargo fmt`.
6. Update the "Round log" in SUPER_UPGRADE.md: what the round did, what is
   still open. Tick the checklist items you completed.
7. Commit with a descriptive message, then **push**:
   `git push origin main`. A round is not finished until it is pushed.

## Guardrails

- Never break save compatibility silently: new save fields need
  `#[serde(default)]` (see `sim/storage/persistence/`).
- Never let an upper layer pick a bottom-layer action — only shape rewards
  and choice weights.
- Never introduce ambient randomness (no `rand` outside the seeded streams,
  no time-of-day); use `hierarchy_roll`.
- Keep the whole test suite green; if a change is too big for one round,
   split it and leave the tree compiling.
