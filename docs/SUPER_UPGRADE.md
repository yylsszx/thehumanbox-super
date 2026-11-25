# The Super Upgrade: a bottom-up multi-layer Q-learning architecture

This fork upgrades the classic artificial-life engine with a three-layer
hierarchy of independent Q learners. Each layer has its own Q table; lower
layers' changes feed upper layers' states, and upper layers' decisions reach
back down as reward shaping — never as direct control.

```
┌────────────────────────────────────────────────────────────────────┐
│  TOP — lineage / tribe Q learner (meso → macro)                    │
│  state: population, technology, war, carrying-capacity stress       │
│  actions: War (宣战) · Ally (结盟) · Missionary (传教) · Research    │
│  couples out: global reward baseline + attack-weight coefficient    │
├────────────────────────────────────────────────────────────────────┤
│  MIDDLE — family / squad Q learner (micro → meso)                   │
│  entity: kin cluster (a Zozo and his family)                       │
│  state: shared food, size, threat, rival proximity                  │
│  actions: Gather (聚拢) · Disperse (分散寻食) · Migrate (集体迁移)   │
│  couples out: per-member reward shaping, scaled by confidence       │
├────────────────────────────────────────────────────────────────────┤
│  BOTTOM — individual Q learner (already existed, now upgraded)      │
│  state: string table (kept) + vector [hunger, thirst, enemy_dist,   │
│         ally_count, kin_density, leader_present, cohesion, threat]  │
│  actions: the full action space, refined under 5 macro intents      │
│  new: linear macro-intent Q generalises across unseen situations    │
└────────────────────────────────────────────────────────────────────┘
```

## What each layer is

1. **Bottom (existing, upgraded)** — the survival learner every organism
   already runs (tabular Q over perception strings, replacing traces,
   curiosity/fear shaping — all untouched). The upgrade:
   - states gain `ally_count` and `leader_present`-style features through the
     new `StateVector` (`sim/hierarchy/vector.rs`);
   - a per-organism `LinearQ` scores the five **macro intents**
     (`Forage`, `Drink`, `Flee`, `Fight`, `Follow`) from that vector and adds
     a small bounded boost to action selection — this is the
     "vector replaces string" generalisation;
   - the same temporal-difference error that trains the table also trains
     the linear weights (`organism_tick::learn_from_the_tick`).

2. **Middle (new)** — `sim/hierarchy/squad.rs`. Kin clusters (rebuilt every
   30 ticks by grid-bucket union-find, matched back to their brains by
   signature/overlap) run a tiny Q table (36 states × 3 actions). Every 120
   ticks an epoch ends: the band's fate (energy, health, deaths, cohesion,
   food, migration progress) becomes the epoch reward, the Q table learns,
   and an epsilon-greedy intent is dispatched. **The squad never controls
   anyone** — it shapes rewards (`bonds.rs`):
   - `Gather`: proximity to the centroid pays, crowding is forgiven;
   - `Disperse`: foraging pays;
   - `Migrate`: steps toward the band's target pay (target picked by a food/
     water/rival scan).
   Shaping strength scales with the brain's **confidence** (how much better
   the chosen intent looks than the alternatives).

3. **Top (new)** — `sim/hierarchy/tribe.rs`. Every lineage with 4+ living
   members runs a Q table (48 states × 4 postures), epoch 600 ticks, reward
   = population change + era progress + nourishment − unrest. The posture:
   - `War`: reward baseline −0.0004/tick for everyone, Fight-macro choice
     weights ×1.6, combat rewards ×1.25;
   - `Ally`: social/trade rewards ×1.2, small positive baseline;
   - `Missionary`: socialising-near-strangers rewards ×1.3, piety grows;
   - `Research`: skilled-work rewards ×1.25, small positive baseline.

## The three coupling mechanisms from the design notes

- **Action-space reduction**: the middle layer dispatches one of a handful of
  intents; the individual refines it into a concrete action out of the
  6144-id space. Mapping rides the existing `Directive` families
  (`MacroIntent::of_action`), so no second gating system exists.
- **Reward propagation**: the squad Q table asks "is the band safe?", the
  bottom Q table learns "did following the band's call pay?". When a band
  fares well, obedient members (8+ aligned actions in the epoch) are paid
  back in hope/joy/comfort at the epoch boundary.
- **Upper decisions react on lower rewards**: tribe → reward baseline and
  attack coefficient; squad → intent-shaped member rewards. Nothing above
  the bottom layer ever picks a bottom-layer action.

## Reproducibility discipline

Fixed iteration order everywhere (sorted lineage ids, squad ids), hash-derived
epsilon rolls (`hierarchy_roll`, no ambient RNG), no wall clock, no threads.
Saves carry the whole hierarchy (`HierarchyState` in the world save, `linear_q`
per organism); old saves load with defaults and re-learn.

## Round log

- **Round 1 (this commit)** — the three layers exist and are wired end to
  end: vector states + linear macro-Q on every organism; squad clustering,
  epochs, dispatch and reward shaping; tribe postures with baseline and
  attack weight; obedience backpropagation; persistence; 23 new tests.
  Untouched for future rounds:
  - [x] deeper credit assignment (per-action obedience ledger instead of a
        counter; propagate squad rewards into the tabular Q of obedient
        members, not just morale) — done in round 2
  - [x] tribe postures that actually steer diplomacy (attitude drift for
        Ally, war-weariness coupling for War) instead of reward-shaping
        only — done in round 3
  - [x] expose squad intents and tribe postures on the wire frame / UI
        (thought bubbles, tribe cards, a "posture" badge) — done in round 4
  - [x] macro-intent gating: let the dispatched intent filter the
        exploration pool, not just bias exploitation — done in round 5
  - [x] inherit linear Q weights to children (currently re-learned) — done
        in round 6
  - [ ] tune epoch lengths and shaping scales against `headless` sweeps

- **Round 2** — deeper credit assignment. The obedience counter became a
  **per-action ledger** on every member: while a squad intent is in force,
  each action books itself into its macro-intent slot
  (`ledger_slot_of_action` — Forage / Drink / Flee / Fight / Follow, plus a
  trailing slot for plain steps, the currency of `Migrate`). The dispatched
  intent's slot counts exactly what the old counter counted; the other
  slots now record defiance and context for free. At the epoch boundary, a
  band that fared well pays its obedient members twice:
  - the round-1 morale payback (hope / joy / comfort), unchanged;
  - new: a **tabular backpropagation** — `collect_squad_payback` walks the
    member's learning trace (newest first, deduplicated, capped at 8), and
    pulls the Q rows of the entries that served the dispatched intent
    toward the squad's verdict: `credit = reward × confidence scale`,
    clamped to 0.05, applied as a convex mix (α = 0.06) so repeated epochs
    converge instead of creeping, with the coarse generalisation row
    sharing a quarter share, exactly like the tick's own TD step.
  Defiance books into its own slots and is credited nowhere. The credit is
  gated on the ledger vouching (`OBEY_MIN = 8` obeying ticks) and nothing
  above the bottom layer ever picks a bottom-layer action — the squad only
  writes into the table its members learn from. Runtime-only (no save
  format change); 4 new tests, 27 hierarchy tests total; fmt + clippy clean.
  The long dispersal integration test exposed a latent overflow (u8 tool
  counter in `salt_meat`), fixed with `saturating_add`; the rest of the
  1370-test suite is green.

- **Round 3** — tribe postures now steer actual diplomacy, not just rewards.
  - **Ally → attitude drift**: an `Ally` epoch warms every member's attitude
    toward every living foreign lineage by `ALLY_DRIFT` (0.05/epoch,
    BTreeMap entries, own lineage excluded, capped at 1.0). `form_alliance`
    needs attitude > 0.2, so four Ally epochs make forging alliances,
    envoys, trade and truces genuinely reachable — the posture changes what
    diplomacy is possible, not merely what it pays.
  - **War → war-weariness coupling**: `TribeBrain.war_epochs` counts
    consecutive epochs spent actually fighting under a `War` posture
    (posturing without a war is free). It couples two ways:
    the epoch reward loses `war_weariness` (0.02/epoch, capped 0.30) so the
    Q table can learn to turn away from an endless war; and the broadcast
    attack weight fades `1.6 → 1.48 → …` (`effective_attack_mult`), so a
    long war softens the whole tribe's fight bias instead of switching it
    off. Peace or a posture change resets the clock.
  - Save format: one new `#[serde(default)]` field (`war_epochs`); old
    saves load and re-learn. 5 new tests (32 hierarchy total); fmt + clippy
    clean.
- **Round 4** — the hierarchy is on the wire: clients can now *see* what the
  middle and top layers are thinking, with zero simulation behaviour
  changed and zero save-format changed.
  - **Thought bubbles**: both dispatch sites (rebuild and epoch) now run the
    dispatched intent through `Organism::think` — "gathering with the band",
    "scattering to forage", "moving with the band" — so members visibly
    carry their band's intent in the existing thought channel. `think`'s own
    semantics do the right thing for free: urgent needs keep their bubble
    (`is_need_thought`, now crate-test-visible, pins the contract that the
    three bubble phrases never collide with the need list), re-choosing the
    same intent dedupes, and `thought_dirty` ships it on the next hot delta.
  - **Tribe posture badge**: a new cold-frame payload `tribe_posture` maps
    lineage → `{posture, since, epochs, war_epochs, attack_mult}` straight
    from each `TribeBrain`; `attack_mult` is the war-weariness-faded weight
    the posture actually broadcasts (`effective_attack_mult`), 1.0 outside
    `War` — the tribe card can badge and fade it.
  - **Squad panel**: a companion cold-frame payload `squad_intents` maps
    lineage → `[{id, intent, members, ids, confidence, target}]` from each
    `SquadBrain`; `ids` carries alive members' organism ids so the client
    badges members without a join, dead members (stale between rebuilds)
    are skipped, and squads with nobody left are not listed. Both payloads
    are sorted (lineage ids, squad ids) and read-only.
  - Wire names are a contract: `SquadIntent::name` ("gather" / "disperse" /
    "migrate") and `TribeAction::name` ("war" / "ally" / "missionary" /
    "research") are lowercase, stable, and pinned by tests.
  - No new save fields (the payloads read runtime + persisted state as it
    is); 5 new tests (35 hierarchy total); fmt + clippy clean.

- **Round 5** — the dispatched intent now gates the exploration pool itself,
  not just the exploitation bias: when a squad's intent is in force, a
  member's epsilon-probing looks first at the actions that serve the band
  instead of the whole action space.
  - **The gate** (`Organism::squad_gate_blocks`, applied inside
    `choose_action`'s exploration branch, next to the existing player-
    directive filter): with a confident dispatch in force and no urgent
    need, the pool shrinks to `intent_served` actions — Gather → Follow
    moves, Disperse → Forage moves, Migrate → plain steps — reusing the
    exact rule the obedience ledger books with, so probing, booking and
    payback always agree. The bottom layer still chooses; the middle layer
    only narrows where experimenting looks. The player's own directive
    still outranks the band.
  - **Three releases, not a hard cage**: an empty gate (or no intent)
    falls back to the full pool; starving, parched or hurt members keep
    everything (the thresholds mirror the tick's own `needs_ok` /
    recovery checks); and — learned from the first cut breaking a 900-tick
    territory test — a fresh or unproven dispatch leaves the pool whole:
    the gate needs the band brain's confidence ≥ 0.20
    (`GATE_CONFIDENCE_MIN`), the same confidence that scales the reward
    shaping. Unproven intents get explored; proven ones get executed.
  - No save-format change (`squad_confidence` is already runtime state);
    2 new tests (37 hierarchy total): the full gate matrix per intent, and
    the urgency / low-confidence releases. The gate-flavoured world shifts
    one long-range aggregate threshold by a hair (28 vs 30 claimed tiles),
    resolved by the confidence gate itself rather than by loosening the
    test. Full suite green (1383), clippy + fmt clean.

- **Round 6** — the bottom layer's linear macro-Q is now a birthright. A
  child used to inherit its mother's tabular Q (with drift) and her
  memories, vocabulary and lessons, but the linear macro-Q — everything her
  line had ever learned about when foraging, drinking, fleeing, fighting or
  following pays — was wiped to zeros at birth, so every child restarted
  that layer from scratch and the lineage's macro-intuitions never
  compounded across generations.
  - `try_reproduce` now copies the mother's `LinearQ.w` into the child with
    the same gentle drift the tabular Q gets (20% of weights nudged by up
    to ±0.03): most weights pass through untouched, a few drift, so the
    inherited instincts can still move on in a child instead of freezing.
  - The seeded-reproduction fixture's captured expectations were
    recaptured (the inherit block consumes the rng after the tabular copy,
    before the attribute rolls, as its comment anticipates); a new test
    pins the contract — same shape, every weight within ±0.03 of the
    mother's, at least three quarters identical, at least one drifted, and
    none fallen back to a fresh zero.
  - No save-format change. Full suite green (1384, all 133 long-range
    integration tests re-run against the shifted birth rng), clippy + fmt
    clean.
