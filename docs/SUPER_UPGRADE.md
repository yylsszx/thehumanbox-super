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
  - [ ] deeper credit assignment (per-action obedience ledger instead of a
        counter; propagate squad rewards into the tabular Q of obedient
        members, not just morale)
  - [ ] tribe postures that actually steer diplomacy (attitude drift for
        Ally, war-weariness coupling for War) instead of reward-shaping only
  - [ ] expose squad intents and tribe postures on the wire frame / UI
        (thought bubbles, tribe cards, a "posture" badge)
  - [ ] macro-intent gating: let the dispatched intent filter the
        exploration pool, not just bias exploitation
  - [ ] inherit linear Q weights to children (currently re-learned)
  - [ ] tune epoch lengths and shaping scales against `headless` sweeps
