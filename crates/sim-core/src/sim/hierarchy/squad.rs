//! Middle layer of the hierarchy: the family / squad Q learner.
//!
//! A squad is a kin cluster inside one lineage — the people who actually move
//! together (the Zozos and their families). The squad brain never moves a
//! single person directly: it observes a coarse squad state (shared food,
//! size, threat, rival proximity), picks one of three intents — `Gather`
//! (聚拢), `Disperse` (分散寻食), `Migrate` (集体迁移) — with an epsilon-greedy
//! policy over its own tiny Q table, and then *shapes* the members' rewards
//! toward that intent. Members stay fully autonomous learners; the squad only
//! tilts the field they learn on, and the strength of the tilt grows with how
//! well the intent has actually worked (the brain's confidence).

use super::vector::{MacroIntent, MACRO_INTENT_COUNT};

/// Rebuild the kin clusters this often (cheap: buckets + union-find).
pub(crate) const SQUAD_REBUILD_EVERY: u64 = 30;
/// How long one squad decision stays in force.
pub(crate) const SQUAD_EPOCH: u64 = 120;
/// Squads tracked per lineage (largest clusters win, the rest stay unassigned).
pub(crate) const MAX_SQUADS_PER_LINEAGE: usize = 8;
/// Minimum / maximum cluster size worth coordinating.
pub(crate) const SQUAD_MIN_SIZE: usize = 2;
pub(crate) const SQUAD_MAX_SIZE: usize = 16;
/// Grid side (tiles) for the clustering buckets.
const SQUAD_CELL: i32 = 8;

pub(crate) const SQUAD_STATES: usize = 36; // food 3 × size 3 × threat 2 × rival 2
pub(crate) const SQUAD_ACTIONS: usize = 3;

/// The per-action obedience ledger lives on each organism: one slot per
/// macro intent plus one for plain steps (the currency of `Migrate`).
pub(crate) const LEDGER_SLOTS: usize = MACRO_INTENT_COUNT + 1;
/// A member collects the epoch's payback only after this many obeying ticks.
pub(crate) const OBEY_MIN: u32 = 8;

/// The intent a squad brain dispatches to its members.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SquadIntent {
    /// Pull together: members near the centroid are rewarded, crowding is forgiven.
    #[default]
    Gather,
    /// Spread out to feed: foraging is rewarded, the cluster loosens.
    Disperse,
    /// Move the band: steps toward the target are rewarded.
    Migrate,
}

impl SquadIntent {
    #[inline]
    pub fn index(self) -> usize {
        match self {
            SquadIntent::Gather => 0,
            SquadIntent::Disperse => 1,
            SquadIntent::Migrate => 2,
        }
    }

    pub const ALL: [SquadIntent; 3] = [SquadIntent::Gather, SquadIntent::Disperse, SquadIntent::Migrate];

    /// The macro intent this squad intent nudges members toward, if any.
    /// `Migrate` works through plain steps, which carry no macro intent.
    #[inline]
    pub fn macro_intent(self) -> Option<MacroIntent> {
        match self {
            SquadIntent::Gather => Some(MacroIntent::Follow),
            SquadIntent::Disperse => Some(MacroIntent::Forage),
            SquadIntent::Migrate => None,
        }
    }

    /// The ledger slot that counts obedience to this intent.
    #[inline]
    pub fn ledger_slot(self) -> usize {
        match self {
            SquadIntent::Gather => MacroIntent::Follow.index(),
            SquadIntent::Disperse => MacroIntent::Forage.index(),
            SquadIntent::Migrate => MACRO_INTENT_COUNT, // the plain-steps slot
        }
    }
}

/// Which ledger slot one concrete action books itself into, if any.
///
/// Macro-intent actions book into their intent's slot; plain steps book into
/// the trailing slot; anything else (crafting, ritual, odd ids) books nowhere.
#[inline]
pub(crate) fn ledger_slot_of_action(action: usize) -> Option<usize> {
    if let Some(intent) = MacroIntent::of_action(action) {
        return Some(intent.index());
    }
    if action <= 7 {
        return Some(MACRO_INTENT_COUNT);
    }
    None
}

/// Snapshot of the squad at an epoch boundary; the delta between two
/// snapshots is the squad brain's learning signal.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SquadStats {
    pub members: u32,
    pub avg_energy: f32,
    pub avg_health: f32,
    pub avg_food: f32,
    /// `1 - mean distance to centroid / 12`, clamped to `0..=1`.
    pub cohesion: f32,
    pub centroid: (f32, f32),
}

/// One squad's brain: a tiny Q table plus the dispatch state for its members.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SquadBrain {
    pub id: u32,
    /// Population indices at the last rebuild; refreshed every rebuild.
    #[serde(default)]
    pub members: Vec<usize>,
    /// Stable hash of member ids, for matching rebuilt clusters to brains.
    #[serde(default)]
    pub signature: u64,
    /// `q[state][action]`.
    #[serde(default = "default_squad_q")]
    pub q: Vec<[f32; SQUAD_ACTIONS]>,
    #[serde(default)]
    pub last_state: Option<u16>,
    #[serde(default)]
    pub last_action: Option<SquadIntent>,
    /// The intent currently dispatched to members.
    #[serde(default)]
    pub intent: SquadIntent,
    /// Where `Migrate` is heading (the epoch's chosen target).
    #[serde(default)]
    pub target: (i32, i32),
    #[serde(default)]
    pub epoch_stats: SquadStats,
    #[serde(default)]
    pub epoch_started: u64,
    #[serde(default)]
    pub epochs: u32,
    /// `0..=1`: how much better the chosen intent is than the alternatives.
    /// Scales the reward shaping the squad applies to members.
    #[serde(default)]
    pub confidence: f32,
    /// Ticks this epoch in which a member acted along the dispatched intent.
    #[serde(default)]
    pub obeyed_ticks: u32,
    /// Ticks this epoch in which a member acted against it.
    #[serde(default)]
    pub defied_ticks: u32,
}

fn default_squad_q() -> Vec<[f32; SQUAD_ACTIONS]> {
    vec![[0.0; SQUAD_ACTIONS]; SQUAD_STATES]
}

impl Default for SquadBrain {
    fn default() -> Self {
        SquadBrain {
            id: 0,
            members: Vec::new(),
            signature: 0,
            q: default_squad_q(),
            last_state: None,
            last_action: None,
            intent: SquadIntent::Gather,
            target: (0, 0),
            epoch_stats: SquadStats::default(),
            epoch_started: 0,
            epochs: 0,
            confidence: 0.0,
            obeyed_ticks: 0,
            defied_ticks: 0,
        }
    }
}

impl SquadBrain {
    /// Discretise the squad's situation into the state index.
    pub fn state_index(stats: &SquadStats, threat: bool, rival_near: bool) -> u16 {
        let food = if stats.avg_energy < 0.3 {
            0
        } else if stats.avg_energy < 0.6 {
            1
        } else {
            2
        } as u16;
        let size = match stats.members {
            0..=3 => 0,
            4..=8 => 1,
            _ => 2,
        } as u16;
        (food * 3 + size) * 4 + u16::from(threat) * 2 + u16::from(rival_near)
    }

    /// The tabular Q-learning step for one finished epoch.
    pub fn learn(&mut self, state: u16, action: SquadIntent, reward: f32, next_state: u16) {
        let alpha = 0.18;
        let gamma = 0.80;
        let best_next = self.q[next_state as usize]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let old = self.q[state as usize][action.index()];
        let target = reward + gamma * best_next;
        self.q[state as usize][action.index()] = old + alpha * (target - old);
    }

    /// Epsilon-greedy intent choice; the roll comes in as a precomputed
    /// `0..=1` draw so the caller keeps the RNG discipline (deterministic
    /// hash rolls, no ambient randomness).
    pub fn choose(&self, state: u16, roll: f32, epsilon: f32) -> SquadIntent {
        let explore = roll < epsilon;
        let row = &self.q[state as usize];
        if explore {
            // Pure intent roll: split the remaining unit of randomness.
            return match ((roll / epsilon.max(1e-6)) * 3.0) as usize {
                0 => SquadIntent::Gather,
                1 => SquadIntent::Disperse,
                _ => SquadIntent::Migrate,
            };
        }
        let mut best = SquadIntent::Gather;
        let mut best_v = row[0];
        for (i, &v) in row.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best = SquadIntent::ALL[i];
            }
        }
        best
    }

    /// How much better the dispatched intent looks than the worst alternative
    /// in the state it was chosen in — the strength of the reward shaping.
    pub fn confidence_in(state: u16, action: SquadIntent, q: &[[f32; SQUAD_ACTIONS]]) -> f32 {
        let row = &q[state as usize];
        let chosen = row[action.index()];
        let worst = row.iter().copied().fold(f32::INFINITY, f32::min);
        let span = chosen - worst;
        // Saturating: a clear winner shapes at full strength, a coin flip not
        // at all. Negative spans (stale dispatch) shape nothing.
        (span / 0.15).clamp(0.0, 1.0)
    }
}

/// Squads up front: cluster a lineage's living members by proximity.
///
/// Deterministic: bucket by grid cell, union-find over neighbouring cells,
/// keep clusters in ascending order of their smallest member index.
pub(crate) fn cluster_lineage<'a>(
    members: impl Iterator<Item = (usize, &'a crate::organism::organism::Organism)>,
) -> Vec<Vec<usize>> {
    use crate::hashing::FxHashMap;
    use std::collections::BTreeMap;

    // Cell → member indices (BTreeMap: fixed iteration order).
    let mut cells: BTreeMap<(i32, i32), Vec<usize>> = BTreeMap::new();
    let mut position: FxHashMap<usize, (i32, i32)> = FxHashMap::default();
    for (idx, org) in members {
        let cell = (
            (org.x as i32).div_euclid(SQUAD_CELL),
            (org.y as i32).div_euclid(SQUAD_CELL),
        );
        cells.entry(cell).or_default().push(idx);
        position.insert(idx, cell);
    }
    // Union-find over occupied cells that touch (8-neighbourhood).
    let mut parent: FxHashMap<(i32, i32), (i32, i32)> = FxHashMap::default();
    fn find(parent: &mut crate::hashing::FxHashMap<(i32, i32), (i32, i32)>, c: (i32, i32)) -> (i32, i32) {
        let mut root = c;
        while let Some(&p) = parent.get(&root) {
            if p == root {
                break;
            }
            root = p;
        }
        let mut cur = c;
        while let Some(&p) = parent.get(&cur) {
            if p == root {
                break;
            }
            let next = p;
            parent.insert(cur, root);
            cur = next;
        }
        root
    }
    for &c in cells.keys() {
        parent.insert(c, c);
    }
    let mut cell_list: Vec<(i32, i32)> = cells.keys().copied().collect();
    cell_list.sort_unstable();
    for (i, &c) in cell_list.iter().enumerate() {
        for &d in cell_list.iter().skip(i + 1) {
            if (d.0 - c.0).abs() <= 1 && (d.1 - c.1).abs() <= 1 {
                let (rc, rd) = (find(&mut parent, c), find(&mut parent, d));
                if rc != rd {
                    parent.insert(rd, rc);
                }
            }
        }
    }
    // Clusters = members under one root, then size-filtered.
    let mut clusters: BTreeMap<(i32, i32), Vec<usize>> = BTreeMap::new();
    for (&c, idxs) in cells.iter() {
        let root = find(&mut parent, c);
        clusters.entry(root).or_default().extend(idxs.iter().copied());
    }
    let mut out: Vec<Vec<usize>> = clusters
        .into_values()
        .filter(|c| c.len() >= SQUAD_MIN_SIZE)
        .map(|mut c| {
            c.sort_unstable();
            c
        })
        .collect();
    out.sort_by(|a, b| a[0].cmp(&b[0]));
    out
}

/// Stable signature for matching a rebuilt cluster to an existing brain.
pub(crate) fn squad_signature(members: &[usize]) -> u64 {
    // FxHash-style rolling hash over the sorted member ids: the same set in
    // any order produces the same signature.
    let mut sorted: Vec<usize> = members.to_vec();
    sorted.sort_unstable();
    sorted.into_iter().fold(0xcbf2_9ce4_8422_2325u64, |h, m| {
        (h ^ (m as u64 + 1)).wrapping_mul(0x1000_0000_01b3)
    })
}

/// The epoch reward for a squad brain, from the delta between snapshots.
///
/// Shared terms (survival, nourishment) reward every intent; the intent terms
/// credit the specific strategy the squad dispatched.
pub(crate) fn squad_epoch_reward(
    before: &SquadStats,
    after: &SquadStats,
    deaths: u32,
    action: SquadIntent,
    target_before: (i32, i32),
) -> f32 {
    let mut reward = 0.0f32;
    // Nourishment and health moved the whole band.
    reward += (after.avg_energy - before.avg_energy) * 2.0;
    reward += (after.avg_health - before.avg_health) * 2.0;
    reward += (after.avg_food - before.avg_food) * 0.05;
    // Staying alive together is the baseline prize.
    if deaths == 0 {
        reward += 0.05;
    }
    reward -= deaths as f32 * 0.15;
    // Growth counts, weakly: a band that raises children is winning.
    let growth = after.members as i64 - before.members as i64;
    reward += growth as f32 * 0.08;

    match action {
        SquadIntent::Gather => {
            reward += (after.cohesion - before.cohesion) * 0.35;
        }
        SquadIntent::Disperse => {
            // Spreading should pay in food gathered.
            reward += (after.avg_food - before.avg_food) * 0.10;
        }
        SquadIntent::Migrate => {
            // Did the band actually get closer to where it was sent?
            let (tx, ty) = (target_before.0 as f64, target_before.1 as f64);
            let dist_before = (before.centroid.0 as f64 - tx).hypot(before.centroid.1 as f64 - ty);
            let dist_after = (after.centroid.0 as f64 - tx).hypot(after.centroid.1 as f64 - ty);
            reward += (dist_before - dist_after) as f32 * 0.02;
        }
    }
    reward
}

/// Aggregate a set of members into [`SquadStats`].
pub(crate) fn squad_stats_of(sim: &crate::sim::simulation::Simulation, members: &[usize]) -> SquadStats {
    let mut stats = SquadStats::default();
    if members.is_empty() {
        return stats;
    }
    let (mut ex, mut ey) = (0.0f32, 0.0f32);
    let mut alive = 0usize;
    for &i in members {
        let o = &sim.organisms[i];
        if !o.alive {
            continue;
        }
        alive += 1;
        stats.avg_energy += o.energy;
        stats.avg_health += o.health;
        stats.avg_food += o.inv_food as f32;
        ex += o.x;
        ey += o.y;
    }
    if alive == 0 {
        return stats;
    }
    stats.members = alive as u32;
    let n = alive as f32;
    stats.avg_energy /= n;
    stats.avg_health /= n;
    stats.avg_food /= n;
    stats.centroid = (ex / n, ey / n);
    let mut mean_dist = 0.0f32;
    for &i in members {
        let o = &sim.organisms[i];
        if !o.alive {
            continue;
        }
        mean_dist += (o.x - stats.centroid.0).abs() + (o.y - stats.centroid.1).abs();
    }
    mean_dist /= n;
    stats.cohesion = (1.0 - (mean_dist / 12.0).min(1.0)).max(0.0);
    stats
}

/// The vector-state hook for squads: cohesion around the dispatched centroid.
pub(crate) fn squad_cohesion_feature(org: &crate::organism::organism::Organism) -> f32 {
    let (cx, cy) = org.squad_centroid;
    let dist = (org.x - cx).abs() + (org.y - cy).abs();
    (1.0 - (dist / 16.0).min(1.0)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brain() -> SquadBrain {
        SquadBrain::default()
    }

    #[test]
    fn squad_q_learning_pulls_values_toward_reward() {
        let mut b = brain();
        let s = SquadBrain::state_index(
            &SquadStats {
                members: 5,
                avg_energy: 0.2,
                ..Default::default()
            },
            true,
            false,
        );
        let s_next = SquadBrain::state_index(
            &SquadStats {
                members: 5,
                avg_energy: 0.7,
                ..Default::default()
            },
            false,
            false,
        );
        let before = b.q[s as usize][SquadIntent::Gather.index()];
        for _ in 0..50 {
            b.learn(s, SquadIntent::Gather, 1.0, s_next);
        }
        assert!(b.q[s as usize][SquadIntent::Gather.index()] > before);
        assert!(b.q[s as usize][SquadIntent::Gather.index()] > 0.5);
    }

    #[test]
    fn epsilon_zero_always_takes_the_learned_best() {
        let mut b = brain();
        let s = 0u16;
        b.q[0] = [-1.0, 0.5, 0.2];
        for _ in 0..10 {
            assert_eq!(b.choose(s, 0.99, 0.0), SquadIntent::Disperse);
        }
    }

    #[test]
    fn confidence_saturates_on_clear_winners_and_vanishes_on_ties() {
        let mut q = default_squad_q();
        q[0] = [0.3, 0.0, 0.0];
        assert!(SquadBrain::confidence_in(0, SquadIntent::Gather, &q) > 0.99);
        q[1] = [0.1, 0.1, 0.1];
        assert_eq!(SquadBrain::confidence_in(1, SquadIntent::Gather, &q), 0.0);
    }

    #[test]
    fn epoch_reward_credits_gather_for_cohesion() {
        let before = SquadStats {
            members: 4,
            cohesion: 0.2,
            ..Default::default()
        };
        let after = SquadStats {
            members: 4,
            cohesion: 0.8,
            ..Default::default()
        };
        let gather = squad_epoch_reward(&before, &after, 0, SquadIntent::Gather, (0, 0));
        let disperse = squad_epoch_reward(&before, &after, 0, SquadIntent::Disperse, (0, 0));
        assert!(gather > disperse);
    }

    #[test]
    fn ledger_slots_match_the_dispatched_intents() {
        assert_eq!(SquadIntent::Gather.ledger_slot(), MacroIntent::Follow.index());
        assert_eq!(SquadIntent::Disperse.ledger_slot(), MacroIntent::Forage.index());
        assert_eq!(SquadIntent::Migrate.ledger_slot(), MACRO_INTENT_COUNT);
        // The dispatched intent's slot is exactly what the old obedience
        // counter would have counted.
        assert_eq!(ledger_slot_of_action(10), Some(MacroIntent::Follow.index()));
        assert_eq!(ledger_slot_of_action(8), Some(MacroIntent::Forage.index()));
        assert_eq!(ledger_slot_of_action(3), Some(MACRO_INTENT_COUNT));
        assert_eq!(ledger_slot_of_action(600), None);
    }

    #[test]
    fn signature_is_stable_under_permutation() {
        let a = squad_signature(&[3, 1, 4, 1, 5]);
        let b = squad_signature(&[5, 1, 4, 3, 1]);
        assert_eq!(a, b);
        assert_ne!(a, squad_signature(&[3, 1, 4, 1, 6]));
    }
}
