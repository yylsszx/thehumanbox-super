//! The bottom-up multi-layer Q-learning hierarchy.
//!
//! Three layers, three independent Q learners, coupled only through states
//! and rewards — exactly as the architecture notes describe:
//!
//! * **Bottom** (`vector.rs`): every organism keeps a small linear Q over
//!   macro intents, fed by a vector state `[hunger, thirst, enemy_proximity,
//!   ally_count, kin_density, leader_present, squad_cohesion, threat_level]`.
//!   It generalises where the string-keyed table has never been and tilts
//!   action selection.
//! * **Middle** (`squad.rs`): kin clusters get a squad brain with a tiny Q
//!   table over (shared food, size, threat, rival) × (Gather, Disperse,
//!   Migrate). It never moves anyone: it dispatches an intent and *shapes the
//!   members' rewards* toward it, with a strength that grows with the brain's
//!   confidence.
//! * **Top** (`tribe.rs`): each lineage gets a tribe brain over (population,
//!   technology, war, stress) × (War, Ally, Missionary, Research). It sets
//!   the whole tribe's reward baseline and multiplies fight-like choice
//!   weights when it postures for war.
//!
//! Reward flows both ways: the squad's collective fate is its own learning
//! signal, and members whose obedience helped that fate are paid back with
//! morale (the backpropagation the notes ask for).
//!
//! Everything here obeys the reproducibility rules: fixed iteration order
//! (sorted lineage ids, squad ids), hash-derived rolls instead of ambient
//! randomness, no wall clock.

pub mod squad;
pub mod tribe;
pub mod vector;

#[cfg(test)]
mod tests;

pub(crate) use squad::{ledger_slot_of_action, LEDGER_SLOTS, OBEY_MIN};
pub use squad::{SquadBrain, SquadIntent, SquadStats};
pub use tribe::{TribeAction, TribeBrain, TribePolicy, TribeStats};
pub use vector::{LinearQ, MacroIntent, StateVector};

use crate::hashing::FxHashMap;

/// How strongly the linear macro-Q score tilts a concrete action's score.
pub(crate) const MACRO_CHOICE_SCALE: f32 = 0.05;
/// The dispatch nudge a squad intent adds to its aligned macro actions.
pub(crate) const SQUAD_DISPATCH_BOOST: f32 = 0.04;

/// Everything the hierarchy keeps between ticks. Saved with the world.
#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct HierarchyState {
    /// Squad brains per lineage id.
    #[serde(default)]
    pub squads: FxHashMap<String, Vec<SquadBrain>>,
    /// Tribe brains per lineage id.
    #[serde(default)]
    pub tribes: FxHashMap<String, TribeBrain>,
    #[serde(default)]
    pub next_squad_id: u32,
}

/// A deterministic `0..=1` roll: tick, salt and key hashed together, so no
/// ambient randomness enters the simulation stream.
pub(crate) fn hierarchy_roll(tick: u64, salt: u64, key: &str) -> f32 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64
        ^ tick.wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ salt.wrapping_mul(0x94D0_49BB_1331_11EB);
    for &b in key.as_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    // Top 24 bits scaled into 0..1: uniform enough for epsilon rolls and
    // stable across platforms.
    ((h >> 40) as f32) / ((1u64 << 24) as f32)
}

impl crate::organism::organism::Organism {
    /// The middle layer's epoch verdict, propagated down into this member.
    ///
    /// When the band's epoch went well, a member who actually obeyed (per the
    /// ledger: enough ticks spent on the dispatched intent) is paid back twice:
    /// in morale, and — new this round — *in its own tabular Q*: the recent
    /// trace entries that served the intent are pulled a little toward the
    /// squad's credit, so the bottom layer's table itself learns which of its
    /// situations rewarded obedience. Bounded, deterministic, and cheap: the
    /// trace is capped, the credit is clamped, and a convex mix keeps the Q
    /// values from creeping forever.
    pub(crate) fn collect_squad_payback(&mut self, intent: SquadIntent, reward: f32, scale: f32) {
        use crate::organism::organism::QRowExt;
        if !self.alive || reward <= 0.0 {
            return;
        }
        let slot = intent.ledger_slot();
        let obeyed = u32::from(self.squad_ledger[slot]);
        if obeyed < OBEY_MIN {
            return;
        }
        // Morale: the round-1 payback, kept as it was.
        self.hope = (self.hope + 0.05).min(1.0);
        self.joy_ticks = self.joy_ticks.saturating_add(240).min(1_200);
        self.comfort = (self.comfort + 0.05).min(1.0);

        // Tabular backpropagation: the squad's verdict becomes a small target
        // for the aligned actions the member actually chose this epoch.
        let credit = (reward.min(0.5) * scale).clamp(0.0, 0.05);
        if credit <= 0.0 {
            return;
        }
        let mut targets: Vec<(String, u16)> = Vec::with_capacity(8);
        for (state, action) in self.learning_trace.iter().rev() {
            let serves = intent_served(intent, usize::from(*action));
            if !serves || targets.iter().any(|(s, a)| s == state && a == action) {
                continue;
            }
            targets.push((state.clone(), *action));
            if targets.len() >= 8 {
                break;
            }
        }
        if targets.is_empty() {
            return;
        }
        let alpha = 0.06;
        for (state, action) in targets {
            let old = self.q_table.get(&state).map_or(0.0, |row| row.get_q(action));
            let new = old + alpha * (credit - old);
            if let Some(row) = self.q_table.get_mut(&state) {
                row.set_q(action, new);
            } else {
                let mut row = crate::organism::organism::QRow::default();
                row.set_q(action, new);
                self.q_table.insert(state.clone(), row);
            }
            // The credit is a coarse signal by design; let it spread into the
            // generalisation row the same way the tick's own TD step does.
            if let Some(key) = crate::organism::organism::generalization_key(&state) {
                let row = self.q_table.entry(key).or_default();
                let prior = row.get_q(action);
                row.set_q(action, prior + alpha * 0.25 * (credit - prior));
            }
        }
    }

    /// The bottom-layer vector state, from data the sensing pass already has.
    ///
    /// `nearby` holds the population indices the perception scan found;
    /// `threat_dist` is the distance to the nearest hostile beast, if any.
    pub(crate) fn compute_state_vector(
        &self,
        organisms: &[crate::organism::organism::Organism],
        nearby: &[usize],
        threat_dist: Option<f32>,
        tick: u64,
    ) -> StateVector {
        let mut allies = 0u32;
        let mut kin_close = 0u32;
        let mut leader = false;
        for &i in nearby {
            let other = &organisms[i];
            if std::ptr::eq(other, self) || !other.alive {
                continue;
            }
            let d = (other.x - self.x).abs() + (other.y - self.y).abs();
            if other.lineage_id == self.lineage_id {
                if d <= 8.0 {
                    allies += 1;
                }
                if d <= 4.0 {
                    kin_close += 1;
                }
                if d <= 8.0 && (other.is_leader || other.is_elder) {
                    leader = true;
                }
            }
        }
        let squad_cohesion = if tick < self.squad_intent_until {
            squad::squad_cohesion_feature(self)
        } else {
            0.0
        };
        StateVector([
            (1.0 - self.energy).clamp(0.0, 1.0),
            (1.0 - self.hydration).clamp(0.0, 1.0),
            threat_dist
                .map(|d| (1.0 - (d / 12.0).min(1.0)).max(0.0))
                .unwrap_or(0.0),
            (allies as f32 / 8.0).min(1.0),
            (kin_close as f32 / 4.0).min(1.0),
            f32::from(leader),
            squad_cohesion,
            self.fear_level.clamp(0.0, 1.0),
        ])
    }

    /// The hierarchy's contribution to one concrete action's choice score:
    /// the linear macro-Q value, scaled small, with the tribe's attack weight
    /// on `Fight` and the squad's dispatch nudge on its aligned intent.
    pub(crate) fn hierarchy_choice_boost(&self, action: usize, tick: u64) -> f32 {
        let intent = MacroIntent::of_action(action);
        let mut boost = 0.0;
        if let Some(mi) = intent {
            let mut score = self.linear_q.score(mi, self.state_vector) * MACRO_CHOICE_SCALE;
            if mi == MacroIntent::Fight {
                // The top layer's "宣战" coefficient: fight-like choice weights
                // are multiplied across the whole tribe.
                score *= self.tribe_attack_mult;
            }
            boost += score;
        }
        if tick < self.squad_intent_until {
            match self.squad_intent {
                SquadIntent::Gather => {
                    if intent == Some(MacroIntent::Follow) {
                        boost += SQUAD_DISPATCH_BOOST;
                    }
                }
                SquadIntent::Disperse => {
                    if intent == Some(MacroIntent::Forage) {
                        boost += SQUAD_DISPATCH_BOOST;
                    }
                }
                SquadIntent::Migrate => {
                    // Steps toward the band's target carry the dispatch; other
                    // roam-flavoured actions carry half of it.
                    if let Some(toward) =
                        step_toward_target(action, self.squad_target.0, self.squad_target.1, self.x, self.y)
                    {
                        boost += if toward { SQUAD_DISPATCH_BOOST } else { 0.0 };
                    } else if crate::organism::decision_bias::Directive::Roam.aligns(action) {
                        boost += SQUAD_DISPATCH_BOOST * 0.5;
                    }
                }
            }
        }
        boost
    }
}

/// Does this concrete action serve the dispatched squad intent? The same
/// rule the bonds phase uses to book actions into the obedience ledger, so
/// the payback's credit assignment and the booking always agree.
fn intent_served(intent: SquadIntent, action: usize) -> bool {
    match intent {
        SquadIntent::Gather => MacroIntent::of_action(action) == Some(MacroIntent::Follow),
        SquadIntent::Disperse => MacroIntent::of_action(action) == Some(MacroIntent::Forage),
        SquadIntent::Migrate => action <= 7,
    }
}

/// For a step action (`0..=7`): does it move toward `(tx, ty)`?
/// `None` for non-step actions.
fn step_toward_target(action: usize, tx: i32, ty: i32, x: f32, y: f32) -> Option<bool> {
    let (dx, dy) = match action {
        0 => (0, -1),
        1 => (0, 1),
        2 => (-1, 0),
        3 => (1, 0),
        4 => (-1, -1),
        5 => (1, -1),
        6 => (-1, 1),
        7 => (1, 1),
        _ => return None,
    };
    let (gx, gy) = (tx as f32 - x, ty as f32 - y);
    Some((dx as f32) * gx + (dy as f32) * gy > 0.0)
}

impl crate::sim::simulation::Simulation {
    /// The hierarchy's slice of one tick: cluster maintenance, squad epochs,
    /// tribe epochs. Cheap cadences — nothing here walks the whole world
    /// except the rebuild, every 30 ticks.
    pub(crate) fn tick_hierarchy(&mut self) {
        let tick = self.tick_count;
        if tick.is_multiple_of(squad::SQUAD_REBUILD_EVERY) {
            self.hierarchy_rebuild_squads();
        }
        if tick.is_multiple_of(squad::SQUAD_EPOCH) {
            self.hierarchy_squad_epoch();
        }
        if tick.is_multiple_of(tribe::TRIBE_EPOCH) {
            self.hierarchy_tribe_epoch();
        }
    }

    /// Re-cluster every lineage's living members into squads, re-attaching
    /// existing brains by signature or member overlap so learning survives
    /// the churn of births, deaths and wanderers.
    fn hierarchy_rebuild_squads(&mut self) {
        use std::collections::BTreeMap;

        let mut per_lineage: BTreeMap<crate::intern::Sym, Vec<usize>> = BTreeMap::new();
        for (i, o) in self.organisms.iter().enumerate() {
            if o.alive {
                per_lineage.entry(o.lineage_id).or_default().push(i);
            }
        }
        let mut member_of = vec![false; self.organisms.len()];
        let mut next_id = self.hierarchy.next_squad_id;
        let mut fresh: FxHashMap<String, Vec<SquadBrain>> = FxHashMap::default();
        for (lineage, idxs) in per_lineage {
            let mut clusters = squad::cluster_lineage(idxs.iter().map(|&i| (i, &self.organisms[i])));
            // Largest clusters become squads; ties broken by smallest index.
            clusters.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
            clusters.truncate(squad::MAX_SQUADS_PER_LINEAGE);
            // A band is a family circle, not a whole tribe: cap the roster
            // (deterministically, by index order).
            for cluster in &mut clusters {
                cluster.truncate(squad::SQUAD_MAX_SIZE);
            }

            let old = self
                .hierarchy
                .squads
                .swap_remove(lineage.as_str())
                .unwrap_or_default();
            let mut used = vec![false; old.len()];
            let mut brains: Vec<SquadBrain> = Vec::with_capacity(clusters.len());
            for cluster in clusters {
                let sig = squad::squad_signature(&cluster);
                let mut matched = None;
                for (k, b) in old.iter().enumerate() {
                    if !used[k] && b.signature == sig {
                        matched = Some(k);
                        break;
                    }
                }
                if matched.is_none() {
                    for (k, b) in old.iter().enumerate() {
                        if used[k] {
                            continue;
                        }
                        let inter = b.members.iter().filter(|m| cluster.contains(m)).count();
                        let floor = b.members.len().min(cluster.len()).max(1) * 2 / 5;
                        if inter >= floor {
                            matched = Some(k);
                            break;
                        }
                    }
                }
                let mut brain = match matched {
                    Some(k) => {
                        used[k] = true;
                        old[k].clone()
                    }
                    None => {
                        next_id = next_id.wrapping_add(1);
                        SquadBrain {
                            id: next_id,
                            ..Default::default()
                        }
                    }
                };
                brain.members = cluster.clone();
                brain.signature = sig;
                for &m in &cluster {
                    member_of[m] = true;
                }
                brains.push(brain);
            }
            fresh.insert(lineage.to_string(), brains);
        }
        self.hierarchy.next_squad_id = next_id;
        self.hierarchy.squads = fresh;

        // Dispatch (or expire) the runtime squad fields on every organism.
        for brains in self.hierarchy.squads.values() {
            for brain in brains {
                let stats = squad::squad_stats_of(self, &brain.members);
                let until = brain.epoch_started + squad::SQUAD_EPOCH;
                for &m in &brain.members {
                    let o = &mut self.organisms[m];
                    o.squad_intent = brain.intent;
                    o.squad_intent_until = o.squad_intent_until.max(until);
                    o.squad_centroid = stats.centroid;
                    o.squad_target = brain.target;
                    o.squad_confidence = brain.confidence;
                }
            }
        }
        for (i, in_squad) in member_of.into_iter().enumerate() {
            if !in_squad && self.organisms[i].squad_intent_until > 0 {
                self.organisms[i].squad_intent_until = 0;
                self.organisms[i].squad_ledger = [0; LEDGER_SLOTS];
            }
        }
    }

    /// Every squad epoch: learn from the finished epoch, pay back obedient
    /// members when the band fared well, then choose and dispatch the next
    /// intent.
    fn hierarchy_squad_epoch(&mut self) {
        let tick = self.tick_count;
        let mut lineages: Vec<String> = self.hierarchy.squads.keys().cloned().collect();
        lineages.sort();

        // Rival centroids once per epoch: nearest foreign lineage centre.
        let mut centroids: Vec<(String, (f32, f32))> = Vec::new();
        {
            use std::collections::BTreeMap;
            let mut acc: BTreeMap<crate::intern::Sym, (f32, f32, u32)> = BTreeMap::new();
            for o in self.organisms.iter().filter(|o| o.alive) {
                let e = acc.entry(o.lineage_id).or_insert((0.0, 0.0, 0));
                e.0 += o.x;
                e.1 += o.y;
                e.2 += 1;
            }
            for (lid, (sx, sy, n)) in acc {
                if n > 0 {
                    centroids.push((lid.to_string(), (sx / n as f32, sy / n as f32)));
                }
            }
        }

        for lineage in lineages {
            let mut brains = self.hierarchy.squads.swap_remove(&lineage).unwrap_or_default();
            brains.sort_by_key(|b| b.id);
            let rival = centroids
                .iter()
                .filter(|(lid, _)| *lid != lineage)
                .map(|(_, c)| c)
                .min_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                })
                .copied();

            brains.retain(|brain| brain.members.iter().any(|&m| self.organisms[m].alive));

            for brain in &mut brains {
                let after = squad::squad_stats_of(self, &brain.members);
                let alive_now = brain.members.iter().filter(|&&m| self.organisms[m].alive).count() as u32;
                let deaths = brain.epoch_stats.members.saturating_sub(alive_now);
                let threat = brain
                    .members
                    .iter()
                    .any(|&m| self.organisms[m].alive && self.organisms[m].fear_level > 0.25);
                let rival_near = rival.is_some_and(|(rx, ry)| {
                    (after.centroid.0 - rx).abs() + (after.centroid.1 - ry).abs() <= 12.0
                });
                let state = SquadBrain::state_index(&after, threat, rival_near);

                if let Some(last_state) = brain.last_state {
                    let last_action = brain.last_action.unwrap_or(brain.intent);
                    let reward = squad::squad_epoch_reward(
                        &brain.epoch_stats,
                        &after,
                        deaths,
                        last_action,
                        brain.target,
                    );
                    brain.learn(last_state, last_action, reward, state);
                    // Backpropagation: obedience that helped the band is paid
                    // back — in morale, and now in the obedient members' own
                    // tabular Q (per-action credit through the ledger).
                    if reward > 0.05 {
                        let scale = 0.25 + 0.75 * brain.confidence;
                        for &m in &brain.members {
                            self.organisms[m].collect_squad_payback(brain.intent, reward, scale);
                        }
                    }
                    for &m in &brain.members {
                        self.organisms[m].squad_ledger = [0; LEDGER_SLOTS];
                    }
                }

                let roll = hierarchy_roll(tick, u64::from(brain.id) ^ 0x5AED, &lineage);
                let epsilon = (0.20 - brain.epochs as f32 * 0.01).max(0.05);
                let action = brain.choose(state, roll, epsilon);
                brain.confidence = SquadBrain::confidence_in(state, action, &brain.q);
                brain.intent = action;
                if action == SquadIntent::Migrate {
                    brain.target = self.pick_migration_target(&after.centroid, rival);
                }
                brain.epoch_stats = after;
                brain.epoch_started = tick;
                brain.last_state = Some(state);
                brain.last_action = Some(action);
                brain.epochs += 1;
            }

            // Dispatch the (possibly new) intent to members.
            for brain in &brains {
                let until = brain.epoch_started + squad::SQUAD_EPOCH;
                for &m in &brain.members {
                    let o = &mut self.organisms[m];
                    o.squad_intent = brain.intent;
                    o.squad_intent_until = until;
                    o.squad_centroid = brain.epoch_stats.centroid;
                    o.squad_target = brain.target;
                    o.squad_confidence = brain.confidence;
                }
            }
            self.hierarchy.squads.insert(lineage, brains);
        }
    }

    /// A migration target: the best-scoring direction at marching distance,
    /// preferring food and water and the far side of a rival.
    fn pick_migration_target(&self, centroid: &(f32, f32), rival: Option<(f32, f32)>) -> (i32, i32) {
        use crate::world::grid::{HEIGHT, WIDTH};
        use crate::world::tiles::Tile;

        const DIRS: [(i32, i32); 8] = [
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ];
        let (cx, cy) = (centroid.0 as i32, centroid.1 as i32);
        let mut best = (cx, cy);
        let mut best_score = f32::MIN;
        for (dx, dy) in DIRS {
            let tx = (cx + dx * 20).clamp(4, WIDTH as i32 - 5);
            let ty = (cy + dy * 20).clamp(4, HEIGHT as i32 - 5);
            let mut score = 0.0f32;
            let mut sy = -6;
            while sy <= 6 {
                let mut sx = -6;
                while sx <= 6 {
                    let tile = self.grid.get(tx + sx, ty + sy);
                    if tile == Tile::Food {
                        score += 1.0;
                    } else if tile == Tile::Water {
                        score += 0.3;
                    }
                    sx += 2;
                }
                sy += 2;
            }
            if let Some((rx, ry)) = rival {
                let away = (dx as f32) * (tx as f32 - rx) + (dy as f32) * (ty as f32 - ry);
                if away > 0.0 {
                    score += 3.0;
                }
            }
            if score > best_score {
                best_score = score;
                best = (tx, ty);
            }
        }
        best
    }

    /// Every tribe epoch: learn from the tribe's fate, choose a posture,
    /// broadcast its reward baseline and attack weight to every member.
    fn hierarchy_tribe_epoch(&mut self) {
        use std::collections::BTreeMap;

        let tick = self.tick_count;
        let mut acc: BTreeMap<crate::intern::Sym, (u32, f32)> = BTreeMap::new();
        for o in self.organisms.iter().filter(|o| o.alive) {
            let e = acc.entry(o.lineage_id).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += o.energy;
        }
        let mut order: Vec<(String, TribeStats)> = Vec::new();
        for (lid, (population, energy_sum)) in acc {
            if population < 4 {
                continue;
            }
            let era = self
                .lineage_eras
                .get(&lid)
                .copied()
                .unwrap_or(crate::sim::civ::progress::eras::LADDER[0]);
            let era_index = crate::sim::civ::progress::eras::LADDER
                .iter()
                .position(|&e| e == era)
                .unwrap_or(0) as u8;
            let name = self
                .lineage_names
                .get(lid.as_str())
                .cloned()
                .unwrap_or_else(|| lid.to_string());
            let at_war = self.battles.iter().any(|b| {
                b.ended_tick.is_none()
                    && (b
                        .attackers
                        .iter()
                        .any(|a| a == &name || a.as_str() == lid.as_str())
                        || b.defenders
                            .iter()
                            .any(|d| d == &name || d.as_str() == lid.as_str()))
            });
            order.push((
                lid.to_string(),
                TribeStats {
                    population,
                    avg_energy: energy_sum / population as f32,
                    era_level: (era_index / 15).min(2),
                    unrest: self.lineage_unrest.get(lid.as_str()).copied().unwrap_or(0.0),
                    at_war,
                    progress: u32::from(era_index),
                },
            ));
        }

        for (lineage, stats) in order {
            let stressed = stats.avg_energy < 0.4;
            let state = TribeBrain::state_index(&stats, stressed);
            let brain = self.hierarchy.tribes.entry(lineage.clone()).or_default();
            if let Some(last_state) = brain.last_state {
                let reward = tribe::tribe_epoch_reward(&brain.epoch_stats, &stats);
                let last_action = brain.action;
                brain.learn(last_state, last_action, reward, state);
            }
            let roll = hierarchy_roll(tick, 0x7AB1, &lineage);
            let epsilon = (0.25 - brain.epochs as f32 * 0.01).max(0.05);
            let action = brain.choose(state, roll, epsilon);
            brain.action = action;
            brain.action_started = tick;
            brain.epoch_stats = stats;
            brain.last_state = Some(state);
            brain.epochs += 1;
            let policy = TribePolicy::of(action);

            let mut members = 0usize;
            for o in self
                .organisms
                .iter_mut()
                .filter(|o| o.alive && o.lineage_id.as_str() == lineage.as_str())
            {
                o.tribe_attack_mult = policy.attack_mult;
                if action == TribeAction::Missionary {
                    o.piety = (o.piety + 0.01).min(1.0);
                }
                members += 1;
            }

            let name = self
                .lineage_names
                .get(lineage.as_str())
                .cloned()
                .unwrap_or_else(|| lineage.chars().take(8).collect());
            let detail = match action {
                TribeAction::War => "beats the drums of war: fight weighs heavier for everyone",
                TribeAction::Ally => "seeks company and open hands",
                TribeAction::Missionary => "sends its believers outward with the word",
                TribeAction::Research => "turns its eyes to craft and discovery",
            };
            crate::sim::world_events::push_event(&mut self.events, tick, "tribe_posture", &name, detail);
            let _ = members;
        }
    }

    /// The hierarchy's contribution to one member's final shaped reward:
    /// the tribe's baseline and posture multiplier, plus the squad intent's
    /// shaping. Called from the bonds phase.
    pub(crate) fn hierarchy_member_shaping(
        &self,
        lineage: &str,
        idx: usize,
        action: usize,
        ox: f32,
        oy: f32,
    ) -> f32 {
        let mut delta = 0.0f32;
        if let Some(brain) = self.hierarchy.tribes.get(lineage) {
            let policy = TribePolicy::of(brain.action);
            delta += policy.reward_baseline;
            let skilled = crate::organism::skills::Skill::of_action(action).is_some();
            let social = crate::organism::decision_bias::Directive::Socialize.aligns(action);
            // Multiply only the positive part: a posture must not make
            // punishments cheaper.
            let positive = reward_positive_part(self, idx);
            delta += (policy.reward_mult(action, skilled, social) - 1.0) * positive;
        }
        let o = &self.organisms[idx];
        if self.tick_count < o.squad_intent_until {
            let confidence = 0.25 + 0.75 * o.squad_confidence;
            match o.squad_intent {
                SquadIntent::Gather => {
                    let d = (ox - o.squad_centroid.0).abs() + (oy - o.squad_centroid.1).abs();
                    let near = (1.0 - (d / 10.0).min(1.0)).max(0.0);
                    delta += 0.006 * confidence * near;
                }
                SquadIntent::Disperse => {
                    if MacroIntent::of_action(action) == Some(MacroIntent::Forage) {
                        delta += 0.005 * confidence;
                    }
                }
                SquadIntent::Migrate => {
                    let (tx, ty) = (o.squad_target.0 as f32, o.squad_target.1 as f32);
                    let before = (o.prev_x - tx).abs() + (o.prev_y - ty).abs();
                    let now = (ox - tx).abs() + (oy - ty).abs();
                    if now < before {
                        delta += 0.004 * confidence;
                    }
                }
            }
        }
        delta
    }
}

/// The positive share of this tick's reward so far, for posture multipliers.
fn reward_positive_part(sim: &crate::sim::simulation::Simulation, idx: usize) -> f32 {
    // The bonds phase has already folded most signals into its local `reward`
    // variable; the part the hierarchy can see from outside is the body's own
    // fare this tick (health and vitals), which is what a posture should
    // amplify. Kept deliberately small and bounded.
    let o = &sim.organisms[idx];
    0.02 * o.health.clamp(0.0, 1.0)
}
