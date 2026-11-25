//! Top layer of the hierarchy: the lineage / tribe Q learner.
//!
//! A tribe is what the culture, language and myth already bind together — a
//! lineage. The tribe brain never commands a person. It observes a coarse
//! tribal state (population band, technology band, war, capacity stress),
//! picks one of four postures — `War` (宣战), `Ally` (结盟), `Missionary`
//! (传教), `Research` (研发) — and then rewrites the *reward baseline* every
//! member learns against, plus a posture-specific shaping: a war posture
//! multiplies the weight of fight-like choices across the whole tribe, an
//! alliance posture rewards trade and company, missionaries are paid for
//! socialising with strangers, research pays for skilled work.
//!
//! The learning signal is the tribe's own fate across an epoch: population
//! change, era progress, unrest and the balance of its wars.

pub(crate) const TRIBE_EPOCH: u64 = 600;
pub(crate) const TRIBE_STATES: usize = 48; // pop 4 × tech 3 × war 2 × stress 2
pub(crate) const TRIBE_ACTIONS: usize = 4;

/// One tribe-epoch of `Ally` posture warms every foreign attitude by this
/// much (reaching `form_alliance`'s 0.2 threshold in four epochs from 0).
pub(crate) const ALLY_DRIFT: f32 = 0.05;
/// Weariness the tribe's Q table feels per consecutive epoch spent actually
/// fighting under a `War` posture (capped).
pub(crate) const WEARINESS_PER_EPOCH: f32 = 0.02;
pub(crate) const WEARINESS_CAP: f32 = 0.30;

/// The posture a tribe brain broadcasts to its people.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TribeAction {
    #[default]
    War,
    Ally,
    Missionary,
    Research,
}

impl TribeAction {
    #[inline]
    pub fn index(self) -> usize {
        match self {
            TribeAction::War => 0,
            TribeAction::Ally => 1,
            TribeAction::Missionary => 2,
            TribeAction::Research => 3,
        }
    }

    pub const ALL: [TribeAction; 4] = [
        TribeAction::War,
        TribeAction::Ally,
        TribeAction::Missionary,
        TribeAction::Research,
    ];

    /// The wire name of this posture: what the frame's `tribe_posture`
    /// payload puts on the tribe card's badge. Lowercase, stable, never
    /// renamed — old clients keep rendering old saves.
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            TribeAction::War => "war",
            TribeAction::Ally => "ally",
            TribeAction::Missionary => "missionary",
            TribeAction::Research => "research",
        }
    }
}

/// The tribe-level snapshot an epoch is judged against.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TribeStats {
    pub population: u32,
    pub avg_energy: f32,
    pub era_level: u8,
    pub unrest: f32,
    pub at_war: bool,
    /// Milestones / discoveries the tribe holds.
    pub progress: u32,
}

/// One tribe's brain.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TribeBrain {
    #[serde(default = "default_tribe_q")]
    pub q: Vec<[f32; TRIBE_ACTIONS]>,
    #[serde(default)]
    pub last_state: Option<u8>,
    /// The posture currently broadcast.
    #[serde(default)]
    pub action: TribeAction,
    #[serde(default)]
    pub action_started: u64,
    #[serde(default)]
    pub epoch_stats: TribeStats,
    #[serde(default)]
    pub epochs: u32,
    /// Consecutive epochs the tribe has spent actually fighting under a `War`
    /// posture. Drives the war-weariness coupling: the Q table's reward and
    /// the broadcast attack weight both fade as it grows.
    #[serde(default)]
    pub war_epochs: u32,
}

fn default_tribe_q() -> Vec<[f32; TRIBE_ACTIONS]> {
    vec![[0.0; TRIBE_ACTIONS]; TRIBE_STATES]
}

impl Default for TribeBrain {
    fn default() -> Self {
        TribeBrain {
            q: default_tribe_q(),
            last_state: None,
            action: TribeAction::Research,
            action_started: 0,
            epoch_stats: TribeStats::default(),
            epochs: 0,
            war_epochs: 0,
        }
    }
}

impl TribeBrain {
    /// Discretise the tribal situation.
    pub fn state_index(stats: &TribeStats, stressed: bool) -> u8 {
        let pop = match stats.population {
            0..=5 => 0,
            6..=15 => 1,
            16..=40 => 2,
            _ => 3,
        } as u8;
        let tech = stats.era_level.min(2);
        (pop * 3 + tech) * 4 + u8::from(stats.at_war) * 2 + u8::from(stressed)
    }

    pub fn learn(&mut self, state: u8, action: TribeAction, reward: f32, next_state: u8) {
        let alpha = 0.20;
        let gamma = 0.85;
        let best_next = self.q[next_state as usize]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let old = self.q[state as usize][action.index()];
        self.q[state as usize][action.index()] = old + alpha * (reward + gamma * best_next - old);
    }

    /// Epsilon-greedy posture; `roll` is a caller-supplied `0..=1` draw.
    pub fn choose(&self, state: u8, roll: f32, epsilon: f32) -> TribeAction {
        if roll < epsilon {
            return match ((roll / epsilon.max(1e-6)) * 4.0) as usize {
                0 => TribeAction::War,
                1 => TribeAction::Ally,
                2 => TribeAction::Missionary,
                _ => TribeAction::Research,
            };
        }
        let row = &self.q[state as usize];
        let mut best = TribeAction::ALL[0];
        let mut best_v = row[0];
        for (i, &v) in row.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best = TribeAction::ALL[i];
            }
        }
        best
    }
}

/// The shaping a posture applies to one member's reward this tick, and the
/// global baseline. Everything here is deliberately small: it tilts learning,
/// it does not play the game.
#[derive(Clone, Copy, Debug)]
pub struct TribePolicy {
    pub action: TribeAction,
    /// Added to every member's reward, every tick, for the posture's duration.
    pub reward_baseline: f32,
    /// Multiplier on Fight-macro choice scores (the "宣战" coefficient).
    pub attack_mult: f32,
}

impl TribePolicy {
    pub fn of(action: TribeAction) -> Self {
        match action {
            // War pays combat, costs everyone a little every tick.
            TribeAction::War => TribePolicy {
                action,
                reward_baseline: -0.0004,
                attack_mult: 1.6,
            },
            TribeAction::Ally => TribePolicy {
                action,
                reward_baseline: 0.0003,
                attack_mult: 1.0,
            },
            TribeAction::Missionary => TribePolicy {
                action,
                reward_baseline: 0.0002,
                attack_mult: 1.0,
            },
            TribeAction::Research => TribePolicy {
                action,
                reward_baseline: 0.0003,
                attack_mult: 1.0,
            },
        }
    }

    /// The posture's multiplier for a concrete action's reward.
    pub fn reward_mult(self, action: usize, skilled: bool, social: bool) -> f32 {
        match self.action {
            TribeAction::War => {
                if super::vector::MacroIntent::of_action(action) == Some(MacroIntent::Fight) {
                    1.25
                } else {
                    1.0
                }
            }
            TribeAction::Ally => {
                if social || matches!(action, 94 | 276..=295) {
                    1.20
                } else {
                    1.0
                }
            }
            TribeAction::Missionary => {
                if social {
                    1.30
                } else {
                    1.0
                }
            }
            TribeAction::Research => {
                if skilled {
                    1.25
                } else {
                    1.0
                }
            }
        }
    }
}

use super::vector::MacroIntent;

/// The epoch reward for a tribe, from the delta between snapshots.
pub(crate) fn tribe_epoch_reward(before: &TribeStats, after: &TribeStats) -> f32 {
    let mut reward = 0.0f32;
    // Population is the currency of dynasties.
    let pop_delta = after.population as f32 - before.population as f32;
    let scale = (before.population.max(1) as f32).recip();
    reward += (pop_delta * scale) * 1.2;
    // Technology is the other one.
    reward += (after.era_level as f32 - before.era_level as f32) * 0.8;
    reward += (after.progress as f32 - before.progress as f32).min(40.0) * 0.005;
    // A starving tribe is a failing tribe.
    reward += (after.avg_energy - before.avg_energy) * 0.6;
    // Unrest eats legitimacy.
    reward -= (after.unrest - before.unrest) * 0.4;
    reward
}

/// War-weariness coupling: epochs spent actually fighting under a `War`
/// posture gradually cost the tribe's own Q table, so the top layer can
/// learn to turn away from an endless war instead of holding it forever.
/// Posturing without a war, and any posture other than `War`, costs nothing.
pub(crate) fn war_weariness(war_epochs: u32, action: TribeAction, at_war: bool) -> f32 {
    if action != TribeAction::War || !at_war || war_epochs == 0 {
        return 0.0;
    }
    (war_epochs as f32 * WEARINESS_PER_EPOCH).min(WEARINESS_CAP)
}

/// The attack weight actually broadcast for a `War` posture: the full
/// coefficient on a fresh war, fading toward fatigue as consecutive war
/// epochs accrue. (`TribePolicy::of(War).attack_mult` is the epoch-0 value.)
pub(crate) fn effective_attack_mult(war_epochs: u32) -> f32 {
    1.0 + (TribePolicy::of(TribeAction::War).attack_mult - 1.0) / (1.0 + 0.25 * war_epochs as f32)
}

/// The `Ally` posture's attitude drift: a tribe that means well warms every
/// relationship it has with every living foreign lineage, so friendly
/// gestures (`form_alliance` needs attitude > 0.2, envoys, trade, truce)
/// become reachable. Deterministic: the caller passes the other lineage ids
/// already sorted; the map itself is a `BTreeMap`.
pub(crate) fn drift_attitudes_for_posture(
    o: &mut crate::organism::organism::Organism,
    others: &[crate::intern::Sym],
    action: TribeAction,
) {
    if action != TribeAction::Ally {
        return;
    }
    for &lid in others {
        if lid == o.lineage_id {
            continue;
        }
        let att = o.lineage_attitudes.entry(lid).or_insert(0.0);
        *att = (*att + ALLY_DRIFT).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posture_names_are_wire_stable() {
        // The tribe card badge reads these verbatim from the frame payload;
        // never rename them — old saves and old clients depend on the words.
        assert_eq!(TribeAction::War.name(), "war");
        assert_eq!(TribeAction::Ally.name(), "ally");
        assert_eq!(TribeAction::Missionary.name(), "missionary");
        assert_eq!(TribeAction::Research.name(), "research");
    }

    #[test]
    fn tribe_q_learning_pulls_values_toward_reward() {
        let mut b = TribeBrain::default();
        let s = TribeBrain::state_index(
            &TribeStats {
                population: 20,
                era_level: 1,
                at_war: false,
                ..Default::default()
            },
            false,
        );
        for _ in 0..50 {
            b.learn(s, TribeAction::Research, 1.0, s);
        }
        assert!(b.q[s as usize][TribeAction::Research.index()] > 0.5);
        assert!(b.q[s as usize][TribeAction::Research.index()] > b.q[s as usize][TribeAction::War.index()]);
    }

    #[test]
    fn war_posture_pays_combat_and_charges_a_baseline() {
        let war = TribePolicy::of(TribeAction::War);
        assert_eq!(war.reward_mult(12, false, false), 1.25); // attack action
        assert_eq!(war.reward_mult(8, false, false), 1.0); // foraging unaffected
        assert!(war.reward_baseline < 0.0);
        assert!(war.attack_mult > 1.0);
    }

    #[test]
    fn research_posture_pays_skilled_work() {
        let research = TribePolicy::of(TribeAction::Research);
        assert_eq!(research.reward_mult(50, true, false), 1.25);
        assert_eq!(research.reward_mult(50, false, false), 1.0);
    }

    #[test]
    fn epoch_reward_prefers_growth_over_decline() {
        let before = TribeStats {
            population: 10,
            ..Default::default()
        };
        let after = TribeStats {
            population: 13,
            ..Default::default()
        };
        let declined = TribeStats {
            population: 8,
            ..Default::default()
        };
        assert!(tribe_epoch_reward(&before, &after) > 0.0);
        assert!(tribe_epoch_reward(&before, &declined) < 0.0);
    }

    #[test]
    fn state_index_spans_the_table() {
        for pop in [0u32, 6, 16, 41] {
            for era in [0u8, 1, 2] {
                for war in [false, true] {
                    for stress in [false, true] {
                        let s = TribeBrain::state_index(
                            &TribeStats {
                                population: pop,
                                era_level: era,
                                at_war: war,
                                ..Default::default()
                            },
                            stress,
                        );
                        assert!((s as usize) < TRIBE_STATES);
                    }
                }
            }
        }
    }

    #[test]
    fn war_weariness_grows_only_while_actually_fighting() {
        let one = war_weariness(1, TribeAction::War, true);
        let ten = war_weariness(10, TribeAction::War, true);
        let fifty = war_weariness(50, TribeAction::War, true);
        assert!(one > 0.0);
        assert!(ten > one);
        assert_eq!(fifty, WEARINESS_CAP, "weariness caps");
        assert_eq!(
            war_weariness(5, TribeAction::War, false),
            0.0,
            "drums without war are free"
        );
        assert_eq!(war_weariness(5, TribeAction::Research, true), 0.0);
        assert_eq!(war_weariness(0, TribeAction::War, true), 0.0);
    }

    #[test]
    fn attack_weight_fades_as_wars_drag_on() {
        let fresh = effective_attack_mult(0);
        assert_eq!(fresh, TribePolicy::of(TribeAction::War).attack_mult);
        assert!(effective_attack_mult(1) < fresh);
        assert!(effective_attack_mult(12) < effective_attack_mult(4));
        assert!(
            effective_attack_mult(100) > 1.0,
            "fatigue softens but never disarms"
        );
    }

    #[test]
    fn ally_posture_drifts_attitudes_but_other_postures_do_not() {
        use crate::intern::Sym;
        let mut o = crate::organism::organism::Organism::new(
            Sym::from("Zozo"),
            "Zozo".to_string(),
            3.0,
            4.0,
            0,
            String::new(),
            Sym::from("Home"),
            20,
            crate::organism::traits::Traits::default(),
        );
        let (a, b, c) = (Sym::from("Alph"), Sym::from("Beta"), Sym::from("Home"));
        o.lineage_attitudes.insert(a, -0.5);
        o.lineage_attitudes.insert(b, 1.0);
        let others = vec![a, b, c];
        drift_attitudes_for_posture(&mut o, &others, TribeAction::Ally);
        assert!(
            (o.lineage_attitudes[&a] - (-0.5 + ALLY_DRIFT)).abs() < 1e-6,
            "rivals thaw"
        );
        assert_eq!(o.lineage_attitudes[&b], 1.0, "warmth caps at 1.0");
        assert!(!o.lineage_attitudes.contains_key(&c), "own lineage never drifts");
        let before = o.lineage_attitudes[&a];
        drift_attitudes_for_posture(&mut o, &others, TribeAction::Research);
        assert_eq!(
            o.lineage_attitudes[&a], before,
            "non-Ally postures leave attitudes alone"
        );
        assert_eq!(
            o.attitude_toward("Alph"),
            before,
            "attitude_toward sees the drift"
        );
    }
}
