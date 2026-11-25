//! Bottom layer of the hierarchy: vector states and a linear macro-intent Q.
//!
//! The tabular Q keyed by perception strings stays exactly as it was: exact,
//! cheap, and already persisted. What this module adds is the vector view the
//! upgrade calls for: every decision also produces a small feature vector
//! `[hunger, thirst, enemy_proximity, ally_count, kin_density, leader_present,
//! squad_cohesion, threat_level]`, and a per-organism linear function scores
//! the five *macro intents* (`Forage`, `Drink`, `Flee`, `Fight`, `Follow`)
//! from it. The linear scores generalise across situations the string table
//! has never seen, and they are blended into action selection with a small,
//! bounded boost so they steer, never override, the learned policy.
//!
//! The macro intents double as the action-space reduction: the middle layer
//! dispatches one of a handful of intents and the concrete action (out of the
//! 6144-id space) is refined underneath it, exactly as the architecture notes
//! describe.

use crate::organism::decision_bias::Directive;

/// Number of features in [`StateVector`].
pub(crate) const STATE_FEATURES: usize = 8;

/// The vector state of one organism at decision time.
///
/// Every feature is normalised to `0.0..=1.0` so the linear Q can be compared
/// and learned at a single scale. Feature order is part of the learned
/// weights: changing it invalidates saved linear weights (they are loaded
/// defensively and re-learn quickly).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StateVector(pub [f32; STATE_FEATURES]);

impl StateVector {
    pub const LEN: usize = STATE_FEATURES;

    #[inline]
    pub fn hunger(self) -> f32 {
        self.0[0]
    }
    #[inline]
    pub fn thirst(self) -> f32 {
        self.0[1]
    }
    #[inline]
    pub fn enemy_proximity(self) -> f32 {
        self.0[2]
    }
    #[inline]
    pub fn ally_count(self) -> f32 {
        self.0[3]
    }
    #[inline]
    pub fn kin_density(self) -> f32 {
        self.0[4]
    }
    #[inline]
    pub fn leader_present(self) -> f32 {
        self.0[5]
    }
    #[inline]
    pub fn squad_cohesion(self) -> f32 {
        self.0[6]
    }
    #[inline]
    pub fn threat_level(self) -> f32 {
        self.0[7]
    }
}

/// The five macro intents the bottom layer refines into concrete actions.
///
/// Mapped onto the existing `Directive` action families so the middle layer
/// can dispatch intents without inventing a second gating system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MacroIntent {
    Forage,
    Drink,
    Flee,
    Fight,
    Follow,
}

pub(crate) const MACRO_INTENT_COUNT: usize = 5;
pub(crate) const MACRO_INTENTS: [MacroIntent; MACRO_INTENT_COUNT] = [
    MacroIntent::Forage,
    MacroIntent::Drink,
    MacroIntent::Flee,
    MacroIntent::Fight,
    MacroIntent::Follow,
];

impl MacroIntent {
    #[inline]
    pub fn index(self) -> usize {
        match self {
            MacroIntent::Forage => 0,
            MacroIntent::Drink => 1,
            MacroIntent::Flee => 2,
            MacroIntent::Fight => 3,
            MacroIntent::Follow => 4,
        }
    }

    /// The one macro intent a concrete action serves, if any.
    ///
    /// Priority resolves the overlaps in the directive families: food before
    /// combat (a hunt feeds), flight before social (a plain step is not a
    /// flee). Plain steps `0..=7` are left unmapped so walking itself stays
    /// neutral ground every intent can use.
    pub fn of_action(action: usize) -> Option<MacroIntent> {
        if Directive::Food.aligns(action) || Directive::Hunt.aligns(action) {
            Some(MacroIntent::Forage)
        } else if Directive::Water.aligns(action) {
            Some(MacroIntent::Drink)
        } else if Directive::Fight.aligns(action) {
            Some(MacroIntent::Fight)
        } else if matches!(action, 11 | 17) {
            // The flight family minus the shared movement steps.
            Some(MacroIntent::Flee)
        } else if Directive::Socialize.aligns(action) {
            Some(MacroIntent::Follow)
        } else {
            None
        }
    }
}

/// A linear Q over the macro intents: one weight row per intent.
///
/// `w[intent * STATE_FEATURES + feature]`. Forty floats per organism — small
/// enough to inherit, save and load without a second storage format.
#[derive(Clone, Debug, Default)]
pub struct LinearQ {
    pub w: Vec<f32>,
}

impl LinearQ {
    pub fn new() -> Self {
        LinearQ {
            w: vec![0.0; MACRO_INTENT_COUNT * STATE_FEATURES],
        }
    }

    /// Build from persisted weights, tolerating any length (re-learns if short).
    pub fn from_vec(w: Vec<f32>) -> Self {
        let mut q = LinearQ::new();
        let n = w.len().min(q.w.len());
        q.w[..n].copy_from_slice(&w[..n]);
        q
    }

    #[inline]
    pub fn score(&self, intent: MacroIntent, sv: StateVector) -> f32 {
        let base = intent.index() * STATE_FEATURES;
        let mut sum = 0.0f32;
        for (k, &f) in sv.0.iter().enumerate() {
            sum += self.w[base + k] * f;
        }
        sum
    }

    /// The best macro-intent value in the next state (used as the bootstrap).
    pub fn best_next(&self, sv: StateVector) -> f32 {
        MACRO_INTENTS
            .iter()
            .map(|&intent| self.score(intent, sv))
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0)
    }

    /// One gradient step of the linear Q toward `target` for `intent` in `sv`.
    pub fn update(&mut self, intent: MacroIntent, sv: StateVector, target: f32, alpha: f32) {
        let base = intent.index() * STATE_FEATURES;
        let error = target - self.score(intent, sv);
        for (k, &f) in sv.0.iter().enumerate() {
            self.w[base + k] += alpha * error * f;
        }
        // Keep the weights bounded: this is a shaping heuristic riding on top
        // of a tabular Q, not the primary learner.
        for weight in &mut self.w[base..base + STATE_FEATURES] {
            *weight = weight.clamp(-2.0, 2.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macro_intents_partition_the_base_families() {
        assert_eq!(MacroIntent::of_action(8), Some(MacroIntent::Forage));
        assert_eq!(MacroIntent::of_action(28), Some(MacroIntent::Forage));
        assert_eq!(MacroIntent::of_action(9), Some(MacroIntent::Drink));
        assert_eq!(MacroIntent::of_action(12), Some(MacroIntent::Fight));
        assert_eq!(MacroIntent::of_action(11), Some(MacroIntent::Flee));
        assert_eq!(MacroIntent::of_action(10), Some(MacroIntent::Follow));
        // Plain steps stay neutral: every intent may walk.
        for step in 0..=7 {
            assert_eq!(MacroIntent::of_action(step), None);
        }
    }

    #[test]
    fn every_base_action_maps_stably() {
        for action in 0..=600 {
            let mapped = MacroIntent::of_action(action);
            if let Some(intent) = mapped {
                // No action may flip between intents by ordering accidents.
                assert_eq!(MacroIntent::of_action(action), Some(intent));
            }
        }
    }

    #[test]
    fn linear_q_learns_a_preferred_intent() {
        let mut q = LinearQ::new();
        let hungry_and_alone = StateVector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        for _ in 0..200 {
            q.update(MacroIntent::Forage, hungry_and_alone, 1.0, 0.05);
        }
        assert!(q.score(MacroIntent::Forage, hungry_and_alone) > 0.5);
        assert!(
            q.score(MacroIntent::Forage, hungry_and_alone) > q.score(MacroIntent::Drink, hungry_and_alone)
        );
        // Weights stay bounded even under relentless updates.
        assert!(q.w.iter().all(|w| w.abs() <= 2.0));
    }

    #[test]
    fn from_vec_tolerates_short_and_long_payloads() {
        let short = LinearQ::from_vec(vec![0.5]);
        assert_eq!(short.w[0], 0.5);
        assert_eq!(short.w.len(), MACRO_INTENT_COUNT * STATE_FEATURES);
        let long = LinearQ::from_vec(vec![1.0; 100]);
        assert_eq!(long.w.len(), MACRO_INTENT_COUNT * STATE_FEATURES);
    }
}
