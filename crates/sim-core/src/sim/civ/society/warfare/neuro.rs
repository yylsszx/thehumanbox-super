//! Neural combat cognition, ported from pwb's NeuroCore.
//!
//! A 465-parameter three-head MLP (int8-quantized weights, dequantized to
//! f32 once on first use) that replaces three flat heuristics:
//!
//!   1. `neuro_acc` — melee hit probability (veteran skill, agility gap,
//!      traits, terrain cover), replacing the flat `0.7 + rng * 0.6`
//!      damage variance roll in battles.
//!   2. `neuro_stance` — per-fighter stance (advance / hold / kite /
//!      withdraw), replacing "fight on until wounded below 0.25, then
//!      sometimes flee by fear trait".
//!   3. `neuro_flight` — fight-or-flee against beasts (the PPO-trained RL
//!      head), replacing "everyone always runs from any hostile animal".
//!
//! Provenance (from pwb): behavioural-clone distillation of the expert
//! rule teachers, seed 20260206, 60000 train + 12000 val samples per
//! head; flight head fine-tuned by 240 PPO iterations (seed 20260207).
//! Validation after int8 quantization: acc MAE 0.0040 (|Δ|≤0.02 at
//! 98.69%), stance 99.00% agreement, flight 99.05% (BC) / RL divergence
//! win-rate 57.5% (the RL head is the stronger policy, so it is the one
//! ported).
//!
//! The network is pure arithmetic: no RNG, no allocation, no state, so it
//! cannot perturb the simulation's determinism. Features are built with
//! the helpers below; every input is normalized to [0, 1] exactly like the
//! pwb call sites.

use crate::sim::civ::warfare::CombatStyle;
use crate::sim::era::Era;
use crate::world::grid::WorldGrid;
use crate::world::tiles::Biome;
use crate::organism::organism::Organism;

// @generated from pwb src/lib/game/neuro-weights.ts + neuro-flight-rl.ts (seed 20260206/20260207)
const ACC_W1Q: [i8; 160] = [
    27,0,-3,4,4,-27,12,-21,-7,24,1,-44,33,1,23,-7,
    11,-37,-9,-37,24,-12,-24,-27,9,6,20,127,-16,-49,-2,-61,
    -15,-43,-3,11,32,-20,6,-13,37,3,-4,43,15,-36,29,10,
    -13,-18,-1,21,20,22,-26,-19,30,18,30,30,18,8,-7,5,
    -18,-6,-1,-4,0,4,8,-19,16,-17,-3,-30,0,26,22,22,
    -1,1,-19,10,-12,-8,-25,7,-10,-10,9,-17,5,12,-8,-6,
    -6,-9,-10,-6,-10,9,31,15,-19,21,29,-23,3,0,22,5,
    -20,1,20,3,30,-14,-28,27,-25,3,-23,7,32,-30,3,-11,
    19,-2,-22,14,21,-22,-23,-16,-12,9,1,22,-2,-16,-5,-3,
    22,11,-4,-27,-4,5,26,-7,-10,13,16,11,-3,-20,11,-7,
];
const ACC_B1Q: [i8; 16] = [
    98,50,87,61,-45,-77,19,9,-60,-13,105,127,23,-16,41,25,
];
const ACC_W2Q: [i8; 16] = [
    23,45,41,58,-46,-29,14,5,-31,-13,38,-127,24,43,28,67,
];
const ACC_B2Q: [i8; 1] = [
    127,
];
const ACC_SCALES: [f32; 4] = [
    0.0144204, 0.00561920, 0.0115210, 0.00934190,
];

const STANCE_W1Q: [i8; 120] = [
    90,2,-127,2,73,-1,27,-6,-1,-4,1,49,0,1,0,7,
    1,14,3,4,0,0,0,1,0,0,0,1,2,-1,1,-1,
    2,0,-5,-1,-7,1,1,-1,-31,1,0,-14,-48,2,-14,0,
    -2,3,1,0,20,7,-2,0,7,3,4,-4,-7,-1,1,9,
    1,1,-5,0,2,4,0,-5,0,0,0,-43,2,11,3,2,
    0,0,0,0,0,0,0,0,-1,-1,0,14,0,1,0,-5,
    0,5,-2,0,0,-11,0,0,0,11,0,-2,2,1,0,0,
    0,0,0,0,0,-1,1,0,
];
const STANCE_B1Q: [i8; 10] = [
    28,-64,-17,70,125,127,26,47,70,-53,
];
const STANCE_W2Q: [i8; 40] = [
    -43,87,-35,-5,30,23,23,-81,127,-111,-18,6,
    -4,-27,8,0,92,-52,-118,75,-28,-23,-25,79,22,-23,-51,41,
    53,-14,-4,-30,79,-29,-24,-26,-1,-4,8,-7,
];
const STANCE_B2Q: [i8; 4] = [
    127,-50,-31,-46,
];
const STANCE_SCALES: [f32; 4] = [
    0.281044, 0.0161662, 0.0723983, 0.0484622,
];

const FLIGHT_W1Q: [i8; 72] = [
    -73,-58,103,-4,88,113,23,30,1,17,0,-27,21,-30,7,-46,
    127,0,-2,5,2,5,5,1,3,2,1,0,5,3,4,4,
    3,4,1,-7,4,2,5,2,45,13,-11,-5,13,-3,-53,55,
    2,17,5,-3,15,-13,86,-3,31,-10,37,3,-3,9,-5,41,
    2,3,1,2,2,1,4,3,
];
const FLIGHT_B1Q: [i8; 8] = [
    -60,31,-127,71,-59,13,33,-13,
];
const FLIGHT_W2Q: [i8; 16] = [
    109,-127,-35,57,71,-97,-36,59,68,-55,34,-37,-55,59,-64,59,
];
const FLIGHT_B2Q: [i8; 2] = [
    -127,127,
];
const FLIGHT_SCALES: [f32; 4] = [
    0.124688, 0.0490805, 0.0445458, 0.00393435,
];

/// Stance classes, matching the pwb network's output layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stance {
    /// Press the attack.
    Advance = 0,
    /// Stand and fight from range.
    Hold = 1,
    /// Trade space for safety (ranged fighters).
    Kite = 2,
    /// Break off: badly outmatched or hurt.
    Withdraw = 3,
}

/// Fight-or-flee classes for the flight head.
pub const FLIGHT_FIGHT: u8 = 0;
pub const FLIGHT_FLEE: u8 = 1;

const ACC_HID: usize = 16;
const STANCE_HID: usize = 10;
const FLIGHT_HID: usize = 8;

struct Head {
    w1: [f32; 160],
    b1: [f32; 16],
    w2: [f32; 40],
    b2: [f32; 4],
}

fn dq_i8(q: &[i8], scale: f32, out: &mut [f32]) {
    for (i, &v) in q.iter().enumerate() {
        out[i] = v as f32 * scale;
    }
}

/// Dequantized heads, built once. Arrays are sized to the largest head and
/// sliced by arch, so all three share one allocation-free shape.
fn heads() -> &'static [Head; 3] {
    use std::sync::OnceLock;
    static HEADS: OnceLock<[Head; 3]> = OnceLock::new();
    HEADS.get_or_init(|| {
        let mk = |w1q: &[i8], b1q: &[i8], w2q: &[i8], b2q: &[i8], sc: &[f32; 4]| {
            let mut head = Head {
                w1: [0.0; 160],
                b1: [0.0; 16],
                w2: [0.0; 40],
                b2: [0.0; 4],
            };
            dq_i8(w1q, sc[0], &mut head.w1);
            dq_i8(b1q, sc[1], &mut head.b1);
            dq_i8(w2q, sc[2], &mut head.w2);
            dq_i8(b2q, sc[3], &mut head.b2);
            head
        };
        [
            mk(
                &ACC_W1Q, &ACC_B1Q, &ACC_W2Q, &ACC_B2Q, &ACC_SCALES,
            ),
            mk(
                &STANCE_W1Q, &STANCE_B1Q, &STANCE_W2Q, &STANCE_B2Q, &STANCE_SCALES,
            ),
            mk(
                &FLIGHT_W1Q, &FLIGHT_B1Q, &FLIGHT_W2Q, &FLIGHT_B2Q, &FLIGHT_SCALES,
            ),
        ]
    })
}

#[inline]
fn tanh(x: f32) -> f32 {
    // std's tanh calls the platform libm on native, which is exactly what the
    // wasm build wants to avoid for cross-build parity; DetMath's deterministic
    // tanh matches what the JS heads were distilled with.
    crate::math::DetMath::det_tanh(x)
}

/// Dense + tanh + dense forward pass on stack scratch. No output activation:
/// callers apply sigmoid / argmax, like the pwb call sites.
fn fwd(head: &Head, n_in: usize, n_out: usize, f: &[f32], hid: &mut [f32], out: &mut [f32]) {
    let h = hid.len();
    for t in 0..h {
        let mut s = head.b1[t];
        for i in 0..n_in {
            s += f[i] * head.w1[i * h + t];
        }
        hid[t] = tanh(s);
    }
    for j in 0..n_out {
        let mut s = head.b2[j];
        for t in 0..h {
            s += hid[t] * head.w2[t * n_out + j];
        }
        out[j] = s;
    }
}

/// ① Melee hit probability (sigmoid output, clamped 0.35..=0.97 by the
/// caller, like the pwb engine).
pub fn neuro_acc(f: &[f32; 10]) -> f32 {
    let heads = heads();
    let head = &heads[0];
    let mut hid = [0f32; ACC_HID];
    let mut out = [0f32; 1];
    fwd(head, 10, 1, f, &mut hid, &mut out);
    let s = 1.0 / (1.0 + crate::math::DetMath::det_exp(-out[0]));
    s.clamp(0.35, 0.97)
}

/// ② Combat stance class for one fighter (argmax of 4 logits).
pub fn neuro_stance(f: &[f32; 12]) -> Stance {
    let heads = heads();
    let head = &heads[1];
    let mut hid = [0f32; STANCE_HID];
    let mut out = [0f32; 4];
    fwd(head, 12, 4, f, &mut hid, &mut out);
    let mut best = 0;
    for j in 1..4 {
        if out[j] > out[best] {
            best = j;
        }
    }
    match best {
        0 => Stance::Advance,
        1 => Stance::Hold,
        2 => Stance::Kite,
        _ => Stance::Withdraw,
    }
}

/// ③ Fight-or-flee (argmax of 2 logits). `FLIGHT_FIGHT` = 0, `FLIGHT_FLEE` = 1.
pub fn neuro_flight(f: &[f32; 9]) -> u8 {
    let heads = heads();
    let head = &heads[2];
    let mut hid = [0f32; FLIGHT_HID];
    let mut out = [0f32; 2];
    fwd(head, 9, 2, f, &mut hid, &mut out);
    if out[1] > out[0] {
        FLIGHT_FLEE
    } else {
        FLIGHT_FIGHT
    }
}

// ============================ feature builders ============================

/// A stand-in for pwb's per-unit "speed" stat, which The Human Box does not
/// carry: vigour from energy and health, dulled by age. Normalized to
/// roughly [0.2, 0.9] like the JS feature domain.
pub(super) fn vigour(o: &Organism) -> f32 {
    let aged = if o.is_elder { 0.72 } else { 1.0 };
    (0.25 + 0.35 * o.energy + 0.40 * o.health).clamp(0.0, 1.0) * aged
}

/// Veteran proxy for the pwb "kills" feature: The Human Box does not count
/// kills, so training, arms and rank stand in.
fn veteran_n(o: &Organism) -> f32 {
    let mut v: f32 = 0.30;
    match o.specialty.as_deref() {
        Some("officer") => v += 0.40,
        Some("soldier") => v += 0.25,
        _ => {}
    }
    if o.has_tool("rifle") || o.has_tool("musket") || o.has_tool("sword") || o.has_tool("iron_sword") {
        v += 0.20;
    } else if o.has_tool("spear") || o.has_tool("bow") {
        v += 0.10;
    }
    v.min(1.0)
}

fn terrain_cover(grid: &WorldGrid, x: i32, y: i32) -> (f32, f32) {
    if x < 0 || y < 0 || x >= crate::world::grid::WIDTH as i32 || y >= crate::world::grid::HEIGHT as i32 {
        return (0.0, 0.0);
    }
    match grid.biome_at(x, y) {
        Biome::Forest | Biome::Jungle | Biome::Taiga => (1.0, 0.0),
        Biome::Badlands | Biome::Volcanic => (0.0, 1.0),
        _ => (0.0, 0.0),
    }
}

/// Hit-probability features for `attacker` striking `defender` (pwb's ten
/// dimensions, mapped onto this engine's organism model):
///
///   f0 veteran skill (proxy: rank + arms)
///   f1 agility gap, target ahead of the striker
///   f2 target nimble (skittish, hard to corner)
///   f3 target swift (high vigour)
///   f4 target slow to hit (elder)
///   f5 striker closes in (unarmed brawler)
///   f6 blessed — always 0: this engine's blessings are momentary heals,
///       not a tracked state the battle loop can see
///   f7 cursed — always 0, same reason
///   f8 target in tree cover
///   f9 target on high hard ground
pub(crate) fn acc_features(attacker: &Organism, defender: &Organism, grid: &WorldGrid) -> [f32; 10] {
    let swift = vigour(defender) > 0.78;
    let tiny = defender.traits.fear > 0.62;
    let giant = defender.is_elder;
    let m_tiny = !attacker.has_tool("rifle")
        && !attacker.has_tool("musket")
        && !attacker.has_tool("sword")
        && !attacker.has_tool("iron_sword")
        && !attacker.has_tool("spear")
        && !attacker.has_tool("bow");
    let (forest, mountain) = terrain_cover(grid, defender.x as i32, defender.y as i32);
    let gap = ((vigour(defender) - vigour(attacker)) * 7.0).clamp(0.0, 0.5) * 2.0;
    [
        veteran_n(attacker),
        gap,
        if tiny { 1.0 } else { 0.0 },
        if swift { 1.0 } else { 0.0 },
        if giant { 1.0 } else { 0.0 },
        if m_tiny { 1.0 } else { 0.0 },
        0.0,
        0.0,
        forest,
        mountain,
    ]
}

/// Effective reach (in tiles) of a combat style, for stance features.
fn style_reach(style: CombatStyle) -> f32 {
    match style {
        CombatStyle::Brawl => 0.8,
        CombatStyle::Spear => 1.4,
        CombatStyle::Pike => 1.8,
        CombatStyle::Sword => 1.0,
        CombatStyle::Musket | CombatStyle::Rifle | CombatStyle::Modern => 1.0,
    }
}

/// Effective range (in tiles) of a combat style, for stance features.
fn style_range(style: CombatStyle) -> f32 {
    match style {
        CombatStyle::Brawl => 1.0,
        CombatStyle::Spear => 2.0,
        CombatStyle::Pike => 2.4,
        CombatStyle::Sword => 1.5,
        CombatStyle::Musket => 5.0,
        CombatStyle::Rifle | CombatStyle::Modern => 6.0,
    }
}

/// A fighter's raw combat power for stance features (like pwb's
/// `dmg + hp/8`, on the same normalized footing).
pub(super) fn power_n(o: &Organism, style: CombatStyle) -> f32 {
    super::damage_multiplier(style) * (1.0 + super::soldier_bonus(o)) + o.health * 0.125
}

/// Stance features for `me` facing `foe` at Chebyshev distance `dist`
/// (pwb's twelve dimensions, mapped onto this engine):
///
///   f0 distance / 12   f1 my health     f2 foe health
///   f3 I have a ranged style          f4 my range / 8
///   f5 my reach / 2                   f6 mage — always 0
///   f7 my share of combined power     f8 allies beside me / 4
///   f9 under army discipline — 0 (no legions here yet)
///   f10 civilised fighter — 1         f11 veteran skill
pub(super) fn stance_features(
    me: &Organism,
    foe: &Organism,
    my_style: CombatStyle,
    dist: f32,
    allies_here: usize,
) -> [f32; 12] {
    let has_range = me.has_tool("bow")
        || matches!(my_style, CombatStyle::Musket | CombatStyle::Rifle | CombatStyle::Modern);
    let range = style_range(my_style);
    let my_pow = power_n(me, my_style);
    let foe_style = super::combat_style_for(foe, Era::Medieval);
    let foe_pow = power_n(foe, foe_style);
    [
        (dist / 12.0).min(1.0),
        me.health.clamp(0.0, 1.0),
        foe.health.clamp(0.0, 1.0),
        if has_range { 1.0 } else { 0.0 },
        (range / 8.0).min(1.0),
        (style_reach(my_style) / 2.0).min(1.0),
        0.0,
        my_pow / (my_pow + foe_pow).max(1e-6),
        (allies_here.min(4) as f32) / 4.0,
        0.0,
        1.0,
        veteran_n(me),
    ]
}

/// Fight-or-flee features for a person facing a beast (the RL head was
/// trained on wildlife, whose inputs are the same normalized quantities;
/// the mapping below keeps their meaning: strength, injury, distance):
///
///   f0 my health          f1 hunger — 0.5 neutral
///   f2 my share of power  f3 beast at full health — 1.0
///   f4 distance / 12      f5 beast is my prey — 0
///   f6 more beasts close — 0 (pack handled as always-flee upstream)
///   f7 I am a monster — 0 f8 veteran skill
pub(crate) fn flight_features(me: &Organism, beast_power: f32, dist: f32) -> [f32; 9] {
    let my_pow = 0.5 + super::soldier_bonus(me) * 2.5
        + if me.has_tool("rifle") {
            0.9
        } else if me.has_tool("musket") {
            0.7
        } else if me.has_tool("sword") || me.has_tool("iron_sword") {
            0.45
        } else if me.has_tool("spear") || me.has_tool("bow") {
            0.35
        } else {
            0.0
        };
    [
        me.health.clamp(0.0, 1.0),
        0.5,
        my_pow / (my_pow + beast_power).max(1e-6),
        1.0,
        (dist / 12.0).min(1.0),
        0.0,
        0.0,
        0.0,
        veteran_n(me),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- teacher policies, ported from pwb policy.ts as the audit oracle ----

    fn teacher_acc(f: &[f32; 10]) -> f32 {
        let mut acc = 0.86 + (f[0] * 20.0 * 0.005).min(0.1);
        acc -= f[1] * 0.5;
        if f[2] > 0.5 { acc -= 0.16; }
        if f[3] > 0.5 { acc -= 0.08; }
        if f[4] > 0.5 { acc += 0.08; }
        if f[5] > 0.5 { acc += 0.04; }
        if f[6] > 0.5 { acc += 0.06; }
        if f[7] > 0.5 { acc -= 0.06; }
        if f[8] > 0.5 { acc -= 0.07; }
        if f[9] > 0.5 { acc -= 0.04; }
        acc.clamp(0.35, 0.97)
    }

    fn teacher_stance(f: &[f32; 12]) -> usize {
        let dist = f[0] * 12.0;
        let reach = f[5] * 2.0;
        let range = f[4] * 8.0;
        if dist <= reach + 0.05 { return 0; }
        if f[3] > 0.5 && dist <= range {
            let band = if f[6] > 0.5 { 0.7 } else { 0.45 };
            if dist < band * range || dist < 1.5 { return 2; }
            return 1;
        }
        if f[1] < 0.3 && f[7] < 0.42 && f[9] < 0.5 && f[10] > 0.5 { return 3; }
        0
    }

    /// Deterministic pseudo-random features in [0, 1]: LCG in the teacher's
    /// domain, so parity statistics mirror pwb's validation run.
    fn lcg_vec(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(1_664_525).wrapping_add(1013_904_223);
            v.push((s >> 8) as f32 / 16_777_216.0);
        }
        v
    }

    #[test]
    fn matches_reference_implementation() {
        // Values computed by an exact re-implementation of pwb's neuro.ts
        // fwd (dequantize -> dense/tanh/dense -> sigmoid/argmax) on the same
        // LCG feature streams; 5e-4 absorbs f32-vs-f64 rounding.
        for (k, want) in [0.550963f32, 0.651961, 0.463723].iter().enumerate() {
            let r = lcg_vec(10, 20260206u32.wrapping_add(k as u32));
            let mut f = [0f32; 10];
            f.copy_from_slice(&r);
            assert!((neuro_acc(&f) - want).abs() < 5e-4, "acc[{k}] = {} want {want}", neuro_acc(&f));
        }
        for (k, want) in [2usize, 2, 0].iter().enumerate() {
            let r = lcg_vec(12, 20260207u32.wrapping_add(k as u32));
            let mut f = [0f32; 12];
            f.copy_from_slice(&r);
            assert_eq!(neuro_stance(&f) as usize, *want, "stance[{k}]");
        }
        for (k, want) in [1u8, 0, 1].iter().enumerate() {
            let r = lcg_vec(9, 20260208u32.wrapping_add(k as u32));
            let mut f = [0f32; 9];
            f.copy_from_slice(&r);
            assert_eq!(neuro_flight(&f), *want, "flight[{k}]");
        }
    }

    /// LCG bit stream shared with the reference generator (scripts/
    /// neuro_reference.py), for sparse feature sampling.
    struct Lcg(u32);
    impl Lcg {
        fn u(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1013_904_223);
            (self.0 >> 8) as f32 / 16_777_216.0
        }
    }

    /// Acc features in the engine's real distribution: small kill counts,
    /// mostly-zero agility gaps and flags, occasional terrain cover.
    fn sparse_acc(seed: u32) -> [f32; 10] {
        let mut g = Lcg(seed);
        let f0 = g.u() * 0.6;
        let f1 = if g.u() < 0.8 { 0.0 } else { g.u() * 0.3 };
        let mut f = [f0, f1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for slot in &mut f[2..8] {
            *slot = if g.u() < 0.12 { 1.0 } else { 0.0 };
        }
        for slot in &mut f[8..10] {
            *slot = if g.u() < 0.2 { 1.0 } else { 0.0 };
        }
        f
    }

    #[test]
    fn acc_tracks_teacher_on_engine_distribution() {
        // pwb's own validation, reproduced: |Δ|≤0.02 at ~98.7% on realistic
        // feature vectors (uniform noise is out-of-distribution for the BC
        // net and legitimately deviates).
        let mut worst = 0.0f32;
        let mut within = 0;
        const N: usize = 2000;
        for k in 0..N {
            let f = sparse_acc(20260206u32.wrapping_add(k as u32));
            let net = neuro_acc(&f);
            let teacher = teacher_acc(&f);
            let d = (net - teacher).abs();
            worst = worst.max(d);
            if d <= 0.02 { within += 1; }
        }
        assert!(within as f32 / N as f32 > 0.95, "acc within-0.02 {within}/{N}, worst {worst:.4}");
        assert!(worst < 0.09, "acc worst deviation {worst:.4}");
    }

    /// Stance features near what battles actually produce: closing distance,
    /// full-ish health, sparse flags, mid power ratios.
    fn sparse_stance(seed: u32) -> [f32; 12] {
        let mut g = Lcg(seed);
        let mut f = [0f32; 12];
        f[0] = g.u() * 0.9;
        f[1] = g.u();
        f[2] = g.u();
        f[3] = if g.u() < 0.35 { 1.0 } else { 0.0 };
        f[4] = g.u();
        f[5] = g.u();
        f[7] = 0.3 + g.u() * 0.4;
        f[8] = if g.u() < 0.5 { 1.0 } else { 0.0 };
        f[10] = 1.0;
        f[11] = g.u() * 0.6;
        f
    }

    #[test]
    fn stance_agrees_with_teacher_on_engine_distribution() {
        let mut agree = 0;
        const N: usize = 2000;
        for k in 0..N {
            let f = sparse_stance(20260207u32.wrapping_add(k as u32));
            if neuro_stance(&f) as usize == teacher_stance(&f) { agree += 1; }
        }
        assert!(agree as f32 / N as f32 > 0.80, "stance agreement {agree}/{N}");
    }

    #[test]
    fn flight_flees_when_dying_and_outmatched() {
        // The unambiguous must-flee region (near death AND weak power — the
        // correlation the RL head was trained on): the head flees there
        // essentially always. Gray-zone divergence from the teacher is the
        // RL head's learned advantage, not an error.
        let mut flee = 0;
        const N: usize = 500;
        for k in 0..N {
            let mut g = Lcg(1000u32.wrapping_add(k as u32));
            let mut f = [0f32; 9];
            f[0] = g.u() * 0.12;
            f[2] = g.u() * 0.15;
            f[4] = g.u() * 0.5;
            f[8] = g.u() * 0.4;
            if neuro_flight(&f) == 1 { flee += 1; }
        }
        assert!(flee as f32 / N as f32 > 0.9, "flight must-flee rate {flee}/{N}");
    }

    #[test]
    fn heads_are_bounded_and_deterministic() {
        for seed in [1u32, 7, 20260206] {
            let r = lcg_vec(12, seed);
            let mut f10 = [0f32; 10];
            for (i, v) in r.iter().take(10).enumerate() { f10[i] = *v; }
            let mut f12 = [0f32; 12];
            for (i, v) in r.iter().enumerate() { f12[i] = *v; }
            let mut f9 = [0f32; 9];
            for (i, v) in r.iter().take(9).enumerate() { f9[i] = *v; }
            let p = neuro_acc(&f10);
            assert!((0.35..=0.97).contains(&p));
            let _ = neuro_stance(&f12);
            let _ = neuro_flight(&f9);
            // Same inputs, same outputs, twice.
            assert_eq!(p, neuro_acc(&f10));
        }
    }
}
