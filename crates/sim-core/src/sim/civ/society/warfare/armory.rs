//! The emergent armory: weapons as architectures, not stat bags (pwb's
//! "weapon emergence architecture v2", ported).
//!
//! A forged weapon is a four-slot design — payload × delivery × guidance ×
//! power — whose slots are drawn by a doctrine-weighted lottery, biased by
//! what the enemy's defenses look like this year. Doctrine drifts with
//! battlefield feedback; defenses grow where a side keeps getting hit. A
//! lineage's designs inherit: each generation mutates one slot and carries
//! the quality vector forward, so arms races branch per bloodline, and two
//! mature designs can be fused (best-of per slot, hybrid vigour, may break
//! the damage soft cap by 15%).
//!
//! In battle, fighters fielding their lineage's design hit harder
//! (`lineage_weapon_multiplier`); a war's casualties credit the winner's
//! weapons and grow the loser's defenses. Big designs also *strike*: a
//! lineage at war occasionally lobs its best weapon at the front —
//! structures crumble, ground burns, bystanders fall.
//!
//! Everything here is persistent state saved with the world; its rolls come
//! from a private stream so the world's RNG stays dense.

use super::*;
use crate::sim::simulation::Simulation;
use crate::world::tiles::Tile;
use rand_chacha::ChaCha8Rng;

/// Payload slot: damage type, multipliers and side-effects.
struct PayloadSpec {
    name: &'static str,
    dmg_mul: f32,
    r_mul: f32,
    fire: f32,
    rad: bool,
    terra: bool,
    morale: f32,
}
const PAYLOADS: [PayloadSpec; 7] = [
    PayloadSpec { name: "AP", dmg_mul: 1.15, r_mul: 0.85, fire: 0.05, rad: false, terra: false, morale: 6.0 },
    PayloadSpec { name: "HE", dmg_mul: 0.95, r_mul: 1.25, fire: 0.25, rad: false, terra: false, morale: 10.0 },
    PayloadSpec { name: "Thermite", dmg_mul: 0.8, r_mul: 1.0, fire: 0.85, rad: false, terra: false, morale: 8.0 },
    PayloadSpec { name: "Pulse", dmg_mul: 0.75, r_mul: 1.1, fire: 0.0, rad: false, terra: false, morale: 24.0 },
    PayloadSpec { name: "Chemical", dmg_mul: 0.7, r_mul: 0.9, fire: 0.0, rad: true, terra: false, morale: 14.0 },
    PayloadSpec { name: "Fusion", dmg_mul: 1.5, r_mul: 1.35, fire: 0.5, rad: true, terra: false, morale: 18.0 },
    PayloadSpec { name: "Antimatter", dmg_mul: 1.8, r_mul: 1.45, fire: 0.0, rad: true, terra: true, morale: 26.0 },
];

/// Delivery slot: interception difficulty and saturation.
struct DeliverySpec {
    name: &'static str,
    stealth: f32,
    cadence: f32,
}
const DELIVERIES: [DeliverySpec; 5] = [
    DeliverySpec { name: "Cannon", stealth: 0.0, cadence: 1.3 },
    DeliverySpec { name: "Rocket", stealth: 0.15, cadence: 1.05 },
    DeliverySpec { name: "Bunker-buster", stealth: 0.35, cadence: 0.75 },
    DeliverySpec { name: "Glide", stealth: 0.55, cadence: 0.6 },
    DeliverySpec { name: "Orbital", stealth: 0.7, cadence: 0.45 },
];

/// Guidance slot: accuracy and decoy resistance.
struct GuideSpec {
    name: &'static str,
    acc: f32,
    decoy_res: f32,
    cadence: f32,
}
const GUIDES: [GuideSpec; 5] = [
    GuideSpec { name: "Dumb", acc: 0.3, decoy_res: 0.0, cadence: 1.4 },
    GuideSpec { name: "Inertial", acc: 0.55, decoy_res: 0.25, cadence: 1.0 },
    GuideSpec { name: "Precision", acc: 0.78, decoy_res: 0.5, cadence: 0.75 },
    GuideSpec { name: "Swarm", acc: 0.68, decoy_res: 0.85, cadence: 0.9 },
    GuideSpec { name: "Quantum", acc: 0.95, decoy_res: 1.0, cadence: 0.5 },
];

/// Power slot: damage amplification and wonder odds.
struct PowerSpec {
    name: &'static str,
    dmg_mul: f32,
    cadence: f32,
    wonder: f32,
}
const POWERS: [PowerSpec; 5] = [
    PowerSpec { name: "Chemical", dmg_mul: 1.0, cadence: 1.0, wonder: 0.0 },
    PowerSpec { name: "Laser", dmg_mul: 1.1, cadence: 1.15, wonder: 0.01 },
    PowerSpec { name: "Superconducting", dmg_mul: 1.2, cadence: 1.3, wonder: 0.02 },
    PowerSpec { name: "Reactor", dmg_mul: 1.35, cadence: 1.5, wonder: 0.03 },
    PowerSpec { name: "Singularity", dmg_mul: 1.55, cadence: 1.75, wonder: 0.05 },
];

const WEAP_EPI: [&str; 10] = [
    "Skyburner", "Earthsplitter", "Sea-Eater", "Silence", "Endsong",
    "Annihilator", "Cloudpiercer", "Star-devourer", "Thunderclap", "Daybreak",
];

/// A lineage's armament doctrine: which philosophy its arsenals pursue.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct WeaponDoctrine {
    pub precision: f32,
    pub saturation: f32,
    pub terror: f32,
    pub attrition: f32,
}

/// A lineage's defensive suite: grows under fire, capped by its armament era.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct WeaponDefense {
    pub point: f32,
    pub armor: f32,
    pub evade: f32,
}

/// One forged weapon design.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmergentWeapon {
    pub id: u32,
    pub lineage: String,
    pub name: String,
    pub gen: u32,
    pub payload: u8,
    pub delivery: u8,
    pub guidance: u8,
    pub power: u8,
    /// Quality vector (payload power / delivery speed / guidance accuracy /
    /// power efficiency), 0.6..=1.4, inherited with mutation.
    pub q: [f32; 4],
    pub dmg: f32,
    pub radius: f32,
    pub fire: f32,
    pub rad: bool,
    pub terra: bool,
    pub morale: f32,
    pub wonder: bool,
    pub kills: u32,
    pub fired: u32,
    pub parent: Option<u32>,
    pub born_tick: u64,
    pub retired: bool,
}

/// Hard cap on tracked designs.
const MAX_WEAPONS: usize = 128;
/// Active designs per lineage.
const MAX_ACTIVE_PER_LINEAGE: usize = 5;
/// A design is mature enough to fuse after this many firings.
const MATURE_AT: u32 = 5;
/// The armory tick's cadence: forging, doctrine drift bookkeeping, strikes.
const ARMORY_EVERY: u64 = 600;

/// The armament tier an era supports: 4 (firearms) at Industrial, 5 (modern)
/// at Modern, 6 (future) at Information.
pub(super) fn armament_tier(era: Era) -> u32 {
    if era >= Era::Information {
        6
    } else if era >= Era::Modern {
        5
    } else if era >= Era::Industrial {
        4
    } else {
        0
    }
}

impl Simulation {

    /// The damage soft cap a design of this tier may reach.
    fn weapon_dmg_cap(tier: u32) -> f32 {
        130.0 + tier as f32 * 14.0
    }

    /// Doctrine accessor: injects a random initial school on first touch, so
    /// each lineage's arms race branches from birth.
    fn doctrine_of(&mut self, lid: &str) -> WeaponDoctrine {
        use rand::RngExt;
        if let Some(d) = self.weapon_doctrines.get(lid) {
            return d.clone();
        }
        let mut r = || 0.14 + self.armory_rng.random::<f32>() * 0.24;
        let d = WeaponDoctrine {
            precision: r(),
            saturation: r(),
            terror: r(),
            attrition: r(),
        };
        self.weapon_doctrines.insert(lid.to_string().into(), d.clone());
        d
    }

    /// Defense accessor: starts at zero, grows under fire.
    fn defense_of(&mut self, lid: &str) -> WeaponDefense {
        if let Some(d) = self.weapon_defenses.get(lid) {
            return d.clone();
        }
        WeaponDefense::default()
    }

    /// The doctrine-weighted, threat-adapted, era-gated component lottery.
    fn pick_component(
        slot: u8,
        doc: &WeaponDoctrine,
        threat: &WeaponDefense,
        tier: u32,
        rng: &mut ChaCha8Rng,
    ) -> u8 {
        use rand::RngExt;
        let w: Vec<f32> = match slot {
            0 => vec![
                // Payload: terror schools reach for the exotic; attrition for
                // the workhorse; thick enemy armour pulls AP and Pulse.
                0.4 + doc.attrition + threat.armor / 55.0,
                0.9,
                0.45,
                0.4 + threat.armor / 80.0,
                0.3 + doc.terror * 1.7,
                if tier >= 6 { 0.25 + doc.terror * 1.3 } else { 0.0 },
                if tier >= 6 { 0.14 + doc.terror } else { 0.0 },
            ],
            1 => vec![
                // Delivery: saturation schools rocket; point defences pull the
                // stealthy fast balls.
                0.8 + doc.attrition * 0.8,
                0.6 + doc.saturation * 1.8,
                0.35 + threat.point / 90.0 + doc.precision * 0.4,
                if tier >= 5 { 0.3 + threat.point / 70.0 } else { 0.0 },
                if tier >= 6 { 0.22 + threat.point / 55.0 } else { 0.0 },
            ],
            2 => vec![
                // Guidance: precision schools aim; evasive foes need swarm or
                // quantum decoy resistance; attrition takes the cheap dumb.
                0.5 + doc.attrition * 1.2,
                0.9,
                0.5 + doc.precision * 1.6,
                0.35 + threat.evade / 70.0 + doc.saturation * 0.5,
                if tier >= 6 { 0.2 + doc.precision * 0.9 + threat.evade / 60.0 } else { 0.0 },
            ],
            _ => vec![
                // Power: higher tiers unlock the amplifiers.
                0.9,
                if tier >= 5 { 0.6 } else { 0.0 },
                if tier >= 5 { 0.45 } else { 0.0 },
                if tier >= 6 { 0.3 } else { 0.0 },
                if tier >= 6 { 0.18 } else { 0.0 },
            ],
        };
        let total: f32 = w.iter().sum();
        let mut pick = rng.random::<f32>() * total;
        for (k, weight) in w.iter().enumerate() {
            pick -= *weight;
            if pick <= 0.0 {
                return k as u8;
            }
        }
        0
    }

    /// Forge a design for `lineage`, optionally iterating on `parent`.
    /// Mirrors pwb's `forgeWeapon`, with the forge bill scaled to this
    /// engine's treasuries.
    fn forge_weapon(&mut self, lineage: &str, parent: Option<&EmergentWeapon>) -> Option<u32> {
        use rand::RngExt;
        if self.weapon_registry.len() >= MAX_WEAPONS {
            return None;
        }
        let active = self
            .weapon_registry
            .iter()
            .filter(|w| !w.retired && w.lineage == lineage)
            .count();
        if active >= MAX_ACTIVE_PER_LINEAGE {
            return None;
        }
        let era = self.lineage_eras.get(lineage).copied().unwrap_or(Era::PreStone);
        let tier = armament_tier(era);
        if tier < 4 {
            return None;
        }
        // The forge bill: later generations iterate dearer; a broke treasury
        // stops the military-industrial complex cold.
        let forge_cost: u64 = 40 + parent.map(|p| p.gen as u64 * 12).unwrap_or(20);
        let treasury = self.governments.get(lineage).map(|g| g.treasury).unwrap_or(0);
        if treasury < forge_cost {
            return None;
        }
        let mut doc = self.doctrine_of(lineage);
        // The threat picture: the average defence suite of current enemies.
        let mut threat = WeaponDefense::default();
        let mut foes = 0;
        let war_pairs: Vec<(String, String)> = self
            .diplomacy
            .wars
            .iter()
            .filter(|w| w.active())
            .map(|w| (w.attacker.clone(), w.defender.clone()))
            .collect();
        for (attacker, defender) in war_pairs {
            let foe = if attacker == lineage { defender } else { attacker };
            let d = self.defense_of(&foe);
            threat.point += d.point;
            threat.armor += d.armor;
            threat.evade += d.evade;
            foes += 1;
        }
        if foes > 0 {
            threat.point /= foes as f32;
            threat.armor /= foes as f32;
            threat.evade /= foes as f32;
        }
        let gen = parent.map(|p| p.gen + 1).unwrap_or(1);
        // Four slots: bloodline inheritance (one slot mutates) or all-new.
        let mut pl = parent.map(|p| p.payload).unwrap_or(1);
        let mut dl = parent.map(|p| p.delivery).unwrap_or(0);
        let mut gd = parent.map(|p| p.guidance).unwrap_or(1);
        let mut pw = parent.map(|p| p.power).unwrap_or(0);
        if let Some(p) = parent {
            match self.armory_rng.random_range(0..4u8) {
                0 => pl = Self::pick_component(0, &doc, &threat, tier, &mut self.armory_rng),
                1 => dl = Self::pick_component(1, &doc, &threat, tier, &mut self.armory_rng),
                2 => gd = Self::pick_component(2, &doc, &threat, tier, &mut self.armory_rng),
                _ => pw = Self::pick_component(3, &doc, &threat, tier, &mut self.armory_rng),
            }
            let _ = &pl;
        } else {
            pl = Self::pick_component(0, &doc, &threat, tier, &mut self.armory_rng);
            dl = Self::pick_component(1, &doc, &threat, tier, &mut self.armory_rng);
            gd = Self::pick_component(2, &doc, &threat, tier, &mut self.armory_rng);
            pw = Self::pick_component(3, &doc, &threat, tier, &mut self.armory_rng);
        }
        // Quality vector: inherit with ±16% mutation, or sample fresh.
        let clamp_q = |v: f32| (v * 1000.0).round() / 1000.0;
        let clamp_slot = |v: f32| v.clamp(0.6, 1.4);
        let q: [f32; 4] = match parent {
            Some(p) => {
                let mut m = || 0.96 + self.armory_rng.random::<f32>() * 0.16;
                let q0 = p.q;
                [
                    clamp_slot(clamp_q(q0[0] * m())),
                    clamp_slot(clamp_q(q0[1] * m())),
                    clamp_slot(clamp_q(q0[2] * m())),
                    clamp_slot(clamp_q(q0[3] * m())),
                ]
            }
            None => {
                let mut s = || clamp_q(0.75 + self.armory_rng.random::<f32>() * 0.55);
                [s(), s(), s(), s()]
            }
        };
        let p_spec = &PAYLOADS[pl as usize];
        let d_spec = &DELIVERIES[dl as usize];
        let g_spec = &GUIDES[gd as usize];
        let pw_spec = &POWERS[pw as usize];
        let wonder = parent.is_none() && self.armory_rng.random::<f32>() < 0.02 + pw_spec.wonder;
        let dmg_cap = Self::weapon_dmg_cap(tier);
        let mut dmg = (16.0 + 26.0 * q[0])
            * p_spec.dmg_mul
            * pw_spec.dmg_mul
            * (1.0 + 0.3 * (tier as f32 - 4.0))
            * (1.0 + 0.2 * (gen as f32 - 1.0))
            * if wonder { 2.0 } else { 1.0 };
        let mut radius = (1.2 + 0.9 * q[1]) * p_spec.r_mul + 0.12 * (gen as f32 - 1.0);
        let mut morale = p_spec.morale * (0.85 + 0.3 * q[2]);
        if let Some(p) = parent {
            // Bloodline iteration strengthens overall, with decaying gains.
            dmg = (p.dmg * (1.0 + (0.1 + self.armory_rng.random::<f32>() * 0.08) * 0.92f32.powi(gen as i32)))
                .max(dmg * 0.85)
                .min(dmg_cap);
            radius = (p.radius * 1.03).max(radius * 0.9);
            morale = (p.morale + 3.0).max(morale * 0.9).min(35.0);
        } else {
            dmg = dmg.min(dmg_cap);
        }
        let mk = if gen > 1 {
            format!(" Mk.{}", ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"][(gen as usize - 1).min(7)])
        } else {
            String::new()
        };
        let epi = WEAP_EPI[self.armory_rng.random_range(0..WEAP_EPI.len() as u32) as usize];
        let name = format!(
            "{}{}·{}·{}{}·{}",
            if wonder { "God-forged " } else { "" },
            p_spec.name,
            d_spec.name,
            g_spec.name,
            mk,
            epi
        );
        let id = self.next_weapon_id;
        self.next_weapon_id += 1;
        let w = EmergentWeapon {
            id,
            lineage: lineage.to_string(),
            name,
            gen,
            payload: pl,
            delivery: dl,
            guidance: gd,
            power: pw,
            q,
            dmg: (dmg * 10.0).round() / 10.0,
            radius: radius.min(6.5).round() / 10.0,
            fire: p_spec.fire * q[0].min(1.1),
            rad: p_spec.rad,
            terra: p_spec.terra,
            morale: morale.min(35.0).round(),
            wonder,
            kills: 0,
            fired: 0,
            parent: parent.map(|p| p.id),
            born_tick: self.tick_count,
            retired: false,
        };
        // The forge's ledger: the bill lands in the treasury.
        if let Some(g) = self.governments.get_mut(lineage) {
            g.treasury = g.treasury.saturating_sub(forge_cost);
        }
        let parent_lineage = parent.map(|p| p.lineage.clone());
        self.weapon_registry.push(w.clone());
        let wname = self
            .lineage_names
            .get(lineage)
            .cloned()
            .unwrap_or_else(|| lineage.to_string());
        let line = if parent.is_some_and(|p| p.lineage != lineage) {
            format!(
                "\u{1F52D} {} reverse-engineers a captured design into the {}.",
                wname, self.weapon_registry.last().unwrap().name
            )
        } else if parent.is_some() {
            // The old design retires when iterated on.
            if let Some(old) = self
                .weapon_registry
                .iter_mut()
                .find(|w| parent.is_some_and(|p| p.id == w.id))
            {
                old.retired = true;
            }
            format!("\u{1F52D} {} reforges its arsenal: the {} (gen {}).", wname, self.weapon_registry.last().unwrap().name, gen)
        } else if wonder {
            format!(
                "\u{1F52D} A wonder weapon! The {} forges the {}.",
                wname, self.weapon_registry.last().unwrap().name
            )
        } else {
            let school = if doc.terror > 0.4 {
                "terror"
            } else if doc.saturation > 0.4 {
                "saturation"
            } else if doc.precision > 0.4 {
                "precision"
            } else {
                "attrition"
            };
            format!(
                "\u{1F52D} {} forges a new {} doctrine design: the {}.",
                wname, school, self.weapon_registry.last().unwrap().name
            )
        };
        push_event(&mut self.events, self.tick_count, "milestone", "world", &line);
        self.headlines.push_back((self.tick_count, line));
        while self.headlines.len() > 80 {
            self.headlines.pop_front();
        }
        let _ = parent_lineage;
        Some(id)
    }

    /// Fuse two mature designs: best-of per slot, quality vector takes the
    /// max per axis, hybrid vigour may break the soft cap by 15%.
    fn fuse_weapon(&mut self, lineage: &str, wa: &EmergentWeapon, wb: &EmergentWeapon) -> Option<u32> {
        use rand::RngExt;
        if self.weapon_registry.len() >= MAX_WEAPONS {
            return None;
        }
        if wa.id == wb.id || wa.lineage != lineage || wb.lineage != lineage {
            return None;
        }
        let era = self.lineage_eras.get(lineage).copied().unwrap_or(Era::PreStone);
        let tier = armament_tier(era);
        if tier < 4 {
            return None;
        }
        let fuse_cost = 90 + wa.gen.max(wb.gen) as u64 * 15;
        let treasury = self.governments.get(lineage).map(|g| g.treasury).unwrap_or(0);
        if treasury < fuse_cost {
            return None;
        }
        let gen = wa.gen.max(wb.gen) + 1;
        let pl = if wa.q[0] * PAYLOADS[wa.payload as usize].dmg_mul
            >= wb.q[0] * PAYLOADS[wb.payload as usize].dmg_mul
        {
            wa.payload
        } else {
            wb.payload
        };
        let dl = if DELIVERIES[wa.delivery as usize].stealth >= DELIVERIES[wb.delivery as usize].stealth {
            wa.delivery
        } else {
            wb.delivery
        };
        let gd = if GUIDES[wa.guidance as usize].acc >= GUIDES[wb.guidance as usize].acc {
            wa.guidance
        } else {
            wb.guidance
        };
        let pw = if POWERS[wa.power as usize].dmg_mul >= POWERS[wb.power as usize].dmg_mul {
            wa.power
        } else {
            wb.power
        };
        let q = [
            wa.q[0].max(wb.q[0]),
            wa.q[1].max(wb.q[1]),
            wa.q[2].max(wb.q[2]),
            wa.q[3].max(wb.q[3]),
        ];
        let p_spec = &PAYLOADS[pl as usize];
        let d_spec = &DELIVERIES[dl as usize];
        let g_spec = &GUIDES[gd as usize];
        let pw_spec = &POWERS[pw as usize];
        let dmg_cap = Self::weapon_dmg_cap(tier) * 1.15;
        let dmg = ((16.0 + 26.0 * q[0]) * p_spec.dmg_mul * pw_spec.dmg_mul
            * (1.0 + 0.3 * (tier as f32 - 4.0))
            * (1.0 + 0.2 * (gen as f32 - 1.0)))
        .min(dmg_cap);
        let radius = ((1.2 + 0.9 * q[1]) * p_spec.r_mul + 0.12 * (gen as f32 - 1.0)).min(6.5);
        let morale = (p_spec.morale * (0.85 + 0.3 * q[2])).min(35.0);
        let epi = WEAP_EPI[self.armory_rng.random_range(0..WEAP_EPI.len() as u32) as usize];
        let name = format!("{}·{}·{} Mk.{}·{}", p_spec.name, d_spec.name, g_spec.name, gen, epi);
        let id = self.next_weapon_id;
        self.next_weapon_id += 1;
        let w = EmergentWeapon {
            id,
            lineage: lineage.to_string(),
            name,
            gen,
            payload: pl,
            delivery: dl,
            guidance: gd,
            power: pw,
            q,
            dmg: (dmg * 10.0).round() / 10.0,
            radius: radius.round() / 10.0,
            fire: p_spec.fire * q[0].min(1.1),
            rad: p_spec.rad,
            terra: p_spec.terra,
            morale: morale.round(),
            wonder: false,
            kills: 0,
            fired: 0,
            parent: Some(wa.id),
            born_tick: self.tick_count,
            retired: false,
        };
        // Both parents retire to make room; the bill lands.
        for old in self.weapon_registry.iter_mut() {
            if old.id == wa.id || old.id == wb.id {
                old.retired = true;
            }
        }
        if let Some(g) = self.governments.get_mut(lineage) {
            g.treasury = g.treasury.saturating_sub(fuse_cost);
        }
        self.weapon_registry.push(w.clone());
        let wname = self
            .lineage_names
            .get(lineage)
            .cloned()
            .unwrap_or_else(|| lineage.to_string());
        let line = format!(
            "\u{1F52D} {} fuses two bloodlines into the {}.",
            wname,
            self.weapon_registry.last().unwrap().name
        );
        push_event(&mut self.events, self.tick_count, "milestone", "world", &line);
        self.headlines.push_back((self.tick_count, line));
        while self.headlines.len() > 80 {
            self.headlines.pop_front();
        }
        Some(id)
    }

    /// The armory tick: lineages at the firearms era and beyond forge,
    /// iterate and fuse on cadence, and big designs at war strike the front.
    pub(crate) fn tick_armory(&mut self) {
        use rand::RngExt;
        if !self.tick_count.is_multiple_of(ARMORY_EVERY) {
            return;
        }
        let lineages: Vec<String> = self
            .lineage_eras
            .iter()
            .filter(|&(_, &era)| armament_tier(era) >= 4)
            .map(|(lid, _)| lid.to_string())
            .collect();
        for lid in lineages {
            if self.armory_rng.random::<f32>() >= 0.55 {
                continue;
            }
            let active: Vec<EmergentWeapon> = self
                .weapon_registry
                .iter()
                .filter(|w| !w.retired && w.lineage == lid)
                .cloned()
                .collect();
            let mature: Vec<EmergentWeapon> = active
                .iter()
                .filter(|w| w.fired >= MATURE_AT)
                .cloned()
                .collect();
            if mature.len() >= 2 && self.armory_rng.random::<f32>() < 0.30 {
                let (wa, wb) = (mature[0].clone(), mature[1].clone());
                self.fuse_weapon(&lid, &wa, &wb);
            } else {
                let parent = active
                    .iter()
                    .max_by(|a, b| a.kills.cmp(&b.kills).then(a.dmg.total_cmp(&b.dmg)))
                    .cloned();
                self.forge_weapon(&lid, parent.as_ref());
            }
        }
        // Strategic strikes: a lineage at war with a heavy design lobbed at
        // the front.
        self.weapon_strikes();
    }

    /// Big designs at war occasionally strike the front: structures crumble,
    /// ground burns, bystanders fall.
    fn weapon_strikes(&mut self) {
        use rand::RngExt;
        let strikes: Vec<(String, EmergentWeapon, [i32; 2])> = self
            .weapon_registry
            .iter()
            .filter(|w| !w.retired && w.dmg >= 60.0)
            .filter_map(|w| {
                let war = self
                    .diplomacy
                    .wars
                    .iter()
                    .find(|war| war.active() && (war.attacker == w.lineage || war.defender == w.lineage))?;
                Some((w.lineage.clone(), w.clone(), war.front))
            })
            .collect();
        for (lid, w, front) in strikes {
            if self.armory_rng.random::<f32>() >= 0.25 {
                continue;
            }
            let (sx, sy) = (
                front[0] + self.armory_rng.random_range(-6..=6),
                front[1] + self.armory_rng.random_range(-6..=6),
            );
            let radius = w.radius.max(1.5) as i32;
            let crush = w.dmg / 30.0;
            let mut ruined = 0u32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let (x, y) = (sx + dx, sy + dy);
                    if x < 0
                        || y < 0
                        || x >= crate::world::grid::WIDTH as i32
                        || y >= crate::world::grid::HEIGHT as i32
                    {
                        continue;
                    }
                    if self.grid.structure_at(x, y) > 0.1 {
                        self.grid.add_structure(x, y, -crush);
                        self.active_structure_tiles.insert((x, y));
                        ruined += 1;
                    }
                    // Incendiary payloads set the ground burning.
                    if w.fire > 0.5 && matches!(self.grid.get(x, y), Tile::Grass | Tile::Ash) {
                        self.grid.set(x, y, Tile::Fire);
                        self.active_structure_tiles.insert((x, y));
                    }
                }
            }
            // Bystanders near the impact.
            let victims: Vec<usize> = self
                .organisms
                .iter()
                .enumerate()
                .filter(|(_, o)| {
                    o.alive
                        && (o.x as i32 - sx).abs() <= radius
                        && (o.y as i32 - sy).abs() <= radius
                        && o.lineage_id.as_str() != lid
                })
                .map(|(i, _)| i)
                .collect();
            let mut killed = 0u32;
            for i in victims {
                if killed >= 3 {
                    break;
                }
                let o = &mut self.organisms[i];
                o.health = (o.health - w.dmg / 60.0).max(0.0);
                o.mark_harm(crate::organism::organism::Harm::War, self.tick_count);
                if o.health <= 0.0 && o.alive {
                    o.alive = false;
                    o.death_cause = "war".to_string();
                    o.log_life(self.tick_count, "death", "killed by a weapon strike".to_string());
                    self.history.deaths_combat += 1;
                    self.history.record_death(self.tick_count);
                    killed += 1;
                }
            }
            let fired = w.fired + 1;
            let kills = w.kills + killed;
            if let Some(slot) = self.weapon_registry.iter_mut().find(|x| x.id == w.id) {
                slot.fired = fired;
                slot.kills = kills;
            }
            let wname = self
                .lineage_names
                .get(&lid)
                .cloned()
                .unwrap_or_else(|| lid.clone());
            let line = format!(
                "\u{1F4A5} The {} strikes the front with the {}: {} structures ruined, {} dead.",
                wname, w.name, ruined, killed
            );
            push_event(&mut self.events, self.tick_count, "danger", "world", &line);
        }
    }

    /// Battle outcomes credit the winner's designs and grow the loser's
    /// defences: the attack-defence spiral has memory.
    pub(crate) fn credit_battle_feedback(&mut self) {
        let now = self.tick_count;
        let finished: Vec<(String, String, u32, u32)> = self
            .battles
            .iter()
            .filter(|b| b.ended_tick == Some(now) && b.outcome.is_some())
            .filter_map(|b| {
                let winner = match b.outcome {
                    Some(crate::sim::warfare::BattleOutcome::AttackerVictory) => b.attackers.first()?,
                    Some(crate::sim::warfare::BattleOutcome::DefenderVictory) => b.defenders.first()?,
                    _ => return None,
                };
                let loser = if &b.attackers[0] == winner { b.defenders.first()? } else { b.attackers.first()? };
                let (winner_cas, loser_cas) = if &b.attackers[0] == winner {
                    (b.casualties_a, b.casualties_d)
                } else {
                    (b.casualties_d, b.casualties_a)
                };
                Some((winner.clone(), loser.clone(), winner_cas, loser_cas))
            })
            .collect();
        for (winner, loser, _winner_cas, loser_cas) in finished {
            // The winner's fielded designs take credit for the fallen.
            let mut credited: Vec<u32> = Vec::new();
            for w in self.weapon_registry.iter_mut() {
                if w.retired || w.lineage != winner {
                    continue;
                }
                w.kills += loser_cas;
                w.fired += 1;
                credited.push(w.id);
            }
            // The loser's defences grow where the blows kept landing,
            // capped by what its armament era can field.
            let loser_era = self.lineage_eras.get(loser.as_str()).copied().unwrap_or(Era::PreStone);
            let cap = 18.0 + armament_tier(loser_era) as f32 * 13.0;
            let d = self.defense_of(&loser);
            let d = WeaponDefense {
                point: (d.point + loser_cas as f32 * 0.4).min(cap),
                armor: (d.armor + loser_cas as f32 * 0.5).min(cap),
                evade: (d.evade + loser_cas as f32 * 0.3).min(cap),
            };
            self.weapon_defenses.insert(loser.clone().into(), d);
            // Doctrine drift: the winner's school leans toward what its
            // designs actually do.
            if let Some(doc) = self.weapon_doctrines.get_mut(&winner) {
                let uses_terror = self
                    .weapon_registry
                    .iter()
                    .any(|w| w.lineage == winner && PAYLOADS[w.payload as usize].morale >= 14.0);
                if uses_terror {
                    doc.terror = (doc.terror + 0.02).min(0.6);
                    doc.precision = (doc.precision - 0.005).max(0.05);
                } else {
                    doc.attrition = (doc.attrition + 0.02).min(0.6);
                }
            }
        }
    }
}

/// The battle-facing multiplier of a lineage's best active design: fighters
/// fielding the arsenal hit up to 35% harder at the damage cap.
pub(super) fn lineage_weapon_multiplier(weapons: &[EmergentWeapon], lineage: Option<&str>, tier: u32) -> f32 {
    if tier < 4 {
        return 1.0;
    }
    let cap = Simulation::weapon_dmg_cap(tier);
    weapons
        .iter()
        .filter(|w| !w.retired && Some(w.lineage.as_str()) == lineage)
        .map(|w| 1.0 + 0.35 * (w.dmg / cap).clamp(0.0, 1.0))
        .fold(1.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// The component lottery respects its era gates and weight sums.
    #[test]
    fn component_lottery_respects_era_gates() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let doc = WeaponDoctrine { precision: 0.3, saturation: 0.3, terror: 0.3, attrition: 0.3 };
        let threat = WeaponDefense::default();
        for slot in 0..4u8 {
            // Tier 4: the future-only options must never come up.
            for _ in 0..200 {
                let pick = Simulation::pick_component(slot, &doc, &threat, 4, &mut rng);
                let len = match slot {
                    0 => 5, // payloads available at tier 4 (Fusion, Antimatter gated)
                    1 => 4, // delivery up to Glide
                    2 => 5,
                    _ => 3, // power up to Superconducting
                };
                assert!((pick as usize) < len, "slot {slot} picked {pick} at tier 4");
            }
        }
        // Tier 6: everything is legal.
        for slot in 0..4u8 {
            for _ in 0..200 {
                let pick = Simulation::pick_component(slot, &doc, &threat, 6, &mut rng);
                let len = match slot {
                    0 => 7,
                    1 => 5,
                    2 => 5,
                    _ => 5,
                };
                assert!((pick as usize) < len);
            }
        }
    }

    /// The battle multiplier tops out at 1.35 for a capped design and is 1.0
    /// without one.
    #[test]
    fn weapon_multiplier_is_bounded() {
        let w = EmergentWeapon {
            id: 1,
            lineage: "l".into(),
            name: "AP·Cannon·Dumb".into(),
            gen: 1,
            payload: 0,
            delivery: 0,
            guidance: 0,
            power: 0,
            q: [1.0; 4],
            dmg: 186.0,
            radius: 2.0,
            fire: 0.0,
            rad: false,
            terra: false,
            morale: 6.0,
            wonder: false,
            kills: 0,
            fired: 0,
            parent: None,
            born_tick: 0,
            retired: false,
        };
        let m = lineage_weapon_multiplier(&[w.clone()], Some("l"), 4);
        assert!((m - 1.35).abs() < 1e-3, "capped design gives 1.35x, got {m}");
        assert_eq!(lineage_weapon_multiplier(&[], Some("l"), 4), 1.0);
        assert_eq!(lineage_weapon_multiplier(&[w.clone()], Some("other"), 4), 1.0);
        // Retired designs do not count.
        let mut retired = w.clone();
        retired.retired = true;
        assert_eq!(lineage_weapon_multiplier(&[retired], Some("l"), 4), 1.0);
    }

    /// A lineage with treasury and tier forges; the bill lands.
    #[test]
    fn forging_costs_the_treasury_and_names_the_design() {
        let mut sim = Simulation::new(88);
        // Put the lineage at the firearms era with a funded treasury.
        sim.lineage_eras.insert("arms".into(), Era::Industrial);
        let mut gov = crate::sim::civ::government::Government::new("arms".to_string(), crate::sim::civ::government::GovernmentKind::Chiefdom, 0);
        gov.treasury = 500;
        sim.governments.insert("arms".to_string(), gov);
        let before = sim.governments.get("arms").unwrap().treasury;
        let id = sim.forge_weapon("arms", None);
        assert!(id.is_some(), "a funded firearms-era lineage forges");
        let after = sim.governments.get("arms").unwrap().treasury;
        assert!(after < before, "the forge bill landed");
        let w = sim
            .weapon_registry
            .iter()
            .find(|w| w.id == id.unwrap())
            .cloned()
            .unwrap();
        assert!(w.name.contains("·"), "the design is named as an architecture: {}", w.name);
        assert!((4.0..=186.0).contains(&w.dmg));
        // Iterating on it mutates a slot and retires the parent.
        let gen2 = sim.forge_weapon("arms", Some(&w));
        assert!(gen2.is_some());
        assert!(sim.weapon_registry.iter().find(|x| x.id == w.id).unwrap().retired);
        let w2 = sim.weapon_registry.iter().find(|x| x.id == gen2.unwrap()).unwrap();
        assert_eq!(w2.gen, 2);
        assert_eq!(w2.parent, Some(w.id));
    }

    /// Battle feedback grows the loser's defences and credits the winner's
    /// fielded designs.
    #[test]
    fn battle_feedback_grows_defenses_and_credits_kills() {
        let mut sim = Simulation::new(89);
        sim.lineage_eras.insert("arms".into(), Era::Industrial);
        sim.lineage_eras.insert("victim".into(), Era::Industrial);
        let mut gov = crate::sim::civ::government::Government::new("arms".to_string(), crate::sim::civ::government::GovernmentKind::Chiefdom, 0);
        gov.treasury = 500;
        sim.governments.insert("arms".to_string(), gov);
        sim.forge_weapon("arms", None);
        // A decisive battle: the winner took casualties 0, the loser 3.
        sim.battles.push(crate::sim::warfare::Battle {
            id: "b1".to_string(),
            attackers: vec!["arms".to_string()],
            defenders: vec!["victim".to_string()],
            attacker_orgs: vec![],
            defender_orgs: vec![],
            scale: crate::sim::warfare::BattleScale::Battle,
            location: (10, 10),
            casualties_a: 0,
            casualties_d: 3,
            initial_a: 10,
            initial_d: 10,
                        routed_a: false,
            routed_d: false,
started_tick: 0,
            ended_tick: Some(sim.tick_count),
            outcome: Some(crate::sim::warfare::BattleOutcome::AttackerVictory),
        });
        sim.credit_battle_feedback();
        let defense = sim.weapon_defenses.get("victim").cloned().unwrap_or_default();
        assert!(defense.armor > 0.0, "the loser's armour grew: {defense:?}");
        let credited = sim
            .weapon_registry
            .iter()
            .find(|w| w.lineage == "arms")
            .map(|w| w.kills)
            .unwrap_or(0);
        assert_eq!(credited, 3, "the winner's design took credit for the fallen");
    }
}
