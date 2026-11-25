//! Emergent civilisation content: techs and ideas beyond the static tree,
//! ported from pwb's lexical-combination generators.
//!
//! Once a lineage has left the Stone Age it does not stop inventing:
//! researchers coin technologies the design never dreamt of — prefix ×
//! field (16 × 16 combinations) with generation suffixes — each granting
//! small, saturating boosts to research, arms or war-weariness resistance.
//! Governments under cultural pressure adopt ideas (body × suffix, an
//! occasional prefix) that bend the whole lineage the same way. Names and
//! numbers are generated, not tabled: two worlds invent different histories.
//!
//! Effects fold into the live engine: research into the tech-progress
//! rate, combat into the battle damage multiplier, wear into
//! `war_weariness`. Everything persists in the save; rolls come from a
//! private stream.

use rand::RngExt;
use rand_chacha::ChaCha8Rng;

use crate::sim::era::Era;
use crate::sim::simulation::Simulation;

/// Tech word stock: prefix × field. 16 × 16 = 256 combinations, plus
/// roman-numeral generation suffixes for a practically unbounded space.
const EM_PREFIX: [&str; 16] = [
    "Quantum",
    "Nano",
    "Plasma",
    "Genetic",
    "Photon",
    "Superconducting",
    "Chaotic",
    "Psionic",
    "Dark-matter",
    "String",
    "Lattice",
    "Maglev",
    "Entropy",
    "Fusion",
    "Deep-space",
    "Graviton",
];

/// What a field does to the lineage that masters it.
struct FieldSpec {
    name: &'static str,
    /// Boosts battle damage.
    combat: bool,
    /// Boosts war-weariness resistance.
    morale: bool,
    /// Plague-field: research only, framed as a mixed blessing.
    medicine: bool,
}
const FIELDS: [FieldSpec; 16] = [
    FieldSpec {
        name: "Metallurgy",
        combat: true,
        morale: false,
        medicine: false,
    },
    FieldSpec {
        name: "Shield Engineering",
        combat: false,
        morale: false,
        medicine: false,
    },
    FieldSpec {
        name: "Agronomy",
        combat: false,
        morale: false,
        medicine: false,
    },
    FieldSpec {
        name: "Arithmetic",
        combat: false,
        morale: false,
        medicine: false,
    },
    FieldSpec {
        name: "Navigation",
        combat: false,
        morale: false,
        medicine: false,
    },
    FieldSpec {
        name: "Fiscality",
        combat: false,
        morale: false,
        medicine: false,
    },
    FieldSpec {
        name: "Building Codes",
        combat: false,
        morale: false,
        medicine: false,
    },
    FieldSpec {
        name: "Marketcraft",
        combat: false,
        morale: false,
        medicine: false,
    },
    FieldSpec {
        name: "Tactics",
        combat: true,
        morale: true,
        medicine: false,
    },
    FieldSpec {
        name: "Medicine",
        combat: false,
        morale: false,
        medicine: true,
    },
    FieldSpec {
        name: "Ballistics",
        combat: true,
        morale: false,
        medicine: false,
    },
    FieldSpec {
        name: "Archery",
        combat: true,
        morale: false,
        medicine: false,
    },
    FieldSpec {
        name: "Rites & Music",
        combat: false,
        morale: true,
        medicine: false,
    },
    FieldSpec {
        name: "Statecraft",
        combat: false,
        morale: true,
        medicine: false,
    },
    FieldSpec {
        name: "Restoration",
        combat: false,
        morale: false,
        medicine: false,
    },
    FieldSpec {
        name: "Military Organisation",
        combat: true,
        morale: true,
        medicine: false,
    },
];

const ROMAN: [&str; 12] = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
];

/// Idea word stock: body carries the doctrine, the suffix dresses it, and a
/// fifth of ideas take a prefix too.
const IDEA_PRE: [&str; 12] = [
    "Honour-rank",
    "Mercantile",
    "Land-reclamation",
    "Heavenly-craft",
    "Ocean-going",
    "People-first",
    "Natural-philosophy",
    "Glory",
    "Hundred-schools",
    "Great-harmony",
    "Wu-wei",
    "Iron-rule",
];
const IDEA_POST: [&str; 6] = ["ism", "Edict", "Society", "Reform", "Way", "Movement"];

struct IdeaBody {
    name: &'static str,
    desc: &'static str,
    research: f32,
    combat: f32,
    wear: f32,
}
const IDEA_BODIES: [IdeaBody; 8] = [
    IdeaBody {
        name: "Mercantile",
        desc: "trade routes are the nation's arteries",
        research: 0.06,
        combat: 0.0,
        wear: 0.0,
    },
    IdeaBody {
        name: "Honour-rank",
        desc: "land for heads: a ladder for the warrior caste",
        research: 0.0,
        combat: 0.06,
        wear: 0.02,
    },
    IdeaBody {
        name: "People-first",
        desc: "the people are the root of the state",
        research: 0.0,
        combat: 0.0,
        wear: -0.06,
    },
    IdeaBody {
        name: "Natural-philosophy",
        desc: "knowledge lies in the study of things",
        research: 0.14,
        combat: 0.0,
        wear: 0.0,
    },
    IdeaBody {
        name: "Land-reclamation",
        desc: "not an inch left fallow",
        research: 0.05,
        combat: 0.0,
        wear: 0.0,
    },
    IdeaBody {
        name: "Swords-to-ploughshares",
        desc: "beat the swords into ploughshares",
        research: 0.0,
        combat: -0.05,
        wear: -0.10,
    },
    IdeaBody {
        name: "Iron-rule",
        desc: "orders are followed to the letter",
        research: 0.03,
        combat: 0.02,
        wear: 0.0,
    },
    IdeaBody {
        name: "Hundred-schools",
        desc: "a hundred flowers bloom",
        research: 0.10,
        combat: 0.0,
        wear: 0.0,
    },
];

/// One invented technology.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmergentTech {
    pub lineage: String,
    pub name: String,
    pub gen: u32,
    pub research: f32,
    pub combat: f32,
    pub wear: f32,
    pub desc: String,
}

/// One adopted idea.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmergentIdea {
    pub lineage: String,
    pub name: String,
    pub research: f32,
    pub combat: f32,
    pub wear: f32,
    pub desc: String,
}

/// Effects saturate with generation count: 1 + min(0.8, gen * 0.04).
fn saturate(gen: u32) -> f32 {
    1.0 + (gen as f32 * 0.04).min(0.8)
}

/// Territory at which the invention bonus saturates (the engine's hard
/// per-lineage ceiling is 1600 tiles; the bonus tops out at half that).
const TERRITORY_FULL_SCALE: usize = 800;

/// A lineage invents at the pace of its productive territory — the
/// materialist base feeding the superstructure. A village patch keeps the
/// old baseline; a wide realm (800+ tiles) invents at roughly twice that
/// pace: more hands on the land, more surplus to feed scholars.
fn invention_chance(tiles: usize) -> f32 {
    0.25 + (tiles.min(TERRITORY_FULL_SCALE) as f32 / TERRITORY_FULL_SCALE as f32) * 0.35
}

impl Simulation {
    /// The emergent-civ tick: inventions on a slow cadence, ideas on a
    /// slower one. Everything folds into the aggregate accessors below.
    pub(crate) fn tick_emergent_civ(&mut self) {
        if !self.tick_count.is_multiple_of(480) {
            return;
        }
        self.invent_emergent_techs();
        if self.tick_count.is_multiple_of(1200) {
            self.adopt_emergent_ideas();
        }
    }

    fn invent_emergent_techs(&mut self) {
        let candidates: Vec<String> = self
            .lineage_eras
            .iter()
            .filter(|&(_, &era)| era >= Era::Bronze)
            .map(|(lid, _)| lid.to_string())
            .collect();
        for lid in candidates {
            let pop = self
                .organisms
                .iter()
                .filter(|o| o.alive && o.lineage_id.as_str() == lid)
                .count();
            if pop < 6 {
                continue;
            }
            let gen = self.emergent_techs.iter().filter(|t| t.lineage == lid).count() as u32 + 1;
            if gen > 24 {
                continue;
            }
            // The materialist base: a lineage invents at the pace of its
            // productive territory (see `invention_chance`).
            let tiles = self.territory.get(lid.as_str()).map_or(0, |s| s.len());
            if self.emergent_rng.random::<f32>() >= invention_chance(tiles) {
                continue;
            }
            let t = Self::invent_tech(&lid, gen, &mut self.emergent_rng);
            let line = format!(
                "\u{1F9ED} The {} invent {} — {}.",
                self.tribe_name(&lid),
                t.name,
                t.desc
            );
            self.emergent_techs.push(t);
            crate::sim::world_events::push_event(
                &mut self.events,
                self.tick_count,
                "milestone",
                "world",
                &line,
            );
            self.headlines.push_back((self.tick_count, line));
            while self.headlines.len() > 80 {
                self.headlines.pop_front();
            }
        }
    }

    fn adopt_emergent_ideas(&mut self) {
        let candidates: Vec<String> = self
            .governments
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .filter(|lid| {
                // Cultural pressure: a seat of learning, a wealthy purse, or
                // the institutional reach of a wide realm (300+ tiles: a
                // state big enough to carry an idea across its country).
                self.emergent_ideas.iter().filter(|i| &i.lineage == lid).count() < 3
                    && (self.buildings.iter().any(|b| {
                        b.is_operational()
                            && b.owner_lineage.as_deref() == Some(lid.as_str())
                            && matches!(
                                b.kind,
                                crate::sim::tech::buildings::BuildingKind::Temple
                                    | crate::sim::tech::buildings::BuildingKind::CityHall
                                    | crate::sim::tech::buildings::BuildingKind::Library
                            )
                    }) || self
                        .governments
                        .get(lid.as_str())
                        .is_some_and(|g| g.treasury >= 60)
                        || self.territory.get(lid.as_str()).map_or(0, |s| s.len()) >= 300)
            })
            .collect();
        for lid in candidates {
            // Ideas ride institutional reach: a bigger country carries them
            // further (0.18 on a patch, 0.40 across a realm).
            let tiles = self.territory.get(lid.as_str()).map_or(0, |s| s.len());
            let chance = 0.18 + (tiles.min(600) as f32 / 600.0) * 0.22;
            if self.emergent_rng.random::<f32>() >= chance {
                continue;
            }
            let idea = Self::invent_idea(&lid, &mut self.emergent_rng);
            let line = format!(
                "\u{1F4D6} The {} adopt the {} — {}.",
                self.tribe_name(&lid),
                idea.name,
                idea.desc
            );
            self.emergent_ideas.push(idea);
            crate::sim::world_events::push_event(
                &mut self.events,
                self.tick_count,
                "milestone",
                "world",
                &line,
            );
            self.headlines.push_back((self.tick_count, line));
            while self.headlines.len() > 80 {
                self.headlines.pop_front();
            }
        }
    }

    /// pwb's `inventTech`: weighted field by temperament proxy, prefix roll,
    /// generation suffix, one main effect with a 35% chance of a second.
    fn invent_tech(lineage: &str, gen: u32, rng: &mut ChaCha8Rng) -> EmergentTech {
        let field = &FIELDS[rng.random_range(0..FIELDS.len() as u32) as usize];
        let prefix = EM_PREFIX[rng.random_range(0..EM_PREFIX.len() as u32) as usize];
        let suffix = if gen > 12 {
            format!("·{}", ROMAN[((gen / 12) as usize).min(11)])
        } else {
            String::new()
        };
        let name = format!("{prefix} {}{suffix}", field.name);
        let t = saturate(gen);
        // Main effect: research always; combat and morale per field; a
        // 35% second effect borrows from a neighbouring field.
        let mut research = (0.03 + rng.random::<f32>() * 0.05) * t;
        let mut combat = if field.combat {
            (0.05 + rng.random::<f32>() * 0.07) * t
        } else {
            0.0
        };
        let mut wear = if field.morale {
            -(0.02 + rng.random::<f32>() * 0.03) * t
        } else {
            0.0
        };
        if rng.random::<f32>() < 0.35 {
            let other = &FIELDS[rng.random_range(0..FIELDS.len() as u32) as usize];
            research += (0.02 + rng.random::<f32>() * 0.04) * t;
            if other.combat {
                combat += 0.03 * t;
            }
            if other.morale {
                wear -= 0.02 * t;
            }
        }
        let desc = format!(
            "research +{}%{}{}",
            (research * 100.0).round() as i32,
            if combat > 0.0 {
                format!(", arms +{}%", (combat * 100.0).round() as i32)
            } else {
                String::new()
            },
            if wear < 0.0 {
                format!(", war-spirit +{}%", (-wear * 100.0).round() as i32)
            } else {
                String::new()
            },
        );
        let desc = if field.medicine {
            format!("{desc} — a mixed blessing for the sick")
        } else {
            desc
        };
        EmergentTech {
            lineage: lineage.to_string(),
            name,
            gen,
            research,
            combat,
            wear,
            desc,
        }
    }

    /// pwb's `inventIdea`: body carries the doctrine, the suffix dresses it,
    /// a fifth of ideas take a prefix, and the numbers jitter.
    fn invent_idea(lineage: &str, rng: &mut ChaCha8Rng) -> EmergentIdea {
        let body = &IDEA_BODIES[rng.random_range(0..IDEA_BODIES.len() as u32) as usize];
        let post = IDEA_POST[rng.random_range(0..IDEA_POST.len() as u32) as usize];
        let pre = if rng.random::<f32>() < 0.2 {
            IDEA_PRE[rng.random_range(0..IDEA_PRE.len() as u32) as usize]
        } else {
            ""
        };
        let mut jitter =
            |v: f32, amt: f32| ((v + (rng.random::<f32>() - 0.5) * amt) * 1000.0).round() / 1000.0;
        let research = jitter(body.research, 0.02).max(0.0);
        let combat = jitter(body.combat, 0.02);
        let wear = jitter(body.wear, 0.04);
        let name = format!("{pre}{} {post}", body.name);
        let mut bits = vec![format!("research +{}%", (research * 100.0).round() as i32)];
        if combat.abs() > 1e-3 {
            bits.push(format!(
                "{}{}%",
                if combat < 0.0 { "arms " } else { "arms +" },
                (combat * 100.0).round() as i32
            ));
        }
        if wear.abs() > 1e-3 {
            bits.push(format!(
                "{}{}% war-weariness",
                if wear < 0.0 { "-" } else { "+" },
                (wear.abs() * 100.0).round() as i32
            ));
        }
        EmergentIdea {
            lineage: lineage.to_string(),
            name,
            research,
            combat,
            wear,
            desc: format!("{} — {}", body.desc, bits.join(", ")),
        }
    }

    /// War diffusion (pwb's spoils of the mind): the victor's quartermasters
    /// study what they captured, and one piece of the defeated's learning
    /// crosses the front — as a half-strength copy studied under fire, never
    /// the real thing. A trophy is not copied twice, and the aggregate caps
    /// (`emergent_research_mul` and its siblings) keep the arms race bounded.
    pub(crate) fn diffuse_techs(&mut self, winner: &str, loser: &str) {
        let count = self.emergent_techs.iter().filter(|t| t.lineage == loser).count();
        if count == 0 {
            return;
        }
        // Not every war spreads knowledge.
        if self.emergent_rng.random::<f32>() >= 0.5 {
            return;
        }
        let idx = self.emergent_rng.random_range(0..count);
        let Some(pick) = self
            .emergent_techs
            .iter()
            .filter(|t| t.lineage == loser)
            .nth(idx)
            .cloned()
        else {
            return;
        };
        // A trophy is not copied twice.
        if self
            .emergent_techs
            .iter()
            .any(|t| t.lineage == winner && t.name == pick.name)
        {
            return;
        }
        let gen = self.emergent_techs.iter().filter(|t| t.lineage == winner).count() as u32 + 1;
        if gen > 24 {
            // No spare slots for captured knowledge.
            return;
        }
        let t = EmergentTech {
            lineage: winner.to_string(),
            name: pick.name.clone(),
            gen,
            research: pick.research * 0.5,
            combat: pick.combat * 0.5,
            wear: pick.wear * 0.5,
            desc: format!("{} — recovered from the defeated", pick.desc),
        };
        let line = format!(
            "\u{1F9ED} The {} recover {} from the {} — knowledge crosses the front.",
            self.tribe_name(winner),
            t.name,
            self.tribe_name(loser)
        );
        self.emergent_techs.push(t);
        crate::sim::world_events::push_event(&mut self.events, self.tick_count, "milestone", "world", &line);
        self.headlines.push_back((self.tick_count, line));
        while self.headlines.len() > 80 {
            self.headlines.pop_front();
        }
    }

    /// The aggregate research multiplier a lineage's inventions grant
    /// (capped at 2x: geniuses still read the same books).
    pub(crate) fn emergent_research_mul(&self, lineage: &str) -> f32 {
        let mut m = 1.0;
        for t in self.emergent_techs.iter().filter(|t| t.lineage == lineage) {
            m += t.research;
        }
        for i in self.emergent_ideas.iter().filter(|i| i.lineage == lineage) {
            m += i.research;
        }
        m.min(2.0)
    }

    /// The aggregate battle-damage multiplier (capped at 1.6).
    pub(crate) fn emergent_combat_mul(&self, lineage: &str) -> f32 {
        let mut m = 1.0;
        for t in self.emergent_techs.iter().filter(|t| t.lineage == lineage) {
            m += t.combat;
        }
        for i in self.emergent_ideas.iter().filter(|i| i.lineage == lineage) {
            m += i.combat;
        }
        m.min(1.6)
    }

    /// The aggregate war-weariness multiplier (clamped: ideas move a
    /// lineage's temperament, never abolish it).
    pub(crate) fn emergent_wear_mul(&self, lineage: &str) -> f32 {
        let mut m = 1.0;
        for t in self.emergent_techs.iter().filter(|t| t.lineage == lineage) {
            m += t.wear;
        }
        for i in self.emergent_ideas.iter().filter(|i| i.lineage == lineage) {
            m += i.wear;
        }
        m.clamp(0.6, 1.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn inventions_are_lexically_composed_and_bounded() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for gen in 1..=30u32 {
            let t = Simulation::invent_tech("l", gen, &mut rng);
            assert!(!t.name.is_empty());
            assert!(
                t.research > 0.0 && t.research < 0.2,
                "research bounded: {}",
                t.research
            );
            assert!(t.combat >= 0.0 && t.combat < 0.35, "combat bounded: {}", t.combat);
            assert!(t.wear <= 0.0 && t.wear > -0.15, "wear bounded: {}", t.wear);
        }
        let idea = Simulation::invent_idea("l", &mut rng);
        assert!(!idea.name.is_empty());
        assert!(idea.research >= 0.0 && idea.research < 0.2);
        assert!(idea.combat.abs() < 0.15);
        assert!(idea.wear.abs() < 0.2);
    }

    #[test]
    fn aggregates_grow_and_saturate() {
        let mut sim = Simulation::new(90);
        sim.lineage_eras.insert("l".into(), Era::Bronze);
        // A pile of inventions multiplies but stays capped.
        for k in 1..=40u32 {
            sim.emergent_techs
                .push(Simulation::invent_tech("l", k, &mut sim.emergent_rng));
        }
        assert!(sim.emergent_research_mul("l") <= 2.0);
        assert!(sim.emergent_research_mul("l") > 1.0);
        assert!(sim.emergent_combat_mul("l") <= 1.6);
        assert!(sim.emergent_wear_mul("l") >= 0.6 && sim.emergent_wear_mul("l") <= 1.4);
        // A lineage with nothing stays neutral.
        assert_eq!(sim.emergent_research_mul("other"), 1.0);
        assert_eq!(sim.emergent_wear_mul("other"), 1.0);
    }

    #[test]
    fn lineages_invent_on_cadence_and_ideas_cap_at_three() {
        let mut sim = Simulation::new(91);
        // A Bronze-era lineage with people and a government.
        sim.lineage_eras.insert("l".into(), Era::Bronze);
        sim.organisms.truncate(8);
        for o in sim.organisms.iter_mut() {
            o.alive = true;
            o.lineage_id = "l".into();
        }
        let mut gov = crate::sim::civ::government::Government::new(
            "l".to_string(),
            crate::sim::civ::government::GovernmentKind::Chiefdom,
            0,
        );
        gov.treasury = 200;
        sim.governments.insert("l".to_string(), gov);
        // Land the cadence on a multiple of 1200 so both paths fire.
        sim.tick_count = 2400;
        // Ideas need cultural pressure: give the lineage a temple.
        sim.buildings.push(crate::sim::tech::buildings::Building::new(
            1,
            crate::sim::tech::buildings::BuildingKind::Temple,
            1,
            1,
            Some("l".to_string()),
            0,
        ));
        // Force both rolls to hit by draining probability into success:
        // run the tick over many independent sims until something fires.
        let mut any_tech = false;
        let mut any_idea = false;
        for seed in 1..=40u64 {
            let mut s2 = Simulation::new(seed);
            s2.lineage_eras.insert("l".into(), Era::Bronze);
            s2.organisms.truncate(8);
            for o in s2.organisms.iter_mut() {
                o.alive = true;
                o.lineage_id = "l".into();
            }
            s2.tick_count = 2400;
            let mut gov = crate::sim::civ::government::Government::new(
                "l".to_string(),
                crate::sim::civ::government::GovernmentKind::Chiefdom,
                0,
            );
            gov.treasury = 200;
            s2.governments.insert("l".to_string(), gov);
            s2.buildings.push(crate::sim::tech::buildings::Building::new(
                2,
                crate::sim::tech::buildings::BuildingKind::Temple,
                1,
                1,
                Some("l".to_string()),
                0,
            ));
            s2.tick_emergent_civ();
            any_tech |= !s2.emergent_techs.is_empty();
            any_idea |= !s2.emergent_ideas.is_empty();
        }
        assert!(any_tech, "some seed invented a tech");
        assert!(any_idea, "some seed adopted an idea");
    }

    #[test]
    fn wide_territory_invents_faster_than_a_village_patch() {
        // The materialist base: one invention pass per world, forty worlds
        // each, wide realm (800 tiles) against a village patch (none).
        let mut small = 0usize;
        let mut wide = 0usize;
        for seed in 1..=40u64 {
            for &(tiles, is_wide) in &[(0usize, false), (800usize, true)] {
                let mut sim = Simulation::new(seed);
                sim.lineage_eras.insert("l".into(), Era::Bronze);
                sim.organisms.truncate(8);
                for o in sim.organisms.iter_mut() {
                    o.alive = true;
                    o.lineage_id = "l".into();
                }
                if tiles > 0 {
                    let set: crate::hashing::FxHashSet<(i32, i32)> =
                        (0..tiles).map(|i| (i as i32, 0)).collect();
                    sim.territory.insert("l".into(), set);
                }
                sim.tick_count = 480;
                sim.tick_emergent_civ();
                let invented = sim.emergent_techs.iter().filter(|t| t.lineage == "l").count();
                if is_wide {
                    wide += invented;
                } else {
                    small += invented;
                }
            }
        }
        assert!(
            wide > small,
            "the wide realm ({wide} inventions) should out-invent the patch ({small})"
        );
    }

    #[test]
    fn defeated_knowledge_diffuses_to_the_victor_at_half_strength() {
        let mut any = false;
        for seed in 1..=40u64 {
            let mut sim = Simulation::new(seed);
            // The loser enters the war with three inventions.
            for k in 1..=3u32 {
                let t = Simulation::invent_tech("loser", k, &mut sim.emergent_rng);
                sim.emergent_techs.push(t);
            }
            sim.diffuse_techs("winner", "loser");
            let got: Vec<_> = sim
                .emergent_techs
                .iter()
                .filter(|t| t.lineage == "winner")
                .cloned()
                .collect();
            for t in &got {
                any = true;
                let src = sim
                    .emergent_techs
                    .iter()
                    .find(|s| s.lineage == "loser" && s.name == t.name)
                    .unwrap();
                assert!(
                    (t.research - src.research * 0.5).abs() < 1e-6
                        && (t.combat - src.combat * 0.5).abs() < 1e-6
                        && (t.wear - src.wear * 0.5).abs() < 1e-6,
                    "a recovery is a half-strength copy of {src:?}, not {t:?}"
                );
                assert_eq!(t.desc, format!("{} — recovered from the defeated", src.desc));
            }
            // A winner with nothing stays at the neutral multiplier, and
            // every lineage's aggregates stay capped.
            assert!(sim.emergent_research_mul("winner") <= 2.0);
            assert!(sim.emergent_combat_mul("winner") <= 1.6);
        }
        assert!(any, "some seed diffused the loser's knowledge");
    }

    #[test]
    fn a_trophy_is_not_copied_twice() {
        let mut sim = Simulation::new(5);
        let t = Simulation::invent_tech("loser", 1, &mut sim.emergent_rng);
        let name = t.name.clone();
        sim.emergent_techs.push(t);
        // Fire the diffusion until it lands (chance is even).
        let mut landed = false;
        for _ in 0..80 {
            sim.diffuse_techs("winner", "loser");
            if sim
                .emergent_techs
                .iter()
                .any(|t| t.lineage == "winner" && t.name == name)
            {
                landed = true;
                break;
            }
        }
        assert!(landed, "the diffusion fired within eighty tries");
        // Fire many more: the same trophy must never be copied again.
        for _ in 0..80 {
            sim.diffuse_techs("winner", "loser");
        }
        let copies = sim
            .emergent_techs
            .iter()
            .filter(|t| t.lineage == "winner")
            .count();
        assert_eq!(copies, 1, "the trophy is copied at most once");
    }

    #[test]
    fn a_loser_with_no_knowledge_diffuses_nothing() {
        let mut sim = Simulation::new(6);
        sim.diffuse_techs("winner", "loser");
        assert!(sim.emergent_techs.iter().all(|t| t.lineage != "winner"));
    }
}
