//! The war occupation layer: the live, contestable front line of a war,
//! ported from pwb's `occCiv/occStr` system and fused with this engine's
//! conquest and treaty machinery.
//!
//! Where a decisive battle used to flip every tile in a radius outright,
//! ground now changes hands in stages:
//!
//!   * an active battle *presses* occupation strength onto the tiles the
//!     other side owns around it (a creeping front line, visible on the
//!     map as striped occupier colour);
//!   * strength accumulates while the occupier keeps a battle in reach and
//!     drains when the front moves on (`front_decay`);
//!   * at 100 the tile is annexed (owner flips, buildings change hands);
//!   * a decisive victory still annexes the ground it crushed outright,
//!     but leaves the fringe occupied rather than annexed — contested
//!     ground the loser can win back by counter-attacking;
//!   * at a war's end, held ground with strength ≥ 70 is annexed by the
//!     settlement and the rest reverts to its owner.
//!
//! Occupation is persistent war state and rides in the save; its jitter
//! comes from a private RNG stream (seeded from the world seed, never the
//! world RNG) so the world's rolls stay dense whether fronts exist or not.

use super::*;

/// Occupation strength at which a tile is annexed outright.
const ANNEX_AT: f32 = 100.0;
/// Occupation strength a war settlement annexes (the rest reverts).
const SETTLE_AT: f32 = 70.0;
/// Front pressure decays on this cadence.
const FRONT_DECAY_EVERY: u64 = 30;
/// Held: an occupier's own battle within this many tiles keeps the front.
const HELD_WITHIN: i32 = 7;
/// Contested: an enemy battle within this many tiles grinds the front down.
const CONTESTED_WITHIN: i32 = 3;
/// Cap on tracked occupied tiles; the weakest go when the front sprawls.
const MAX_OCCUPIED: usize = 4096;
/// How far from the overrun ground a victor scrounges its armouries.
const SCAVENGE_REACH: i32 = 25;
/// Chance per nearby adult to scavenge one piece of captured gear.
const SCAVENGE_CHANCE: f32 = 0.12;
/// At most this many fittings per annexation: a strip, not a redistribution.
const SCAVENGE_CAP: usize = 6;
/// An annexation smaller than this is a skirmish's worth of ground.
const SCAVENGE_MIN_TILES: usize = 3;
/// One battle's front press: (centre, radius, presser power, pressed power,
/// presser, pressed). Each side presses with its own armament.
type FrontPress = ((i32, i32), i32, f32, f32, String, String);
/// A battle scanned before its powers are computed: location, each side's
/// name and each side's fighter count.
type BattleScan = ((i32, i32), Option<String>, Option<String>, usize, usize);

impl Simulation {
    /// One tick of the front: every active battle presses occupation onto
    /// the ground its opposite number holds around it, then the periodic
    /// decay pass keeps unmaintained fronts honest. Called in the war chain
    /// right after the battles resolve.
    pub(super) fn tick_occupation(&mut self) {
        let before = self.occupation.len();
        // Collect first: the power reads need `self`, and `battles` is
        // borrowed by the scan.
        let raw: Vec<BattleScan> = self
            .battles
            .iter()
            .filter(|b| b.ended_tick.is_none())
            .map(|b| {
                (
                    b.location,
                    b.attackers.first().cloned(),
                    b.defenders.first().cloned(),
                    b.attacker_orgs.len(),
                    b.defender_orgs.len(),
                )
            })
            .collect();
        let mut presses: Vec<FrontPress> = Vec::new();
        for (loc, presser, pressed, att_n, def_n) in raw {
            let (Some(presser), Some(pressed)) = (presser, pressed) else {
                continue;
            };
            if presser == pressed {
                continue;
            }
            // Each side presses with the army it brought: its era arms it,
            // its doctrine shapes the grind (see `front_power`).
            let presser_power = self.front_power(&presser, att_n);
            let pressed_power = self.front_power(&pressed, def_n);
            presses.push((loc, 3, presser_power, pressed_power, presser, pressed));
        }
        for (loc, radius, presser_power, pressed_power, presser, pressed) in presses {
            self.front_push(loc, radius, presser_power, &presser, &pressed);
            self.front_push(loc, radius, pressed_power, &pressed, &presser);
        }
        self.front_decay();
        // The front's strength pattern feeds the territory wire view, so any
        // change this tick marks the view dirty.
        if self.occupation.len() != before || !self.battles.is_empty() {
            self.territory_revision = self.territory_revision.wrapping_add(1);
        }
    }

    /// The power one side presses a front with: each fighter mustered adds a
    /// quarter unit and the field two, then the army's armament scales it —
    /// a line armed for its era pushes ground a stone-age mob cannot (pwb's
    /// fronts moved at the speed of the army that pushed them), and a
    /// saturation doctrine grinds ground faster than a precision one.
    fn front_power(&self, presser: &str, fighters: usize) -> f32 {
        let base = 2.0 + 0.25 * fighters.min(10) as f32;
        let era: crate::sim::era::Era = self
            .lineage_eras
            .get(presser)
            .copied()
            .unwrap_or(crate::sim::era::Era::PreStone);
        // 1.0 at pre-stone, ~1.5 by medieval, ~2.1 by the information age.
        let era_mult = 1.0 + 0.12 * (era as u8 as f32);
        let doctrine_mult = self
            .weapon_doctrines
            .get(presser)
            .map_or(1.0, |d| 0.85 + d.saturation * 0.5);
        base * era_mult * doctrine_mult
    }

    /// Press `power` of occupation for `presser` onto tiles `pressed` owns
    /// within `radius` of `loc`. Own occupation accumulates (annexing at
    /// 100); fresh enemy-held ground is taken at 4..30 strength unless an
    /// enemy already holds it firmly (≥ 25: it must be ground down first).
    fn front_push(&mut self, loc: (i32, i32), radius: i32, power: f32, presser: &str, pressed: &str) {
        use rand::RngExt;
        let presser: crate::intern::Sym = presser.into();
        let pressed: crate::intern::Sym = pressed.into();
        let mut annex: Vec<(i32, i32)> = Vec::new();
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let cheb = dx.abs().max(dy.abs());
                let (x, y) = (loc.0 + dx, loc.1 + dy);
                if x < 0
                    || y < 0
                    || x >= crate::world::grid::WIDTH as i32
                    || y >= crate::world::grid::HEIGHT as i32
                {
                    continue;
                }
                if matches!(self.grid.get(x, y), Tile::Water | Tile::Void) {
                    continue;
                }
                let owner = match self.tile_owner.get(&(x, y)) {
                    Some(owner) => owner,
                    None => continue,
                };
                if *owner != pressed && *owner != presser {
                    // A third party's sovereign land: not this front's business.
                    continue;
                }
                let falloff = 1.0 - (cheb as f32 / (radius as f32 + 1.0)) * 0.75;
                let pw = (power * falloff).max(1.0);
                match self.occupation.get_mut(&(x, y)) {
                    Some((occ, strength)) if *occ == presser => {
                        *strength += pw + self.occupation_rng.random::<f32>() * 3.0;
                        if *strength >= ANNEX_AT {
                            annex.push((x, y));
                        }
                    }
                    entry => {
                        let contested = entry.is_some();
                        let enemy_ground = *owner == pressed;
                        // Ground the enemy occupies firmly has to be ground
                        // down before it can be pressed; free enemy ground
                        // is taken at a token strength a counter-attack can
                        // shrug off — unless the force pressing it is
                        // overwhelming, which annexes on the spot.
                        if enemy_ground && !contested {
                            let start = if pw >= ANNEX_AT {
                                pw
                            } else {
                                4.0 + pw.min(26.0) + self.occupation_rng.random::<f32>() * 3.0
                            };
                            self.occupation.insert((x, y), (presser, start.min(ANNEX_AT)));
                            if start >= ANNEX_AT {
                                annex.push((x, y));
                            }
                        }
                    }
                }
            }
        }
        if !annex.is_empty() {
            self.annex_tiles(presser.as_str(), &annex);
        }
    }

    /// The periodic pass: a front the occupier still contests holds, one an
    /// enemy fights over grinds down, and an abandoned one drains away.
    fn front_decay(&mut self) {
        if !self.tick_count.is_multiple_of(FRONT_DECAY_EVERY) {
            return;
        }
        // Active fronts with their sides, precollected so the scan below
        // borrows nothing mutable.
        let fronts: Vec<((i32, i32), Vec<String>)> = self
            .battles
            .iter()
            .filter(|b| b.ended_tick.is_none())
            .map(|b| {
                (
                    b.location,
                    b.attackers.iter().chain(b.defenders.iter()).cloned().collect(),
                )
            })
            .collect();
        let mut updates: Vec<((i32, i32), f32)> = Vec::new();
        let mut annex: Vec<(String, Vec<(i32, i32)>)> = Vec::new();
        let mut revert: Vec<(i32, i32)> = Vec::new();
        for (&tile, &(occ, strength)) in self.occupation.iter() {
            let occ_s = occ.as_str();
            let mut next = strength;
            let mut held = false;
            let mut contested = false;
            for &(loc, ref sides) in fronts.iter() {
                let cheb = (loc.0 - tile.0).abs().max((loc.1 - tile.1).abs());
                if cheb > HELD_WITHIN {
                    continue;
                }
                if sides.iter().any(|s| s == occ_s) {
                    // The occupier fights on near this tile: the front holds.
                    next += 2.0;
                    held = true;
                    break;
                }
                if cheb <= CONTESTED_WITHIN {
                    contested = true;
                }
            }
            if !held {
                next -= if contested { 16.0 } else { 4.0 };
            }
            if next <= 0.0 {
                revert.push(tile);
            } else if next >= ANNEX_AT {
                match annex.iter_mut().find(|(lid, _)| lid == occ_s) {
                    Some((_, tiles)) => tiles.push(tile),
                    None => annex.push((occ_s.to_string(), vec![tile])),
                }
            } else {
                updates.push((tile, next));
            }
        }
        for (tile, strength) in updates {
            if let Some((_, slot)) = self.occupation.get_mut(&tile) {
                *slot = strength;
            }
        }
        for tile in revert {
            self.occupation.swap_remove(&tile);
        }
        for (lid, tiles) in annex {
            self.annex_tiles(&lid, &tiles);
        }
        self.trim_occupation();
    }

    /// Keep the tracked front bounded: past `MAX_OCCUPIED` tiles the weakest
    /// holdings go, ties by tile for determinism.
    fn trim_occupation(&mut self) {
        if self.occupation.len() <= MAX_OCCUPIED {
            return;
        }
        let excess = self.occupation.len() - MAX_OCCUPIED;
        let mut ranked: Vec<(u32, i32, i32)> = self
            .occupation
            .iter()
            .map(|(&(x, y), &(_, s))| (s.to_bits(), x, y))
            .collect();
        ranked.select_nth_unstable(excess);
        for &(_, x, y) in ranked.iter().take(excess) {
            self.occupation.swap_remove(&(x, y));
        }
    }

    /// Occupation spoils (pwb's overrun armouries): annexing another
    /// lineage's ground strips its stores. The victor's adults near the
    /// overrun ground scavenge gear rolled at the *loser's* armament tier
    /// — captured arms, not the winner's own patterns, so a conquest can
    /// outclass the home forge. Bounded: a small chance per nearby adult,
    /// at most a handful of fittings per annexation.
    fn occupation_spoils(&mut self, from: &str, to: &str, tiles: &[(i32, i32)]) {
        use rand::RngExt;
        if tiles.len() < SCAVENGE_MIN_TILES {
            // A skirmish's worth of ground: nothing worth stripping.
            return;
        }
        let loser_tier = self
            .lineage_eras
            .get(from)
            .copied()
            .map(|e| crate::organism::items::era_tier(e as u8))
            .unwrap_or(0);
        if loser_tier == 0 {
            // A pre-metal loser has no armouries to overrun.
            return;
        }
        let cx = tiles.iter().map(|t| t.0).sum::<i32>() / tiles.len() as i32;
        let cy = tiles.iter().map(|t| t.1).sum::<i32>() / tiles.len() as i32;
        let mut fitted: Vec<(usize, &'static crate::organism::items::ItemDef)> = Vec::new();
        for (i, o) in self.organisms.iter().enumerate() {
            if !o.alive
                || o.lineage_id.as_str() != to
                || (o.x as i32 - cx).abs().max((o.y as i32 - cy).abs()) > SCAVENGE_REACH
                || !crate::sim::age_stage::AgeStage::from_age(o.age, o.max_age).can_combat()
            {
                continue;
            }
            if self.rng.random::<f32>() < SCAVENGE_CHANCE {
                if let Some(def) = crate::organism::items::roll_item(&mut self.rng, loser_tier) {
                    fitted.push((i, def));
                    if fitted.len() >= SCAVENGE_CAP {
                        break;
                    }
                }
            }
        }
        for (i, def) in fitted {
            if crate::organism::items::equip(&mut self.organisms[i].gear, def).is_some() {
                let name = self.organisms[i].name.clone();
                crate::sim::world_events::push_event(
                    &mut self.events,
                    self.tick_count,
                    "loot",
                    &name,
                    &format!("scavenged {} from the overrun armouries", def.name),
                );
            }
        }
    }

    /// A war settlement's front resolution: ground held with real strength
    /// (≥ 70) is annexed by the side standing on it; the rest reverts to its
    /// owner, exactly as pwb's `settleFront` did at the peace table. Returns
    /// the number of tiles annexed.
    pub(crate) fn settle_fronts(&mut self, a: &str, b: &str) -> u32 {
        let mut annex: Vec<(String, Vec<(i32, i32)>)> = Vec::new();
        let mut revert: Vec<(i32, i32)> = Vec::new();
        for (&tile, &(occ, strength)) in self.occupation.iter() {
            let occ_s = occ.as_str();
            if occ_s != a && occ_s != b {
                continue;
            }
            if strength >= SETTLE_AT {
                match annex.iter_mut().find(|(lid, _)| lid == occ_s) {
                    Some((_, tiles)) => tiles.push(tile),
                    None => annex.push((occ_s.to_string(), vec![tile])),
                }
            } else {
                revert.push(tile);
            }
        }
        for tile in revert {
            self.occupation.swap_remove(&tile);
        }
        let mut annexed = 0u32;
        for (lid, tiles) in annex {
            annexed += tiles.len() as u32;
            self.annex_tiles(&lid, &tiles);
        }
        annexed
    }

    /// Annex occupied tiles to `to`: ownership flips (per prior owner), the
    /// occupation entry drops, and buildings on the ground change hands to
    /// the occupier's nearest town.
    pub(super) fn annex_tiles(&mut self, to: &str, tiles: &[(i32, i32)]) {
        if tiles.is_empty() {
            return;
        }
        let mut territory_dirty = false;
        // Ground the occupier already owns just drops its occupation entry.
        let mut by_from: HashMap<String, Vec<(i32, i32)>> = HashMap::default();
        let mut already_mine: Vec<(i32, i32)> = Vec::new();
        for &tile in tiles {
            self.occupation.swap_remove(&tile);
            match self.tile_owner.get(&tile) {
                Some(owner) if owner.as_str() == to => already_mine.push(tile),
                Some(owner) => by_from.entry(owner.as_str().to_string()).or_default().push(tile),
                None => by_from.entry(String::new()).or_default().push(tile),
            }
        }
        for (from, group) in by_from {
            if from.is_empty() {
                let set = self.territory.entry(to.into()).or_default();
                for t in group {
                    set.insert(t);
                    self.tile_owner.insert(t, to.into());
                }
                territory_dirty = true;
            } else {
                self.move_tiles(&from, to, &group);
                territory_dirty = true;
                // The overrun armouries are stripped before the maps are
                // redrawn: captured gear joins the victor's kit.
                self.occupation_spoils(&from, to, &group);
            }
        }
        let _ = already_mine;
        // Buildings on annexed ground change hands.
        let taken_set: crate::hashing::FxHashSet<(i32, i32)> = tiles.iter().copied().collect();
        let winner_towns: Vec<(u32, crate::sim::civ::civ_tick::town::Town)> = self
            .towns
            .iter()
            .filter(|t| t.lineage == to)
            .map(|t| (t.id, t.clone()))
            .collect();
        let mut captured = false;
        for b in self.buildings.iter_mut() {
            if taken_set.contains(&(b.x, b.y)) {
                b.owner_lineage = Some(to.into());
                b.town = winner_towns
                    .iter()
                    .map(|(id, t)| (t.distance(b.x, b.y), *id))
                    .filter(|&(d, _)| d <= crate::sim::civ::civ_tick::town::PARENT_REACH)
                    .min()
                    .map(|(_, id)| id);
                captured = true;
            }
        }
        if captured {
            self.building_state_revision = self.building_state_revision.wrapping_add(1);
        }
        if territory_dirty {
            self.territory_revision = self.territory_revision.wrapping_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sim with one lineage holding a disc of land, for pressing fronts.
    fn sim_with_land() -> (Simulation, String, (i32, i32)) {
        sim_with_land_seed(87)
    }

    fn sim_with_land_seed(seed: u64) -> (Simulation, String, (i32, i32)) {
        let mut sim = Simulation::new(seed);
        let mut centre = (0, 0);
        'find: for y in 3..crate::world::grid::HEIGHT as i32 - 3 {
            for x in 3..crate::world::grid::WIDTH as i32 - 3 {
                // Prefer solid ground: a centre whose 7×7 neighbourhood is
                // all land (the first land tile in a scan is a coastline,
                // and an annexation pressed there may not have three land
                // tiles to its name).
                let land = (0..7).flat_map(|dy| (0..7).map(move |dx| (dy, dx))).filter(|&(dy, dx)| {
                    !matches!(sim.grid.get(x + dx - 3, y + dy - 3), Tile::Water | Tile::Void)
                }).count();
                if land == 49 {
                    centre = (x, y);
                    break 'find;
                }
            }
        }
        if centre == (0, 0) {
            'fallback: for y in 0..crate::world::grid::HEIGHT as i32 {
                for x in 0..crate::world::grid::WIDTH as i32 {
                    if !matches!(sim.grid.get(x, y), Tile::Water | Tile::Void) {
                        centre = (x, y);
                        break 'fallback;
                    }
                }
            }
        }
        sim.claim_territory("holder", centre.0, centre.1, 6);
        (sim, "holder".to_string(), centre)
    }

    #[test]
    fn a_strong_press_annexes_the_core_and_occupies_the_fringe() {
        let (mut sim, lid, (cx, cy)) = sim_with_land();
        let core: Vec<(i32, i32)> = sim
            .territory
            .get(lid.as_str())
            .unwrap()
            .iter()
            .copied()
            .filter(|&(x, y)| (x - cx).abs().max((y - cy).abs()) <= 1)
            .collect();
        sim.front_push((cx, cy), 4, 120.0, "invader", lid.as_str());
        let invader_tiles = sim.territory.get("invader").map(|s| s.len()).unwrap_or(0);
        assert!(invader_tiles > 0, "the pressed core was annexed to the invader");
        for t in core.iter().take(3) {
            assert!(
                sim.tile_owner.get(t).is_some_and(|o| o.as_str() == "invader"),
                "core tile {t:?} now belongs to the invader"
            );
        }
        assert!(!sim.occupation.is_empty(), "the fringe is under occupation");
    }

    #[test]
    fn a_weak_press_settles_at_the_peace_table_as_reversion() {
        let (mut sim, lid, (cx, cy)) = sim_with_land();
        sim.front_push((cx, cy), 2, 6.0, "invader", lid.as_str());
        assert!(!sim.occupation.is_empty(), "fringe occupied");
        sim.settle_fronts("invader", lid.as_str());
        assert!(
            sim.occupation.is_empty(),
            "a weak front reverts entirely at the peace table"
        );
        let holder = sim.territory.get(lid.as_str()).map(|s| s.len()).unwrap_or(0);
        assert!(holder > 0, "the holder kept its land");
    }

    #[test]
    fn an_iron_line_presses_harder_than_a_stone_age_mob() {
        let (mut sim, _lid, _centre) = sim_with_land();
        // No era entry: pre-stone. The same fighter count, armed for iron,
        // must press clearly harder (pwb's fronts moved at the speed of the
        // army that pushed them).
        let stone = sim.front_power("invader", 10);
        sim.lineage_eras
            .insert("invader".into(), crate::sim::era::Era::Iron);
        let iron = sim.front_power("invader", 10);
        assert!(
            iron > stone * 1.3,
            "iron {iron} should press far harder than pre-stone {stone}"
        );
    }

    #[test]
    fn a_saturation_doctrine_grinds_ground_faster_than_precision() {
        let (mut sim, _lid, _centre) = sim_with_land();
        sim.weapon_doctrines.insert(
            "saturation".into(),
            crate::sim::civ::warfare::WeaponDoctrine {
                precision: 0.2,
                saturation: 1.0,
                terror: 0.2,
                attrition: 0.2,
            },
        );
        let precision = sim.front_power("no_doctrine", 10);
        let saturation = sim.front_power("saturation", 10);
        assert!(
            saturation > precision,
            "saturation {saturation} should grind faster than the doctrine-less {precision}"
        );
    }

    #[test]
    fn annexing_ground_scavenges_the_loser_s_armouries() {
        // Statistical, in the suite's house style: over forty worlds, some
        // victor near the overrun ground walks away in captured gear.
        let mut any = false;
        for seed in 1..=40u64 {
            let (mut sim, lid, (cx, cy)) = sim_with_land_seed(seed);
            // An iron-age loser has armouries worth stripping; the victor
            // stands on the overrun ground, adult and empty-handed.
            sim.lineage_eras.insert(lid.clone().into(), crate::sim::era::Era::Iron);
            sim.organisms.truncate(16);
            for o in sim.organisms.iter_mut() {
                o.alive = true;
                o.lineage_id = "invader".into();
                o.age = 30;
                o.max_age = 90;
                o.x = cx as f32;
                o.y = cy as f32;
                o.gear.clear();
            }
            // An overwhelming press annexes the core outright (≥ 3 tiles,
            // so the armouries are worth the trip).
            sim.front_push((cx, cy), 3, 200.0, "invader", &lid);
            let annexed = sim.territory.get("invader").map_or(0, |s| s.len());
            if annexed < 3 {
                continue;
            }
            let looted = sim
                .events
                .iter()
                .any(|e| e.etype == "loot" && e.detail.contains("overrun armouries"));
            let geared = sim.organisms.iter().any(|o| !o.gear.is_empty());
            any |= looted && geared;
        }
        assert!(any, "some seed's victors scavenged captured gear");
    }
}
