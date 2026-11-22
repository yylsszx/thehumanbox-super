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
/// One battle's front press: (centre, radius, power, presser, pressed).
type FrontPress = ((i32, i32), i32, f32, String, String);

impl Simulation {
    /// One tick of the front: every active battle presses occupation onto
    /// the ground its opposite number holds around it, then the periodic
    /// decay pass keeps unmaintained fronts honest. Called in the war chain
    /// right after the battles resolve.
    pub(super) fn tick_occupation(&mut self) {
        let before = self.occupation.len();
        let presses: Vec<FrontPress> = self
            .battles
            .iter()
            .filter(|b| b.ended_tick.is_none())
            .filter_map(|b| {
                let presser = b.attackers.first()?.clone();
                let pressed = b.defenders.first()?.clone();
                if presser == pressed {
                    return None;
                }
                // The strength a side has present: each fighter mustered
                // adds a quarter unit of pressure, the field itself two.
                let here = (b.attacker_orgs.len() + b.defender_orgs.len()).min(10);
                let power = 2.0 + 0.25 * here as f32;
                Some((b.location, 3, power, presser, pressed))
            })
            .collect();
        for (loc, radius, power, presser, pressed) in presses {
            self.front_push(loc, radius, power, &presser, &pressed);
            self.front_push(loc, radius, power, &pressed, &presser);
        }
        self.front_decay();
        // The front's strength pattern feeds the territory wire view, so any
        // change this tick marks the view dirty.
        if self.occupation.len() != before || !self.battles.is_empty() {
            self.territory_revision = self.territory_revision.wrapping_add(1);
        }
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
        let mut sim = Simulation::new(87);
        let mut centre = (0, 0);
        'find: for y in 0..crate::world::grid::HEIGHT as i32 {
            for x in 0..crate::world::grid::WIDTH as i32 {
                if !matches!(sim.grid.get(x, y), Tile::Water | Tile::Void) {
                    centre = (x, y);
                    break 'find;
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
}
