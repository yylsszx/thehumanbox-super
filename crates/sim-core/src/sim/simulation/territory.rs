use super::*;
use serde_json::json;

/// How often (in ticks) the organic growth pass runs. At the base 120 ms
/// tick that is a visible creep every 3 seconds; a 40x world redraws its
/// borders roughly twice a second of real time.
pub(super) const TERRITORY_GROWTH_STEP: u64 = 25;
/// Frontier queues stay bounded: past this the freshest candidates wait for
/// a rebuild rather than piling up.
const MAX_FRONTIER: usize = 2048;
/// Growth candidates deferred back to the queue by chance, so fronts creep
/// unevenly instead of sweeping a perfect circle.
const DEFER_CHANCE: f32 = 0.35;
/// Hard ceiling on one lineage's territory, whatever its population.
const MAX_TERRITORY_ANY: usize = 1600;
/// The patch a tribe holds before anyone has discovered borders or territory.
const UNDISCOVERED_CAP: usize = 28;
/// How fast a dead tribe's borders dissolve: tiles released per pass.
const DISSOLVE_PER_PASS: usize = 8;

impl Simulation {
    /// Whether the tile at (x, y) is land `lid` may set foot on as a
    /// destination: unclaimed ground, or its own sovereign ground. Another
    /// lineage's claimed tile is theirs to roam — wander and work targets
    /// respect the border instead of drifting through a neighbour's country.
    pub(crate) fn land_is_open_to(&self, lid: &crate::intern::Sym, x: i32, y: i32) -> bool {
        self.tile_owner.get(&(x, y)).is_none_or(|owner| *owner == *lid)
    }

    /// A completed building plants its lineage's flag: unclaimed ground the
    /// structure stands on joins that lineage's territory, and the frontier
    /// grows outward from it. Territory display therefore follows the real
    /// settlements instead of only wherever the founding camp happened to be.
    pub(crate) fn seed_territory_at(&mut self, lid: &crate::intern::Sym, x: i32, y: i32) {
        if !crate::world::grid::WorldGrid::in_bounds(x, y) {
            return;
        }
        if matches!(self.grid.get(x, y), Tile::Water | Tile::Void) {
            return;
        }
        if self.tile_owner.contains_key(&(x, y)) {
            // Already someone's sovereign ground — conquest's job, not a
            // builder's.
            return;
        }
        let set = self.territory.entry(*lid).or_default();
        if set.insert((x, y)) {
            self.tile_owner.insert((x, y), *lid);
            self.push_frontier_candidates(lid, (x, y));
            self.territory_revision = self.territory_revision.wrapping_add(1);
        }
    }

    pub(super) fn tick_settlements(&mut self) {
        crate::sim::civ::settlements::tick(self);
    }

    /// Claim all non-water tiles within `radius` of `(cx, cy)` for lineage `lid`.
    ///
    /// This is the *explicit* claim: a founding camp, the player's claim-land
    /// and establish-borders commands, and the settlement-founding seed. It is
    /// bounded by the disc asked for; the per-lineage size caps live in the
    /// growth pass, which grows or trims each territory toward what its people
    /// can actually hold. Newly claimed tiles feed the lineage's frontier
    /// queue so organic growth continues from the new border.
    pub(crate) fn claim_territory(&mut self, lid: &str, cx: i32, cy: i32, radius: i32) {
        // Pre-compute the tile list so we can update both maps without
        // holding two mutable borrows on `self` simultaneously.
        let mut to_claim: Vec<(i32, i32)> = Vec::new();
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > radius * radius {
                    continue;
                }
                let tx = (cx + dx).clamp(0, crate::world::grid::WIDTH as i32 - 1);
                let ty = (cy + dy).clamp(0, crate::world::grid::HEIGHT as i32 - 1);
                if matches!(self.grid.get(tx, ty), Tile::Water | Tile::Void) {
                    continue;
                }
                to_claim.push((tx, ty));
            }
        }
        let lid_sym: crate::intern::Sym = lid.into();
        if !self.territory.contains_key(&lid_sym) {
            self.territory.insert(lid_sym, Default::default());
        }
        let Some(tiles) = self.territory.get_mut(&lid_sym) else {
            return;
        };
        for p in &to_claim {
            tiles.insert(*p);
        }
        // Update the inverse map. New claims overwrite (most-recent wins).
        for p in to_claim {
            self.tile_owner.insert(p, lid_sym);
        }
        // The claimed disc is the new border growth continues from: refill
        // the frontier queue from it so organic expansion resumes here.
        self.rebuild_territory_frontier(&lid_sym);
        self.territory_revision = self.territory_revision.wrapping_add(1);
    }

    /// One pass of organic territory growth: every living lineage creeps
    /// outward along its own border toward the land its people can hold, and
    /// a dead lineage's borders dissolve. Called on the
    /// `TERRITORY_GROWTH_STEP` cadence from the main tick.
    pub(super) fn tick_territory_growth(&mut self) {
        // Per-lineage census: people alive, and whether anyone has the
        // borders/territory discovery (they know where the tribe ends).
        let mut census: HashMap<crate::intern::Sym, (u32, bool)> = HashMap::default();
        for org in self.organisms.iter().filter(|o| o.alive) {
            let e = census.entry(org.lineage_id).or_default();
            e.0 += 1;
            e.1 |= org.discoveries.has(Hot::Territory) || org.discoveries.has(Hot::Borders);
        }
        // Sorted ids: the pass is deterministic run to run.
        let mut lids: Vec<crate::intern::Sym> = self.territory.keys().cloned().collect();
        lids.sort_unstable_by(|a, b| a.as_str().cmp(b.as_str()));
        for lid in lids {
            let (pop, has_borders) = census.get(&lid).copied().unwrap_or((0, false));
            let len = self.territory.get(&lid).map_or(0, |s| s.len());
            if pop == 0 {
                // The tribe is gone: its borders dissolve slowly, the
                // farthest marches first, until nothing is left.
                if len > 0 {
                    self.evict_territory_farthest(&lid, DISSOLVE_PER_PASS.min(len));
                }
                continue;
            }
            let cap = if has_borders {
                (24 + pop as usize * 6).min(MAX_TERRITORY_ANY)
            } else {
                UNDISCOVERED_CAP
            };
            if len > cap {
                // The people can no longer hold this much land: let go of
                // the farthest marches.
                self.evict_territory_farthest(&lid, (len - cap).min(24));
            } else if len < cap {
                let want = 1 + pop as usize / 12;
                self.grow_territory(&lid, want);
            }
        }
    }

    /// Grow `lid`'s territory by up to `want` tiles, popping unowned
    /// candidates off its frontier queue. Candidates may be deferred back
    /// (see `DEFER_CHANCE`) so the border creeps unevenly, the way WorldBox
    /// borders do, rather than sweeping a perfect circle.
    fn grow_territory(&mut self, lid: &crate::intern::Sym, want: usize) {
        use rand::RngExt;
        let mut grown = 0usize;
        let mut attempts = 0usize;
        let max_attempts = want * 4 + 8;
        while grown < want && attempts < max_attempts {
            attempts += 1;
            // Frontiers run dry: refill from the current border before
            // giving up on this pass.
            let candidate = match self.territory_frontiers.get_mut(lid).and_then(|q| q.pop_front()) {
                Some(c) => c,
                None => {
                    self.rebuild_territory_frontier(lid);
                    match self.territory_frontiers.get_mut(lid).and_then(|q| q.pop_front()) {
                        Some(c) => c,
                        None => break, // sealed: water, map edge or neighbours all round
                    }
                }
            };
            // Validate the candidate against the world as it is now.
            if !crate::world::grid::WorldGrid::in_bounds(candidate.0, candidate.1) {
                continue;
            }
            if matches!(self.grid.get(candidate.0, candidate.1), Tile::Water | Tile::Void) {
                continue;
            }
            match self.tile_owner.get(&candidate) {
                Some(owner) if *owner == *lid => continue, // already mine
                Some(_) => continue,                       // another tribe's sovereign land: conquest's job
                None => {}
            }
            // Organic front: some candidates are deferred so the creep is
            // uneven. Deferred tiles go to the back of the queue.
            if self.territory_rng.random::<f32>() < DEFER_CHANCE {
                if let Some(q) = self.territory_frontiers.get_mut(lid) {
                    if q.len() < MAX_FRONTIER {
                        q.push_back(candidate);
                    }
                }
                continue;
            }
            // Claim it.
            if let Some(set) = self.territory.get_mut(lid) {
                set.insert(candidate);
            }
            self.tile_owner.insert(candidate, *lid);
            self.territory_revision = self.territory_revision.wrapping_add(1);
            grown += 1;
            // The new tile's own unowned neighbours are the next frontier.
            self.push_frontier_candidates(lid, candidate);
        }
    }

    /// Push the unowned, walkable 4-neighbours of `at` onto `lid`'s frontier
    /// queue. Candidates are validated again when popped, so stale entries
    /// are harmless.
    pub(super) fn push_frontier_candidates(&mut self, lid: &crate::intern::Sym, at: (i32, i32)) {
        const NEIGHBOURS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        let mut fresh: Vec<(i32, i32)> = Vec::new();
        for (dx, dy) in NEIGHBOURS {
            let (x, y) = (at.0 + dx, at.1 + dy);
            if !crate::world::grid::WorldGrid::in_bounds(x, y) {
                continue;
            }
            if matches!(self.grid.get(x, y), Tile::Water | Tile::Void) {
                continue;
            }
            if self.tile_owner.contains_key(&(x, y)) {
                continue;
            }
            fresh.push((x, y));
        }
        if fresh.is_empty() {
            return;
        }
        let q = self.territory_frontiers.entry(*lid).or_default();
        for p in fresh {
            if q.len() >= MAX_FRONTIER {
                break;
            }
            q.push_back(p);
        }
    }

    /// Refill `lid`'s frontier queue from its current border. Iteration is
    /// sorted so every rebuild produces the identical queue - the growth
    /// pass stays deterministic after a save and load.
    pub(crate) fn rebuild_territory_frontier(&mut self, lid: &crate::intern::Sym) {
        let Some(tiles) = self.territory.get(lid) else {
            self.territory_frontiers.swap_remove(lid);
            return;
        };
        let mut sorted: Vec<(i32, i32)> = tiles.iter().copied().collect();
        sorted.sort_unstable();
        if let Some(q) = self.territory_frontiers.get_mut(lid) {
            q.clear();
        } else {
            self.territory_frontiers.insert(*lid, Default::default());
        }
        for t in sorted {
            self.push_frontier_candidates(lid, t);
        }
    }

    /// Release the `count` tiles of `lid`'s territory farthest from its
    /// centre. Ties by distance break by coordinates, never by set order:
    /// the same world must evict the same tiles from run to run.
    fn evict_territory_farthest(&mut self, lid: &crate::intern::Sym, count: usize) {
        let Some(tiles) = self.territory.get(lid) else {
            return;
        };
        if tiles.is_empty() || count == 0 {
            return;
        }
        let (mut sx, mut sy) = (0i64, 0i64);
        for &(x, y) in tiles.iter() {
            sx += x as i64;
            sy += y as i64;
        }
        let n = tiles.len() as i64;
        let (cx, cy) = (sx / n, sy / n);
        let mut ranked: Vec<(i64, i32, i32)> = tiles
            .iter()
            .map(|&(x, y)| {
                let dx = x as i64 - cx;
                let dy = y as i64 - cy;
                (dx * dx + dy * dy, x, y)
            })
            .collect();
        // Farthest first; ties by coordinates, never by set order.
        ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        ranked.truncate(count);
        for (_, x, y) in ranked {
            if let Some(set) = self.territory.get_mut(lid) {
                set.swap_remove(&(x, y));
            }
            // Evictions only clear the inverse entry if it was owned by
            // *this* lineage - another lineage may have a more recent claim.
            if let Some(owner) = self.tile_owner.get(&(x, y)) {
                if *owner == *lid {
                    self.tile_owner.swap_remove(&(x, y));
                }
            }
        }
        self.territory_revision = self.territory_revision.wrapping_add(1);
    }

    /// Rebuild the wire view of territory: per-lineage claimed tiles, the
    /// contested tiles two or more lineages claim, and the war occupation
    /// front grouped per occupier. Called on the slow cadence and, whenever
    /// territory changed, on the hot ship cadence.
    pub(crate) fn rebuild_territory_cache(&mut self) {
        // Contested tiles: any tile claimed by 2+ lineages.
        let mut tile_claim_count: HashMap<(i32, i32), u32> = HashMap::default();
        for tiles in self.territory.values() {
            for &tile in tiles {
                *tile_claim_count.entry(tile).or_insert(0) += 1;
            }
        }
        let contested: Vec<[i32; 2]> = tile_claim_count
            .into_iter()
            .filter(|(_, c)| *c >= 2)
            .map(|((x, y), _)| [x, y])
            .collect();
        // The war occupation front, grouped per occupier: tile + strength
        // (0..100). The client paints held ground in the occupier's colour
        // at a strength-scaled striped overlay.
        let mut occupied_json: Vec<serde_json::Value> = Vec::new();
        {
            let mut by_occupier: HashMap<&crate::intern::Sym, Vec<[f32; 3]>> = HashMap::default();
            for (tile, (lid, strength)) in self.occupation.iter() {
                by_occupier
                    .entry(lid)
                    .or_default()
                    .push([tile.0 as f32, tile.1 as f32, *strength]);
            }
            for (lid, tiles) in by_occupier {
                occupied_json.push(json!({"lid": lid, "tiles": tiles}));
            }
        }
        self.cached_territory = serde_json::json!({
            "claimed": self.territory.iter()
                .map(|(lid, tiles)| {
                    let pts: Vec<[i32;2]> = tiles.iter().map(|&(x,y)| [x,y]).collect();
                    json!({"lid": lid, "tiles": pts})
                }).collect::<Vec<_>>(),
            "contested": contested,
            "occupied": occupied_json,
        });
    }

    /// The claim as it was before borrowing the lineage id and picking the
    /// evicted tiles instead of sorting them all, kept to check the new one
    /// against.
    #[cfg(test)]
    pub(super) fn claim_territory_reference(&mut self, lid: &str, cx: i32, cy: i32, radius: i32) {
        // Pre-compute the tile list so we can update both maps without
        // holding two mutable borrows on `self` simultaneously.
        let mut to_claim: Vec<(i32, i32)> = Vec::new();
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > radius * radius {
                    continue;
                }
                let tx = (cx + dx).clamp(0, crate::world::grid::WIDTH as i32 - 1);
                let ty = (cy + dy).clamp(0, crate::world::grid::HEIGHT as i32 - 1);
                if matches!(self.grid.get(tx, ty), Tile::Water | Tile::Void) {
                    continue;
                }
                to_claim.push((tx, ty));
            }
        }
        let lid_sym: crate::intern::Sym = lid.into();
        let tiles = self.territory.entry(lid_sym).or_default();
        for p in &to_claim {
            tiles.insert(*p);
        }
        for p in to_claim {
            self.tile_owner.insert(p, lid_sym);
        }
    }

    pub(super) fn apply_conquests(&mut self) {
        use crate::sim::warfare::BattleOutcome;
        let now = self.tick_count;
        let conquests: Vec<(String, String, (i32, i32))> = self
            .battles
            .iter()
            .filter(|b| b.ended_tick == Some(now))
            .filter_map(|b| {
                let (loser, winner) = match b.outcome {
                    Some(BattleOutcome::AttackerVictory) => (b.defenders.first()?, b.attackers.first()?),
                    Some(BattleOutcome::DefenderVictory) => (b.attackers.first()?, b.defenders.first()?),
                    _ => return None,
                };
                if loser == winner {
                    return None;
                }
                Some((loser.clone(), winner.clone(), b.location))
            })
            .collect();
        for (loser, winner, (lx, ly)) in conquests {
            let Some(loser_tiles) = self.territory.get(loser.as_str()).cloned() else {
                continue;
            };
            // The victory surge: the beaten line collapses outright where it
            // was crushed (the inner disc annexes at once), while the ground
            // beyond falls under occupation instead of flipping — a
            // contestable fringe the loser can win back by counter-attacking,
            // and that the peace table will divide by who actually holds it.
            let inner: Vec<(i32, i32)> = loser_tiles
                .iter()
                .copied()
                .filter(|&(x, y)| (x - lx).abs().max((y - ly).abs()) <= 7)
                .collect();
            let fringe: Vec<(i32, i32)> = loser_tiles
                .iter()
                .copied()
                .filter(|&(x, y)| {
                    let cheb = (x - lx).abs().max((y - ly).abs());
                    cheb > 7 && cheb <= 14
                })
                .collect();
            let taken = inner.len() + fringe.len();
            if taken < 4 {
                continue;
            }
            self.annex_tiles(winner.as_str(), &inner);
            // A beaten army still covers its rout: the fringe is pressed but
            // not overwhelmed, so it shows as occupied ground that decays
            // unless the winner keeps fighting there.
            use rand::RngExt;
            for &(x, y) in fringe.iter() {
                let cheb = (x - lx).abs().max((y - ly).abs());
                let press = 12.0 * (1.0 - (cheb as f32 / 15.0) * 0.75);
                let start = (4.0 + press + self.occupation_rng.random::<f32>() * 3.0).min(30.0);
                let occupied = match self.occupation.get(&(x, y)) {
                    Some((occ, _)) if occ.as_str() == winner.as_str() => false,
                    Some((_, strength)) => *strength < 25.0,
                    None => true,
                };
                if occupied {
                    self.occupation.insert((x, y), (winner.as_str().into(), start));
                }
            }
            self.territory_revision = self.territory_revision.wrapping_add(1);
            let wname = self
                .lineage_names
                .get(&winner)
                .cloned()
                .unwrap_or_else(|| winner.clone());
            let lname = self
                .lineage_names
                .get(&loser)
                .cloned()
                .unwrap_or_else(|| loser.clone());
            let line = if inner.is_empty() {
                format!(
                    "\u{2694}\u{FE0F} The {} overran the frontier of {}.",
                    wname, lname
                )
            } else {
                format!(
                    "\u{2694}\u{FE0F} The {} conquered {} tiles of {} land.",
                    wname,
                    inner.len(),
                    lname
                )
            };
            push_event(&mut self.events, now, "milestone", "world", &line);
            self.headlines.push_back((now, line));
            while self.headlines.len() > 80 {
                self.headlines.pop_front();
            }
        }
    }
}

#[cfg(test)]
mod sovereignty_tests {
    use super::*;

    fn first_walkable(sim: &Simulation) -> (i32, i32) {
        for y in 0..crate::world::grid::HEIGHT as i32 {
            for x in 0..crate::world::grid::WIDTH as i32 {
                if !matches!(sim.grid.get(x, y), Tile::Water | Tile::Void) {
                    return (x, y);
                }
            }
        }
        panic!("no walkable tile on the map");
    }

    /// A completed building plants the flag: unclaimed ground under it joins
    /// the builder's territory, seeds the frontier, and bumps the revision so
    /// the wire view redraws. Foreign sovereign ground is left alone.
    #[test]
    fn a_building_plants_the_flag_where_it_stands() {
        let mut sim = Simulation::new(87);
        let (x, y) = first_walkable(&sim);
        let lid: crate::intern::Sym = "builder".into();
        let before = sim.territory_revision;
        sim.seed_territory_at(&lid, x, y);
        assert!(
            sim.territory.get(&lid).is_some_and(|s| s.contains(&(x, y))),
            "the tile under the building is now the builder's"
        );
        assert_eq!(sim.tile_owner.get(&(x, y)).map(|o| o.as_str()), Some("builder"));
        assert!(
            sim.territory_frontiers.get(&lid).is_some_and(|q| !q.is_empty()),
            "the frontier grows outward from the new structure"
        );
        assert!(sim.territory_revision != before, "the view redraws");
        // Already someone's ground — including the builder's own: no change.
        let owner_before = sim.tile_owner.get(&(x, y)).copied();
        sim.seed_territory_at(&lid, x, y);
        sim.seed_territory_at(&"other".into(), x, y);
        assert_eq!(sim.tile_owner.get(&(x, y)).copied(), owner_before);
    }

    /// Wander and work destinations respect the flag: unclaimed or own ground
    /// is open, a neighbour's claimed country is not.
    #[test]
    fn land_is_open_to_respects_the_flag() {
        let mut sim = Simulation::new(87);
        let (x, y) = first_walkable(&sim);
        let mine: crate::intern::Sym = "mine".into();
        let other: crate::intern::Sym = "other".into();
        assert!(sim.land_is_open_to(&mine, x, y), "unclaimed ground is open");
        sim.tile_owner.insert((x, y), mine);
        assert!(sim.land_is_open_to(&mine, x, y), "own ground is open");
        assert!(!sim.land_is_open_to(&other, x, y), "a neighbour's is not");
    }
}
