//! The per-tick lineage tallies borrow a lineage's id until they need to keep
//! it. These check them against the versions that copied it for every person.

use super::*;
use crate::sim::agents::growth;

fn lived_in(seed: u64, ticks: u32) -> Simulation {
    let mut sim = Simulation::new(seed);
    sim.tick_n(ticks);
    sim
}

#[test]
fn member_index_matches_the_cloning_version() {
    for seed in [7u64, 42] {
        let sim = lived_in(seed, 1_500);
        let fast = lineage_member_index(&sim.organisms);
        let mut original: FxHashMap<crate::intern::Sym, Vec<usize>> = FxHashMap::default();
        for (idx, org) in sim.organisms.iter().enumerate() {
            if org.alive {
                original.entry(org.lineage_id).or_default().push(idx);
            }
        }
        // The same keys in the same table order, each with the same members.
        let flatten = |map: &FxHashMap<crate::intern::Sym, Vec<usize>>| {
            map.iter().map(|(k, v)| (*k, v.clone())).collect::<Vec<_>>()
        };
        assert_eq!(flatten(&fast), flatten(&original), "seed {seed}");
        assert!(fast.len() > 3);
    }
}

#[test]
fn population_slots_match_the_cloning_version() {
    for seed in [7u64, 42] {
        let mut sim = lived_in(seed, 1_500);
        // Some unborn children, so the pending-birth arm is exercised.
        for (i, org) in sim.organisms.iter_mut().enumerate() {
            if !org.alive && i % 4 == 0 {
                org.age = 0;
                org.parent_id = format!("parent-{i}");
                org.father_id = Some("father".to_string());
            }
        }
        let fast = growth::lineage_population_slots(&sim.organisms);
        let mut original: FxHashMap<crate::intern::Sym, usize> = FxHashMap::default();
        for organism in sim
            .organisms
            .iter()
            .filter(|organism| organism.alive || growth::is_pending_birth(organism))
        {
            *original.entry(organism.lineage_id).or_insert(0) += 1;
        }
        let flatten = |map: &FxHashMap<crate::intern::Sym, usize>| {
            map.iter().map(|(k, v)| (k.to_string(), *v)).collect::<Vec<_>>()
        };
        assert_eq!(flatten(&fast), flatten(&original), "seed {seed}");
        assert!(fast.values().sum::<usize>() > 50);
    }
}

#[test]
fn lineage_aggregates_match_the_cloning_version() {
    for seed in [7u64, 42] {
        let mut sim = lived_in(seed, 1_500);
        sim.rebuild_lineage_aggregates();
        let mut original: HashMap<crate::intern::Sym, LineageAggregate> = HashMap::default();
        for org in sim.organisms.iter().filter(|org| org.alive) {
            let entry = original.entry(org.lineage_id).or_default();
            entry.population += 1;
            entry.x_sum += org.x;
            entry.y_sum += org.y;
            entry.literacy_sum += org.literacy;
            entry.energy_sum += org.energy;
        }
        // The live map keeps its table from earlier ticks, so compare by key:
        // the tally is only ever looked up, never walked.
        let flatten = |map: &HashMap<crate::intern::Sym, LineageAggregate>| {
            let mut rows = map
                .iter()
                .map(|(k, a)| {
                    (
                        k.to_string(),
                        a.population,
                        a.x_sum.to_bits(),
                        a.y_sum.to_bits(),
                        a.literacy_sum.to_bits(),
                        a.energy_sum.to_bits(),
                    )
                })
                .collect::<Vec<_>>();
            rows.sort();
            rows
        };
        assert_eq!(
            flatten(&sim.lineage_aggregates),
            flatten(&original),
            "seed {seed}"
        );
        assert!(original.len() > 3);
        // Rebuilding on top of last tick's tally starts from nothing.
        sim.rebuild_lineage_aggregates();
        assert_eq!(
            flatten(&sim.lineage_aggregates),
            flatten(&original),
            "seed {seed} twice"
        );
    }
}

/// Territory as the maps hold it: lineages, their tiles and the tile owners,
/// each sorted. A map's iteration order is random per map (every `HashMap`
/// has its own hasher state), so two identical maps can yield their entries
/// in different orders; the claims must match as sets, not as orders.
type TerritoryState = (Vec<(String, Vec<(i32, i32)>)>, Vec<((i32, i32), String)>);

fn territory_state(sim: &Simulation) -> TerritoryState {
    let mut lineages: Vec<(String, Vec<(i32, i32)>)> = sim
        .territory
        .iter()
        .map(|(lineage, tiles)| {
            let mut sorted: Vec<(i32, i32)> = tiles.iter().copied().collect();
            sorted.sort_unstable();
            (lineage.to_string(), sorted)
        })
        .collect();
    lineages.sort();
    let mut owners: Vec<((i32, i32), String)> = sim
        .tile_owner
        .iter()
        .map(|(&tile, owner)| (tile, owner.to_string()))
        .collect();
    owners.sort();
    (lineages, owners)
}

#[test]
fn territory_claims_match_the_sorting_version() {
    for seed in [7u64, 42] {
        let mut fast = lived_in(seed, 300);
        let mut original = lived_in(seed, 300);
        for y in 94..=146 {
            for x in 144..=206 {
                fast.grid.set(x, y, crate::world::tiles::Tile::Grass);
                original.grid.set(x, y, crate::world::tiles::Tile::Grass);
            }
        }
        let lineages: Vec<String> = {
            let mut ids: Vec<String> = fast.organisms.iter().map(|o| o.lineage_id.to_string()).collect();
            ids.sort();
            ids.dedup();
            ids
        };
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for step in 0..1_500 {
            // Three lineages share a patch of land, so their territories
            // overlap and fill up, with the odd claim off the edge of the world.
            let lineage = &lineages[(next() % 3.min(lineages.len() as u64)) as usize];
            let cx = 150 + (next() % 50) as i32 + if step % 97 == 0 { -190 } else { 0 };
            let cy = 100 + (next() % 40) as i32;
            let radius = [1, 3, 4, 6, 6][(next() % 5) as usize];
            fast.claim_territory(lineage, cx, cy, radius);
            original.claim_territory_reference(lineage, cx, cy, radius);
            assert_eq!(
                territory_state(&fast),
                territory_state(&original),
                "seed {seed} step {step}"
            );
        }
        // Claims are cap-free now (the growth pass owns territory size), so a
        // heavily claimed lineage holds more than the old 400-tile cap.
        let largest = fast.territory.values().map(|t| t.len()).max().unwrap_or(0);
        assert!(
            largest > 400,
            "a lineage should hold more than the old cap: {largest}"
        );
    }
}

/// The lineage with the most living people, as (id, population).
fn biggest_lineage(sim: &Simulation) -> (String, u32) {
    let mut counts: HashMap<String, u32> = HashMap::default();
    for org in sim.organisms.iter().filter(|o| o.alive) {
        *counts.entry(org.lineage_id.to_string()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(id, count)| (*count, id.clone()))
        .expect("a world with people")
}

#[test]
fn organic_growth_creeps_borders_and_respects_the_cap() {
    let mut sim = lived_in(11, 900);
    let (lid, pop) = biggest_lineage(&sim);
    // The people must know where the tribe ends for the cap to scale with
    // them; hand them the borders discovery.
    for org in sim.organisms.iter_mut() {
        if org.alive && org.lineage_id.as_str() == lid {
            org.discoveries.insert("Territory".to_string());
        }
    }
    // Seed: the founding camp's small patch. Clear whatever the lineage
    // gathered during the 900 ticks first, so the seed is the true start.
    let org = sim
        .organisms
        .iter()
        .find(|o| o.alive && o.lineage_id.as_str() == lid)
        .expect("a living member");
    let (hx, hy) = (org.home_x as i32, org.home_y as i32);
    sim.territory.shift_remove(lid.as_str());
    sim.tile_owner.retain(|_, owner| owner.as_str() != lid);
    sim.claim_territory(&lid, hx, hy, 1);
    let start = sim.territory.get(lid.as_str()).map_or(0, |t| t.len());
    assert!(start <= 5, "the seed patch is small: {start}");
    let cap = (24 + pop as usize * 6).min(1600);

    // Creep for many passes: borders grow along the frontier.
    for _ in 0..600 {
        sim.tick_territory_growth();
    }
    let size = sim.territory.get(lid.as_str()).map_or(0, |t| t.len());
    assert!(
        size <= cap,
        "territory stays within what its people hold: {size} > {cap}"
    );
    assert!(size > start + 10, "borders crept outward: {start} -> {size}");

    // Organic, not stamped: the whole territory is 4-connected to the seed
    // patch, because every grown tile was claimed next to owned land.
    let initial: std::collections::HashSet<(i32, i32)> = sim
        .territory
        .get(lid.as_str())
        .map(|t| t.iter().copied().collect())
        .unwrap_or_default();
    for _ in 0..600 {
        sim.tick_territory_growth();
    }
    let tiles: std::collections::HashSet<(i32, i32)> = sim
        .territory
        .get(lid.as_str())
        .map(|t| t.iter().copied().collect())
        .unwrap_or_default();
    let mut seen: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<(i32, i32)> = std::collections::VecDeque::new();
    queue.push_back((hx, hy));
    seen.insert((hx, hy));
    while let Some((x, y)) = queue.pop_front() {
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let next = (x + d.0, y + d.1);
            if tiles.contains(&next) && !seen.contains(&next) {
                seen.insert(next);
                queue.push_back(next);
            }
        }
    }
    assert_eq!(
        seen.len(),
        tiles.len(),
        "every claimed tile is reachable from the seed"
    );

    // Growth never steals another tribe's sovereign land: every NEW tile was
    // unowned when it was claimed (the seed disc may legitimately overlap).
    let stolen: Vec<(i32, i32)> = tiles
        .iter()
        .copied()
        .filter(|t| !initial.contains(t))
        .filter(|t| sim.tile_owner.get(t).is_some_and(|owner| owner.as_str() != lid))
        .collect();
    assert!(stolen.is_empty(), "growth claimed owned land: {stolen:?}");
}

#[test]
fn dead_tribes_borders_dissolve() {
    let mut sim = lived_in(23, 900);
    let (lid, _) = biggest_lineage(&sim);
    let org = sim
        .organisms
        .iter()
        .find(|o| o.alive && o.lineage_id.as_str() == lid)
        .expect("a living member");
    let (hx, hy) = (org.home_x as i32, org.home_y as i32);
    sim.claim_territory(&lid, hx, hy, 4);
    let size = sim.territory.get(lid.as_str()).map_or(0, |t| t.len());
    assert!(size > 30, "a claim to dissolve: {size}");
    // The tribe dies out; nobody may hold its land.
    for org in sim.organisms.iter_mut() {
        if org.lineage_id.as_str() == lid {
            org.alive = false;
        }
    }
    for _ in 0..(size / 8 + 4) {
        sim.tick_territory_growth();
    }
    let left = sim.territory.get(lid.as_str()).map_or(0, |t| t.len());
    assert_eq!(left, 0, "a dead tribe's borders dissolved");
    let lingering = sim
        .tile_owner
        .iter()
        .filter(|(_, owner)| owner.as_str() == lid)
        .count();
    assert_eq!(lingering, 0, "the inverse map dropped the dissolved tiles too");
}
