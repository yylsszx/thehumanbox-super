//! The ecological food web: grass limits the grazers, the predators limit
//! themselves, a starving hunter dares the pens, and rich empty grasslands
//! reseed themselves from off the map.

use super::*;
use crate::hashing::FxHashMap;

fn empty_org_index() -> FxHashMap<crate::intern::Sym, usize> {
    FxHashMap::default()
}

fn push_animal(sim: &mut Simulation, kind: AnimalKind, x: f32, y: f32) -> usize {
    let id = sim.next_animal_id;
    sim.next_animal_id += 1;
    let mut a = Animal::new(id, x, y, kind);
    if kind.predator() {
        a.energy = 0.1;
    }
    sim.animals.push(a);
    sim.animals.len() - 1
}

/// Herds past the meadows' carrying capacity starve competitively, bare the
/// grass they stand on, and the wilds announce the overgrazing.
#[test]
fn over_capacity_herds_starve_and_bare_the_meadow() {
    let mut sim = Simulation::new(0xEC0);
    sim.organisms.clear();
    sim.animals.clear();
    for _ in 0..300 {
        sim.spawn_animal_of_kind(AnimalKind::Rabbit);
    }
    let energy_before: f32 = sim.animals.iter().map(|a| a.energy).sum();
    let grass_before = sim
        .grid
        .tiles
        .iter()
        .filter(|&&t| matches!(Tile::from_i8(t), Tile::Grass | Tile::Food))
        .count();
    // Three ecology passes: the weakest rabbits starve, the herd bares
    // meadow tiles to ash, and the phase turns to overgraze.
    for tick in [7u64, 27, 47] {
        sim.tick_count = tick;
        sim.tick_ecology();
    }
    let energy_after: f32 = sim.animals.iter().map(|a| a.energy).sum();
    assert!(
        energy_after < energy_before,
        "an over-capacity herd starves ({} -> {})",
        energy_before,
        energy_after
    );
    assert_eq!(
        sim.ecology.phase,
        crate::sim::simulation::ecology::EcoPhase::Overgraze,
        "the wilds count the herds as overgrazing"
    );
    let grass_after = sim
        .grid
        .tiles
        .iter()
        .filter(|&&t| matches!(Tile::from_i8(t), Tile::Grass | Tile::Food))
        .count();
    assert!(
        grass_after < grass_before,
        "grazing pressure bares meadow tiles ({} -> {})",
        grass_before,
        grass_after
    );
    assert!(
        sim.events.iter().any(|e| e.etype == "eco"),
        "the overgrazing makes the news"
    );
}

/// A collapsing larder shuts the predators' hunt: with the prey stock under
/// the conservation threshold no wild kill happens even inches away.
#[test]
fn a_collapsing_larder_shuts_the_predators_hunt() {
    let mut sim = Simulation::new(0xEC1);
    sim.organisms.clear();
    sim.animals.clear();
    // Four rabbits and three hungry wolves right beside them: well under
    // the 25-prey threshold, so the kill budget for the window is zero.
    let mut rabbits = Vec::new();
    for i in 0..4 {
        rabbits.push(push_animal(&mut sim, AnimalKind::Rabbit, 100.0 + i as f32, 100.0));
    }
    for i in 0..3 {
        push_animal(&mut sim, AnimalKind::Wolf, 101.0, 101.0 + i as f32);
    }
    for tick in (7u64..200).step_by(20) {
        sim.tick_count = tick;
        sim.tick_ecology();
        sim.tick_animals(&empty_org_index());
    }
    let rabbits_alive = rabbits.iter().filter(|&&ri| sim.animals[ri].alive).count();
    assert_eq!(
        rabbits_alive, 4,
        "the conservation window holds: hungry wolves beside rabbits may not kill"
    );
}

/// A starving predator with no wild game in reach turns to the village pens
/// — the rails keep out a fed wolf, not a desperate one — and the raid
/// makes the news.
#[test]
fn a_starving_wolf_with_no_wild_game_dares_the_pens() {
    let mut sim = Simulation::new(0xEC2);
    sim.organisms.clear();
    sim.animals.clear();
    // A sheep kept in a pen, and no wild prey anywhere near a starving wolf.
    let sheep = push_animal(&mut sim, AnimalKind::Sheep, 100.0, 100.0);
    sim.animals[sheep].keeper = Some("lin-a".to_string());
    sim.animals[sheep].pen = Some((100, 100));
    let sheep_id = sim.animals[sheep].id;
    let wolf = push_animal(&mut sim, AnimalKind::Wolf, 101.0, 100.0);

    sim.tick_count = 7;
    sim.tick_ecology();
    assert!(
        sim.animals[wolf].raiding,
        "a starving wolf with no wild game in reach turns to the pens"
    );
    sim.tick_animals(&empty_org_index());
    // The taken sheep leaves the animals vec, so look it up by id.
    let sheep_still_alive = sim.animals.iter().any(|a| a.id == sheep_id && a.alive);
    assert!(!sheep_still_alive, "the raider carries off the penned sheep");
    assert!(
        sim.events
            .iter()
            .any(|e| e.etype == "eco" && e.detail.contains("broke into a pen")),
        "the beast raid makes the news"
    );
}

/// Rich empty grasslands invite migration waves from off the map: nature
/// reseeds a defaunated world by itself.
#[test]
fn rich_empty_grass_invites_a_herd_from_off_map() {
    let mut sim = Simulation::new(0xEC3);
    sim.organisms.clear();
    sim.animals.clear();
    let mut herds_arrived = false;
    // The migration gate opens after the 25-year cooldown (300k ticks; the
    // pass draws past it, as a real long-running world would have); the roll retries
    // each pass until the wave lands — the private stream is deterministic,
    // and the pass cadence keeps clear of the drip-respawn floors (which
    // only fire from `tick_animals`).
    for tick in (300_007u64..300_840).step_by(20) {
        sim.tick_count = tick;
        sim.tick_ecology();
        if sim.animals.iter().filter(|a| a.alive).count() > 0 {
            herds_arrived = true;
            break;
        }
    }
    assert!(herds_arrived, "herds migrate in to the rich untouched grass");
    assert!(
        sim.events
            .iter()
            .any(|e| e.etype == "eco" && e.detail.contains("migrate")),
        "the migration makes the news"
    );
}

/// Civilised hunting slackens as the wilds thin — the multiplier the catch
/// rolls run through.
#[test]
fn hunting_pressure_slackens_as_the_wilds_thin() {
    use crate::sim::simulation::ecology::EcoPhase;
    assert_eq!(EcoPhase::Balanced.hunting_pressure(), 1.0);
    assert_eq!(EcoPhase::Overgrass.hunting_pressure(), 1.0);
    assert_eq!(EcoPhase::Overgraze.hunting_pressure(), 0.7);
    assert_eq!(EcoPhase::Collapse.hunting_pressure(), 0.3);
}

/// The ecology engine runs deterministically under the full tick: two
/// worlds from the same seed, ecology pass and all, walk the same line.
#[test]
fn the_ecology_engine_runs_deterministically_under_the_full_tick() {
    fn fingerprint(seed: u64, ticks: u64) -> (u64, u64, u64, usize) {
        let mut sim = Simulation::new(seed);
        for _ in 0..ticks {
            sim.tick();
        }
        let mut animals = 0u64;
        let mut energy = 0u64;
        for a in &sim.animals {
            if a.alive {
                animals = animals.wrapping_mul(31).wrapping_add(a.id as u64);
                energy = energy.wrapping_add((a.energy * 4096.0) as u64);
            }
        }
        let mut tiles = 0u64;
        for (i, &t) in sim.grid.tiles.iter().enumerate() {
            tiles = tiles
                .wrapping_mul(31)
                .wrapping_add(t as i64 as u64)
                .wrapping_add(i as u64 % 7);
        }
        (animals, energy, tiles, sim.events.len())
    }
    // Enough ticks to run the ecology pass many times, cross a season
    // boundary and let the food web bite — but stay under the migration
    // cooldown's horizon so the comparison stays about the pass itself.
    assert_eq!(fingerprint(0xEC4, 2_500), fingerprint(0xEC4, 2_500));
}
