//! The ecological food-web engine: pwb's Lotka-Volterra cycle ported into
//! the THB simulation layer.
//!
//! The baseline game kept animal numbers afloat with flat drip-respawn
//! floors; the grass never limited the grazers and the predators never
//! over-hunted their own larder. This engine couples all three troves:
//!
//!   * the grass stock (sampled each pass) sets a carrying capacity for the
//!     wild grazers — past it they starve competitively (the weakest loses),
//!     and their grazing pressure bares the meadow tiles they stand on;
//!     the ash heals back into grass on fertile ground, closing the loop.
//!   * the predator stock is capped by the prey stock (one predator per
//!     `PREY_PER_PREDATOR` prey): an over-dense pack thins itself by hunger.
//!   * predator kills draw from a budget that shrinks with the prey stock
//!     — an over-hunted larder gets a conservation window instead of an
//!     absorbing extinction, and starving beasts that find no wild game
//!     dare the village pens (a beast raid — ecology pressing on the
//!     civilisations, pwb's "wolves at the door" channel).
//!   * civilised hunting slackens when the wilds thin (the villagers put
//!     the bow away rather than hunt the last deer), and rich empty
//!     grasslands invite migration waves that repopulate the wilds —
//!     wolves follow a prey boom a generation later.
//!
//! The whole state is transient (recomputed each pass, rebuilt from the
//! seed on load) and all randomness comes from a private stream derived
//! from the world seed, never the world RNG.

use super::*;

/// Cadence of the ecology pass, in ticks (pwb runs it every 20).
pub(super) const ECOLOGY_STEP: u64 = 20;
/// Ticks of a season; four seasons make a year (see sim/config.rs).
const SEASON_LENGTH: u64 = 3000;
/// Ticks between eco headlines (fifteen pwb years, in THB ticks).
const NOTE_COOLDOWN: u64 = 15 * 4 * SEASON_LENGTH;
/// Ticks between migration waves (twenty-five pwb years).
const MIGRATION_COOLDOWN: u64 = 25 * 4 * SEASON_LENGTH;

/// Grass tiles one wild grazer needs: the home range of a deer in an
/// 800x450 world (~70k grass tiles on a default seed → capacity ~175).
const GRASS_TILES_PER_GRAZER: f32 = 400.0;
/// Herbivore-to-capacity ratio past which the wilds count as overgrazed.
const OVERGRAZE_RATIO: f32 = 1.15;
/// Ratio below which the grass stands rich and ungrazed.
const OVERGRASS_RATIO: f32 = 0.25;
/// Prey animals one wild predator is worth in a healthy food web.
const PREY_PER_PREDATOR: f32 = 4.5;
/// Tiles sampled each pass to estimate the grass stock (the world is
/// 360k tiles; a 3k sample estimates it within ~2%).
const GRASS_SAMPLES: usize = 3000;
/// Starvation wounds dished out to over-capacity grazers / over-dense
/// predators per pass (the weakest of a three-way tournament falls).
const MAX_STARVE_WOUNDS: usize = 10;
const MAX_PREDATOR_WOUNDS: usize = 8;
/// Meadow tiles a hungry herd can bare to ash each pass, and how many
/// ash tiles heal back on fertile ground.
const MAX_BARE_TILES: usize = 2;
const MAX_REGROW_TILES: usize = 6;
/// Energy below which a starving wild predator dares the village pens.
pub(crate) const RAID_ENERGY: f32 = 0.25;
/// Beast raiders flagged per pass (each may take one beast from a pen).
const MAX_RAIDERS: usize = 3;
/// Tiles a desperate hunter scans for wild game before it turns to the pens.
const RAID_HUNT_RADIUS: f32 = 8.0;
/// People within this range make a spot too settled for a migration wave.
const MIGRATION_PEOPLE_RADIUS: f32 = 12.0;
const MIGRATION_PEOPLE_MAX: usize = 4;

/// The ecological phase the wilds are in — headlines, and the pressure the
/// phase puts back on the world (hunting slackens as the wilds thin).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum EcoPhase {
    #[default]
    Balanced,
    /// Vegetation rich, grazers few — migration waves invited.
    Overgrass,
    /// Grazers past the carrying capacity — hunger spreads, meadows bare.
    Overgraze,
    /// The wilds nearly empty — hunting slackens hard.
    Collapse,
}

impl EcoPhase {
    /// Multiplier on civilised hunting success in this phase: a village
    /// puts the bow away when the game is gone rather than hunt the last
    /// deer (pwb's auto-conservation channel).
    pub(crate) fn hunting_pressure(self) -> f32 {
        match self {
            EcoPhase::Balanced | EcoPhase::Overgrass => 1.0,
            EcoPhase::Overgraze => 0.7,
            EcoPhase::Collapse => 0.3,
        }
    }
}

/// Transient ecology state: rebuilt from the world seed on load, never saved.
/// The phase, cooldowns and kill budget are all derived or rolling values.
pub(crate) struct EcologyState {
    /// Which phase the wilds are in (headlines and pressure).
    pub(crate) phase: EcoPhase,
    /// Rolling predator kill allowance, reset each pass from the prey stock.
    pub(crate) kill_allowance: usize,
    tick_of_last_note: u64,
    tick_of_last_migration: u64,
    /// Private stream: seeded from the world seed, never the world RNG.
    rng: ChaCha8Rng,
}

impl EcologyState {
    pub fn build(seed: u64) -> Self {
        EcologyState {
            phase: EcoPhase::Balanced,
            kill_allowance: 8,
            tick_of_last_note: 0,
            tick_of_last_migration: 0,
            rng: ChaCha8Rng::seed_from_u64(seed ^ 0x1c93_aa7d),
        }
    }

    /// Lend the private stream out for a pass. The caller returns it via
    /// `restore_rng` on every path (only the pass draws from it, so the
    /// world RNG sequence is untouched).
    fn take_rng(&mut self) -> ChaCha8Rng {
        std::mem::replace(&mut self.rng, ChaCha8Rng::seed_from_u64(0))
    }
    fn restore_rng(&mut self, rng: ChaCha8Rng) {
        self.rng = rng;
    }
}

impl Simulation {
    /// The ecology pass: every `ECOLOGY_STEP` ticks (offset 7, like pwb's
    /// predator hunt cadence) the wilds take stock, starve, bare, heal,
    /// migrate and make headlines. Called from `tick` before the animal
    /// pass of the same tick so the raid flags and the kill budget are
    /// fresh for the animals that move and hunt this tick.
    pub(super) fn tick_ecology(&mut self) {
        let tick = self.tick_count;
        let mut rng = self.ecology.take_rng();

        // ---- 0. Reset the beast-raid flags; the census re-arms them. ----
        for a in &mut self.animals {
            a.raiding = false;
        }

        // ---- 1. Census: wild grazers, wild fish and wild predators. ----
        // Livestock in pens and bonded pets are the civilisations' beasts,
        // not the wilds'; sleeping bears and birds away for the winter do
        // not graze this pass.
        let mut grazers: Vec<usize> = Vec::new();
        let mut predators: Vec<usize> = Vec::new();
        let mut fish_alive: usize = 0;
        let mut herb_alive: usize = 0;
        for (ai, a) in self.animals.iter().enumerate() {
            if !a.alive || a.is_kept() || a.bonded_org.is_some() || a.away || a.sleeping {
                continue;
            }
            if a.kind.aquatic() {
                if a.kind.is_prey() {
                    fish_alive += 1;
                }
            } else if a.kind.is_prey() {
                herb_alive += 1;
                grazers.push(ai);
            } else if a.kind.predator() {
                predators.push(ai);
            }
        }

        // ---- 2. The grass stock and the carrying capacity it sets. ----
        let mut grass_hits = 0usize;
        for _ in 0..GRASS_SAMPLES {
            let i = rng.random_range(0..(WIDTH * HEIGHT));
            if matches!(Tile::from_i8(self.grid.tiles[i]), Tile::Grass | Tile::Food) {
                grass_hits += 1;
            }
        }
        let grass_est = (grass_hits as f32 / GRASS_SAMPLES as f32) * (WIDTH * HEIGHT) as f32;
        let capacity = (grass_est / GRASS_TILES_PER_GRAZER).max(1.0);
        let ratio = herb_alive as f32 / capacity;
        let prey_alive = herb_alive + fish_alive;

        // ---- 3. Competitive starvation: grazers past the capacity ----
        // compete for the grass, and the weakest of the herd goes hungry
        // first (a three-way tournament, pwb's natural selection roll).
        if !grazers.is_empty() && herb_alive as f32 > capacity {
            let deficit = herb_alive as f32 - capacity;
            let wounds = MAX_STARVE_WOUNDS.min(((deficit * 0.2).ceil() as usize).max(1));
            for _ in 0..wounds {
                let mut victim: Option<usize> = None;
                let mut worst = f32::INFINITY;
                for _ in 0..3 {
                    let cand = grazers[rng.random_range(0..grazers.len())];
                    let e = self.animals[cand].energy;
                    if e < worst {
                        worst = e;
                        victim = Some(cand);
                    }
                }
                if let Some(vi) = victim {
                    if self.animals[vi].alive {
                        self.animals[vi].energy = (self.animals[vi].energy - 0.12).max(0.0);
                    }
                }
            }
        }

        // ---- 4. Predator self-thinning: one predator per 4.5 prey. ----
        // An over-dense pack starves on its own numbers.
        let predator_cap = herb_alive as f32 / PREY_PER_PREDATOR;
        if !predators.is_empty() && predators.len() as f32 > predator_cap && predators.len() > 4 {
            let deficit = predators.len() as f32 - predator_cap;
            let wounds = MAX_PREDATOR_WOUNDS.min(((deficit * 0.25).ceil() as usize).max(1));
            for _ in 0..wounds {
                let pi = predators[rng.random_range(0..predators.len())];
                self.animals[pi].energy = (self.animals[pi].energy - 0.10).max(0.0);
            }
        }

        // ---- 5. Grazing pressure and the healing of the meadows. ----
        // A herd past 70% of the capacity bares the tiles it stands on to
        // ash; ash on fertile ground heals back to grass, so the meadow
        // and the herd chase each other's numbers.
        if !grazers.is_empty() && ratio > 0.7 {
            for _ in 0..MAX_BARE_TILES {
                let gi = grazers[rng.random_range(0..grazers.len())];
                let (x, y) = (self.animals[gi].x as i32, self.animals[gi].y as i32);
                if WorldGrid::in_bounds(x, y) && self.grid.get(x, y) == Tile::Grass {
                    self.grid.set(x, y, Tile::Ash);
                }
            }
        }
        for _ in 0..MAX_REGROW_TILES {
            let i = rng.random_range(0..(WIDTH * HEIGHT));
            if Tile::from_i8(self.grid.tiles[i]) != Tile::Ash {
                continue;
            }
            let x = (i % WIDTH) as i32;
            let y = (i / WIDTH) as i32;
            if self.grid.fertility[i] < 0.25 {
                continue;
            }
            let grass_adj = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)].iter().any(|&(dx, dy)| {
                WorldGrid::in_bounds(x + dx, y + dy)
                    && matches!(
                        Tile::from_i8(self.grid.tiles[WorldGrid::idx(x + dx, y + dy)]),
                        Tile::Grass | Tile::Food
                    )
            });
            if grass_adj {
                self.grid.set(x, y, Tile::Grass);
            }
        }

        // ---- 6. The kill budget: prey stock → predator allowance. ----
        // A well-stocked larder funds eight wild kills a pass; a thinning
        // one halves it; a collapsing one shuts the hunt (the conservation
        // window that keeps an over-hunted food web from hitting zero).
        let mut allowance = 8usize;
        if prey_alive <= 150 {
            allowance = 4;
        }
        if prey_alive <= 60 {
            allowance = 2;
        }
        if prey_alive <= 25 {
            allowance = 0;
        }
        self.ecology.kill_allowance = allowance;

        // ---- 7. Beast raids: desperate hunters dare the pens. ----
        // A starving wild predator (deeper than the ordinary hungry band)
        // with no wild game within reach turns to the villages' pens as
        // its last hunting ground — the rails keep out a fed wolf, not a
        // starving one. The raid flags are read by the animal movement and
        // kill passes for the whole window between ecology passes.
        let mut raiders = 0usize;
        for &ai in &predators {
            if raiders >= MAX_RAIDERS {
                break;
            }
            let a = &self.animals[ai];
            if a.energy > RAID_ENERGY || a.summoned {
                continue;
            }
            let wild_game_near = self.animals.iter().any(|p| {
                p.alive
                    && !p.is_kept()
                    && p.bonded_org.is_none()
                    && p.kind.is_prey()
                    && (p.x - a.x).abs() + (p.y - a.y).abs() <= RAID_HUNT_RADIUS
            });
            if !wild_game_near {
                self.animals[ai].raiding = true;
                raiders += 1;
            }
        }

        // ---- 8. The phase and its headlines. ----
        let next_phase = if herb_alive < 8 {
            EcoPhase::Collapse
        } else if ratio > OVERGRAZE_RATIO {
            EcoPhase::Overgraze
        } else if ratio < OVERGRASS_RATIO {
            EcoPhase::Overgrass
        } else {
            EcoPhase::Balanced
        };
        if next_phase != self.ecology.phase {
            // The first transition is always newsworthy (a fresh world
            // announces its state once); afterwards the fifteen-year
            // cooldown keeps the headlines from flooding.
            let cooled = self.ecology.tick_of_last_note == 0
                || tick.saturating_sub(self.ecology.tick_of_last_note) > NOTE_COOLDOWN;
            if cooled {
                self.ecology.tick_of_last_note = tick;
                let (actor, detail): (&str, String) = match next_phase {
                    EcoPhase::Overgrass => (
                        "the wilds",
                        "grass stands rich and ungrazed — the wilds await new herds".to_string(),
                    ),
                    EcoPhase::Balanced => (
                        "the wilds",
                        "grazers and predators fall back into balance".to_string(),
                    ),
                    EcoPhase::Overgraze => (
                        "the wilds",
                        format!(
                            "the herds exceed the meadows' carrying capacity ({}/{}) — hunger spreads",
                            herb_alive,
                            capacity.round() as u32
                        ),
                    ),
                    EcoPhase::Collapse => (
                        "the wilds",
                        "the wild animals are all but gone — the wilds near collapse".to_string(),
                    ),
                };
                push_event(&mut self.events, tick, "eco", actor, &detail);
            }
            self.ecology.phase = next_phase;
        }

        // ---- 9. Migration waves: nature reseeds itself. ----
        // Rich empty grasslands invite herbivore herds from off the map;
        // a prey boom with no predators invites the wolves that follow it
        // (both at pwb's generational cooldown, and away from towns).
        if tick.saturating_sub(self.ecology.tick_of_last_migration) > MIGRATION_COOLDOWN {
            let herds_wanted = ratio < 0.22 && grass_est > 6_000.0 && rng.random::<f32>() < 0.35;
            let wolves_wanted = herb_alive > 100 && predators.len() < 3;
            if herds_wanted || wolves_wanted {
                self.ecology.tick_of_last_migration = tick;
                if herds_wanted {
                    self.ecology_migration_wave(&mut rng, 6, true);
                    push_event(
                        &mut self.events,
                        tick,
                        "eco",
                        "the wilds",
                        "herds migrate in to the rich untouched grass",
                    );
                } else {
                    self.ecology_migration_wave(&mut rng, 3, false);
                    push_event(
                        &mut self.events,
                        tick,
                        "eco",
                        "the wilds",
                        "wolves follow the herds in from the far wilds",
                    );
                }
            }
        }

        self.ecology.restore_rng(rng);
    }

    /// One migration wave: `count` beasts settle at a rich grass spot away
    /// from settled ground. Grazers arrive as a mixed herd (rabbit, deer,
    /// sheep and the like); a wolf wave settles near the biggest herd it
    /// can find. Placement draws from the private ecology stream only.
    fn ecology_migration_wave(&mut self, rng: &mut ChaCha8Rng, count: usize, grazers: bool) {
        let kinds: &[AnimalKind] = if grazers {
            &[
                AnimalKind::Rabbit,
                AnimalKind::Deer,
                AnimalKind::Sheep,
                AnimalKind::Horse,
                AnimalKind::Boar,
            ]
        } else {
            &[AnimalKind::Wolf, AnimalKind::Wolf, AnimalKind::Fox]
        };
        // A rich spot: mostly grass around it, and few people settled near.
        let mut spot: Option<(i32, i32)> = None;
        for _ in 0..80 {
            let x = rng.random_range(4..(WIDTH as i32 - 4));
            let y = rng.random_range(4..(HEIGHT as i32 - 4));
            if self.grid.get(x, y) != Tile::Grass {
                continue;
            }
            let grass_around = [(-2i32, 0i32), (2, 0), (0, -2), (0, 2), (1, 1), (-1, -1)]
                .iter()
                .filter(|&&(dx, dy)| {
                    WorldGrid::in_bounds(x + dx, y + dy)
                        && matches!(
                            Tile::from_i8(self.grid.tiles[WorldGrid::idx(x + dx, y + dy)]),
                            Tile::Grass | Tile::Food
                        )
                })
                .count();
            if grass_around < 5 {
                continue;
            }
            let people_near = self
                .organisms
                .iter()
                .filter(|o| {
                    o.alive && (o.x - x as f32).abs() + (o.y - y as f32).abs() <= MIGRATION_PEOPLE_RADIUS
                })
                .count();
            if people_near > MIGRATION_PEOPLE_MAX {
                continue;
            }
            spot = Some((x, y));
            break;
        }
        let Some((sx, sy)) = spot else {
            return;
        };
        for _ in 0..count {
            let kind = kinds[rng.random_range(0..kinds.len())];
            let dx = rng.random_range(-3i32..=3);
            let dy = rng.random_range(-3i32..=3);
            let (x, y) = (sx + dx, sy + dy);
            if !WorldGrid::in_bounds(x, y) || !kind.fits_ground(self.grid.get(x, y)) {
                continue;
            }
            let id = self.next_animal_id;
            self.next_animal_id += 1;
            self.animals
                .push(Animal::new(id, x as f32 + 0.5, y as f32 + 0.5, kind));
        }
    }
}
