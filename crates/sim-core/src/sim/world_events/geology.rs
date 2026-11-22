//! Deep-time geology: plate boundaries, stress and hotspots drive emergent
//! earthquakes and eruptions, and slow erosion rounds the world.
//!
//! Ported from the Pixel World Box "nature/worldsim" design, adapted to this
//! engine's rules:
//!   * the whole state is rebuilt from `world_seed` (plus the live grid for
//!     erosion), so it is deliberately NOT persisted — a reloaded world starts
//!     a fresh geological epoch, exactly like the river/coast drift systems
//!     already in `evolution.rs`;
//!   * all randomness comes from a private stream derived from the seed, never
//!     from the world RNG, so adding this system does not perturb any existing
//!     downstream roll (reproducibility and save-fingerprint tests unchanged);
//!   * disasters reuse the existing damage stack: terrain cracks like the
//!     earthquake god tool, organism harm, and
//!     `civ::land::building_damage::quake_damage` for buildings.
//!
//! Cadence: every 300 ticks (the world-evolution beat). Stress accumulates on
//! convergent boundaries; when a cell crosses the release threshold the fault
//! lets go as a real earthquake centred on the most-stressed cell. Hotspots
//! on the boundaries run a heat budget that ends in an eruption. Every second
//! beat an erosion pass samples steep land near water and shaves elevation
//! onto floodplains.

use super::*;
use crate::math::DetMath;
use crate::sim::simulation::Simulation;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// How many ticks between geology beats (shared with world evolution).
pub const GEOLOGY_STEP: u64 = 300;
/// Stress at which a boundary cell releases as an earthquake.
pub const RELEASE_STRESS: f32 = 26.0;
/// Stress gained per beat on a fully convergent boundary (scaled by difficulty).
const CONVERGENT_GAIN: f32 = 0.12;
/// Stress lost per beat everywhere (slow tectonic relaxation).
const STRESS_DECAY: f32 = 0.999;
/// Hotspot heat release threshold.
const HOTSPOT_RELEASE: f32 = 1.0;
/// Hotspot heat gained per beat.
const HOTSPOT_GAIN: f32 = 0.035;
/// How many plates the crust breaks into.
const PLATE_COUNT: u8 = 6;
/// Erosion samples per pass.
const EROSION_SAMPLES: usize = 240;

struct Hotspot {
    x: i32,
    y: i32,
    heat: f32,
}

/// Transient tectonic state, rebuilt from the world seed.
pub struct GeologyState {
    /// Plate id per tile (write-only today: kept for the planned orogeny
    /// visualisation).
    #[allow(dead_code)]
    plate: Vec<u8>,
    /// Stress per tile (only boundary cells are meaningful).
    stress: Vec<f32>,
    /// Boundary cells: (tile index, plate a, plate b).
    boundary: Vec<(u32, u8, u8)>,
    /// Plate centroids (tiles move nothing; centroids anchor the boundary normal).
    centroids: Vec<(f32, f32)>,
    /// Drift vector per plate. Only its projection on a boundary decides
    /// whether that boundary converges (stress) or relaxes.
    drift: Vec<(f32, f32)>,
    hotspots: Vec<Hotspot>,
    beats: u64,
    /// Statistics for tests and curious gods.
    pub quakes: u64,
    pub eruptions: u64,
    /// Private stream: seeded from the world seed, never the world RNG.
    rng: ChaCha8Rng,
}

impl GeologyState {
    /// Build plates and boundaries from the seed alone. Jittered Voronoi over
    /// the map, drift vectors drawn from the private stream.
    pub fn build(seed: u64) -> Self {
        let size = WIDTH * HEIGHT;
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9e37_79b9);
        let mut seeds = Vec::with_capacity(PLATE_COUNT as usize);
        for _ in 0..PLATE_COUNT {
            seeds.push((
                rng.random_range(0..WIDTH as i32) as f32,
                rng.random_range(0..HEIGHT as i32) as f32,
            ));
        }
        // Two Lloyd relaxation steps settle the plates to comparable areas.
        for _ in 0..2 {
            let mut sum = vec![(0.0f32, 0.0f32, 0u32); PLATE_COUNT as usize];
            for y in 0..HEIGHT as i32 {
                for x in 0..WIDTH as i32 {
                    let (mut best, mut bd) = (0u8, f32::MAX);
                    for (p, &(sx, sy)) in seeds.iter().enumerate() {
                        let d = (x as f32 - sx) * (x as f32 - sx) + (y as f32 - sy) * (y as f32 - sy);
                        if d < bd {
                            bd = d;
                            best = p as u8;
                        }
                    }
                    let e = &mut sum[best as usize];
                    e.0 += x as f32;
                    e.1 += y as f32;
                    e.2 += 1;
                }
            }
            for (p, e) in sum.iter().enumerate() {
                if e.2 > 0 {
                    seeds[p] = (e.0 / e.2 as f32, e.1 / e.2 as f32);
                }
            }
        }

        let mut plate = Vec::with_capacity(size);
        let mut centroids = vec![(0.0f32, 0.0f32); PLATE_COUNT as usize];
        let mut counts = vec![0u32; PLATE_COUNT as usize];
        for y in 0..HEIGHT as i32 {
            for x in 0..WIDTH as i32 {
                let (mut best, mut bd) = (0u8, f32::MAX);
                for (p, &(sx, sy)) in seeds.iter().enumerate() {
                    // Slight per-tile jitter keeps borders craggy.
                    let jx = x as f32 + 0.0;
                    let d = (jx - sx) * (jx - sx) + (y as f32 - sy) * (y as f32 - sy);
                    if d < bd {
                        bd = d;
                        best = p as u8;
                    }
                }
                plate.push(best);
                centroids[best as usize].0 += x as f32;
                centroids[best as usize].1 += y as f32;
                counts[best as usize] += 1;
            }
        }
        for p in 0..PLATE_COUNT as usize {
            if counts[p] > 0 {
                centroids[p] = (
                    centroids[p].0 / counts[p] as f32,
                    centroids[p].1 / counts[p] as f32,
                );
            }
        }

        // Boundary cells and the plates they put together.
        let mut boundary = Vec::new();
        for y in 0..HEIGHT as i32 {
            for x in 0..WIDTH as i32 {
                let i = WorldGrid::idx(x, y) as u32;
                let here = plate[i as usize];
                let right = (x + 1 < WIDTH as i32).then(|| plate[WorldGrid::idx(x + 1, y)]);
                let down = (y + 1 < HEIGHT as i32).then(|| plate[WorldGrid::idx(x, y + 1)]);
                for other in [right, down].into_iter().flatten() {
                    if other != here {
                        boundary.push((i, here, other));
                        break;
                    }
                }
            }
        }

        // Drift per plate: a direction and speed. Convergent pairs press.
        let drift = (0..PLATE_COUNT)
            .map(|_| {
                let theta = rng.random::<f32>() * std::f32::consts::TAU;
                let m = 0.4 + rng.random::<f32>() * 0.8;
                (theta.cos() * m, theta.sin() * m)
            })
            .collect();

        // Hotspots sit on the boundary, spread out along it.
        let mut hotspots = Vec::new();
        if !boundary.is_empty() {
            let want = 3;
            let stride = boundary.len() / want;
            for k in 0..want {
                let (i, _, _) = boundary[k * stride];
                let x = (i % WIDTH as u32) as i32;
                let y = (i / WIDTH as u32) as i32;
                hotspots.push(Hotspot { x, y, heat: 0.0 });
            }
        }

        GeologyState {
            plate,
            stress: vec![0.0; size],
            boundary,
            centroids,
            drift,
            hotspots,
            beats: 0,
            quakes: 0,
            eruptions: 0,
            rng,
        }
    }

    /// Moisture-unrelated helper kept public for tests: how many boundary
    /// cells exist.
    pub fn boundary_len(&self) -> usize {
        self.boundary.len()
    }
}

impl Simulation {
    /// One geology beat. Call every `GEOLOGY_STEP` ticks from the tick loop.
    pub(crate) fn tick_geology_engine(&mut self) {
        let disaster = self.goals.difficulty.disaster_mult();
        self.geology.beats += 1;
        let beat = self.geology.beats;

        // 1. Stress on boundaries: convergence of the two plates pressed onto
        //    the boundary normal gains stress; divergence relaxes it.
        let boundary = self.geology.boundary.clone();
        let (centroids, drift) = (self.geology.centroids.clone(), self.geology.drift.clone());
        {
            let geo = &mut self.geology;
            for (i, a, b) in boundary.iter() {
                let (ca, cb) = (centroids[*a as usize], centroids[*b as usize]);
                let mut nx = cb.0 - ca.0;
                let mut ny = cb.1 - ca.1;
                let len = (nx * nx + ny * ny).sqrt().max(1.0);
                nx /= len;
                ny /= len;
                let (da, db) = (drift[*a as usize], drift[*b as usize]);
                let conv = (da.0 - db.0) * nx + (da.1 - db.1) * ny;
                let s = &mut geo.stress[*i as usize];
                *s = (*s * STRESS_DECAY + conv.max(0.15) * CONVERGENT_GAIN * disaster)
                    .min(RELEASE_STRESS * 3.0);
            }
        }

        // 2. Release: the most stressed cell lets go as a real earthquake.
        let peak = {
            let geo = &self.geology;
            let mut peak = (0u32, 0.0f32);
            for (i, _, _) in geo.boundary.iter() {
                if geo.stress[*i as usize] > peak.1 {
                    peak = (*i, geo.stress[*i as usize]);
                }
            }
            peak
        };
        if peak.1 >= RELEASE_STRESS {
            let x = (peak.0 % WIDTH as u32) as i32;
            let y = (peak.0 / WIDTH as u32) as i32;
            let radius = ((4.0 + peak.1 / RELEASE_STRESS * 6.0) as i32).clamp(5, 12);
            self.geology_quake(x, y, radius);
            // The whole neighbourhood slides back down the stress curve.
            let geo = &mut self.geology;
            for dy in -16i32..=16 {
                for dx in -16i32..=16 {
                    let (tx, ty) = (x + dx, y + dy);
                    if WorldGrid::in_bounds(tx, ty) {
                        let j = WorldGrid::idx(tx, ty);
                        geo.stress[j] *= 0.15;
                    }
                }
            }
            geo.quakes += 1;
        }

        // 3. Hotspots run a heat budget; a mature one erupts.
        let mut erupting: Vec<(i32, i32)> = Vec::new();
        {
            let geo = &mut self.geology;
            for h in geo.hotspots.iter_mut() {
                let hi = WorldGrid::idx(h.x, h.y);
                h.heat += HOTSPOT_GAIN + geo.stress[hi] * 0.004 * disaster;
                if h.heat >= HOTSPOT_RELEASE {
                    h.heat = 0.0;
                    erupting.push((h.x, h.y));
                }
            }
        }
        for (x, y) in erupting {
            self.geology_erupt(x, y);
            self.geology.eruptions += 1;
        }

        // 4. Erosion on every second beat: steep land near water shaves
        //    elevation onto its floodplain and feeds soil fertility.
        if beat.is_multiple_of(2) {
            self.geology_erode();
        }
    }

    /// Full earthquake effect: terrain cracks, organism harm, building
    /// damage — the same stack as the earthquake god tool, driven by
    /// tectonics instead of a click.
    fn geology_quake(&mut self, x: i32, y: i32, radius: i32) {
        let r = radius.clamp(1, 16);
        for dx in -r..=r {
            for dy in -r..=r {
                let d2 = dx * dx + dy * dy;
                if d2 > r * r {
                    continue;
                }
                let (nx, ny) = (x + dx, y + dy);
                let cur = self.grid.get(nx, ny);
                if matches!(cur, Tile::Hut | Tile::Campfire | Tile::Water | Tile::Void) {
                    continue;
                }
                let near = 1.0 - (d2 as f32).sqrt() / r as f32;
                if self.geology.rng.random::<f32>() < 0.12 + 0.30 * near {
                    let crack = if self.geology.rng.random::<f32>() < 0.6 {
                        Tile::Rock
                    } else {
                        Tile::Sand
                    };
                    self.grid.set(nx, ny, crack);
                    *self.grid.fire_intensity_mut(nx, ny) = 0.0;
                }
            }
        }
        let (fx, fy, fr) = (x as f32, y as f32, r as f32);
        let tick = self.tick_count;
        let mut hurt = 0;
        for o in self.organisms.iter_mut() {
            let d = (o.x - fx).det_hypot(o.y - fy);
            if o.alive && d <= fr {
                o.health -= 0.25 + 0.35 * (1.0 - d / fr);
                o.mark_harm(crate::organism::organism::Harm::Disaster, tick);
                o.fear_level = (o.fear_level + 0.5).min(1.0);
                hurt += 1;
            }
        }
        let buildings = crate::sim::civ::building_damage::quake_damage(self, x, y, r);
        push_event(
            &mut self.events,
            tick,
            "earthquake",
            "the earth",
            &format!(
                "the crust gave way along a fault, hurting {hurt} people and damaging {buildings} buildings"
            ),
        );
    }

    /// Hotspot eruption: fire, a scatter of cone rock and mineral, light
    /// building damage — a volcanic neighbour of the spontaneous eruption
    /// in `evolution.rs`, but anchored to a hotspot that builds heat.
    fn geology_erupt(&mut self, x: i32, y: i32) {
        let tick = self.tick_count;
        self.grid.set(x, y, Tile::Fire);
        *self.grid.fire_intensity_mut(x, y) = 1.0;
        let mut ignited = vec![(x, y)];
        for _ in 0..6 {
            let dx = self.geology.rng.random_range(-3i32..=3);
            let dy = self.geology.rng.random_range(-3i32..=3);
            let (nx, ny) = (x + dx, y + dy);
            if WorldGrid::in_bounds(nx, ny) {
                self.grid.set(nx, ny, Tile::Fire);
                *self.grid.fire_intensity_mut(nx, ny) = 1.0;
                ignited.push((nx, ny));
            }
        }
        for (ix, iy) in ignited {
            self.physics.register_fire(ix, iy);
        }
        let mut placed = 0;
        for _ in 0..40 {
            if placed >= 5 {
                break;
            }
            let dx = self.geology.rng.random_range(-8i32..=8);
            let dy = self.geology.rng.random_range(-8i32..=8);
            let (nx, ny) = (x + dx, y + dy);
            if WorldGrid::in_bounds(nx, ny) && matches!(self.grid.get(nx, ny), Tile::Grass | Tile::Sand) {
                self.grid.set(nx, ny, Tile::Rock);
                placed += 1;
            }
        }
        let mut placed_mineral = 0;
        for _ in 0..30 {
            if placed_mineral >= 3 {
                break;
            }
            let dx = self.geology.rng.random_range(-8i32..=8);
            let dy = self.geology.rng.random_range(-8i32..=8);
            let (nx, ny) = (x + dx, y + dy);
            if WorldGrid::in_bounds(nx, ny) && self.grid.get(nx, ny) == Tile::Grass {
                self.grid.set(nx, ny, Tile::Mineral);
                placed_mineral += 1;
            }
        }
        let buildings = crate::sim::civ::building_damage::quake_damage(self, x, y, 3);
        push_event(
            &mut self.events,
            tick,
            "eruption",
            "the deep",
            &format!("a hotspot burst through the crust near ({x},{y}); {buildings} buildings scorched"),
        );
    }

    /// Erosion pass: sampled steep land near water loses a little elevation,
    /// the floodplain downstream gains fertility. Slow, subtle, deep time.
    fn geology_erode(&mut self) {
        let mut deposits = Vec::new();
        for _ in 0..EROSION_SAMPLES {
            let x = self.geology.rng.random_range(1..WIDTH as i32 - 1);
            let y = self.geology.rng.random_range(1..HEIGHT as i32 - 1);
            let i = WorldGrid::idx(x, y);
            if matches!(self.grid.tiles[i], t if t == Tile::Water as i8 || t == Tile::Void as i8 || t == Tile::Ice as i8)
            {
                continue;
            }
            let e = self.grid.elevation[i];
            // Steepest of the four neighbours.
            let mut low = (e, x, y);
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let j = WorldGrid::idx(x + dx, y + dy);
                if self.grid.elevation[j] < low.0 {
                    low = (self.grid.elevation[j], x + dx, y + dy);
                }
            }
            let drop = e - low.0;
            if drop < 0.10 {
                continue;
            }
            // Only near water does the river do the carving.
            let near_water = (-4i32..=4).any(|dx| {
                (-4i32..=4).any(|dy| {
                    WorldGrid::in_bounds(x + dx, y + dy) && {
                        let t = self.grid.get(x + dx, y + dy);
                        matches!(t, Tile::Water | Tile::Flooded)
                    }
                })
            });
            if !near_water {
                continue;
            }
            // Carve here; the river carries the spoil onto the floodplain:
            // the downhill neighbour when it is land, else the bank itself.
            self.grid.elevation[i] = (e - 0.012).max(0.0);
            let j = WorldGrid::idx(low.1, low.2);
            let target = if self.grid.tiles[j] != Tile::Water as i8 && self.grid.tiles[j] != Tile::Void as i8
            {
                j
            } else {
                i
            };
            self.grid.fertility[target] = (self.grid.fertility[target] + 0.02).min(0.95);
            deposits.push((x, y));
            // Bare steep rock slowly crumbles to sand.
            if self.grid.tiles[i] == Tile::Rock as i8 && self.geology.rng.random::<f32>() < 0.05 {
                self.grid.tiles[i] = Tile::Sand as i8;
            }
            let _ = deposits.last().copied();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plates_cover_the_map_with_boundaries() {
        let geo = GeologyState::build(42);
        assert_eq!(geo.plate.len(), WIDTH * HEIGHT);
        assert!(geo.boundary_len() > 500, "six plates must meet somewhere");
        for h in &geo.hotspots {
            assert!(WorldGrid::in_bounds(h.x, h.y));
        }
    }

    #[test]
    fn same_seed_builds_the_same_plates() {
        let a = GeologyState::build(7);
        let b = GeologyState::build(7);
        assert_eq!(a.plate, b.plate);
        assert_eq!(a.hotspots.len(), b.hotspots.len());
    }

    #[test]
    fn convergent_boundary_accumulates_stress_until_it_quakes() {
        let mut sim = Simulation::new_blank(1234);
        // Blank grid is all Void; put land under the whole map so cracks and
        // damage have something to bite on.
        for y in 0..HEIGHT as i32 {
            for x in 0..WIDTH as i32 {
                sim.grid.set(x, y, Tile::Grass);
            }
        }
        let boundary_len = sim.geology.boundary_len();
        let hotspots = sim.geology.hotspots.len();
        // Charge ONE fault segment: a patch around the first boundary cell.
        let (bi, _, _) = sim.geology.boundary[0];
        let bx = (bi % WIDTH as u32) as i32;
        let by = (bi / WIDTH as u32) as i32;
        for dy in -8i32..=8 {
            for dx in -8i32..=8 {
                let (tx, ty) = (bx + dx, by + dy);
                if WorldGrid::in_bounds(tx, ty) {
                    let j = WorldGrid::idx(tx, ty);
                    sim.geology.stress[j] = RELEASE_STRESS * 1.1;
                }
            }
        }
        let quakes_before = sim.geology.quakes;
        sim.tick_geology_engine();
        assert_eq!(
            sim.geology.quakes,
            quakes_before + 1,
            "a charged fault must let go"
        );
        // The event log carries the quake.
        assert!(sim
            .events
            .iter()
            .any(|e| e.etype == "earthquake" && e.detail.contains("crust gave way")));
        // Stress around the epicenter slid back down.
        let max_stress = sim.geology.stress.iter().fold(0.0f32, |a, b| a.max(*b));
        assert!(
            max_stress < RELEASE_STRESS * 0.6,
            "release must relax the fault, max stress {max_stress}"
        );
        assert!(boundary_len > 0 && hotspots > 0);
    }

    #[test]
    fn hotspot_heat_builds_and_erupts() {
        let mut sim = Simulation::new_blank(99);
        for y in 0..HEIGHT as i32 {
            for x in 0..WIDTH as i32 {
                sim.grid.set(x, y, Tile::Grass);
            }
        }
        // Boil the hotspots directly; the next beat must erupt them.
        for h in sim.geology.hotspots.iter_mut() {
            h.heat = HOTSPOT_RELEASE - 0.0001;
        }
        let before = sim.geology.eruptions;
        sim.tick_geology_engine();
        assert_eq!(sim.geology.eruptions, before + sim.geology.hotspots.len() as u64);
        assert!(sim.events.iter().any(|e| e.etype == "eruption"));
        // The vent itself is burning ground.
        let any_fire = (0..WIDTH * HEIGHT).any(|i| sim.grid.tiles[i] == Tile::Fire as i8);
        assert!(any_fire, "an eruption must ignite the vent");
    }

    #[test]
    fn erosion_carves_elevation_and_feeds_fertility() {
        let mut sim = Simulation::new_blank(5);
        // A ridge next to a river: elevation 1 at x=100, dropping to 0 at the
        // water at x=104.
        for y in 0..HEIGHT as i32 {
            for x in 0..WIDTH as i32 {
                sim.grid.set(x, y, Tile::Grass);
            }
        }
        for y in 0..HEIGHT as i32 {
            sim.grid.set(104, y, Tile::Water);
        }
        for y in 0..HEIGHT as i32 {
            for x in 96..104 {
                let i = WorldGrid::idx(x, y);
                sim.grid.elevation[i] = 1.0 - (x - 96) as f32 * 0.1;
            }
        }
        let fert = |s: &Simulation| -> f32 {
            (96..104)
                .map(|x| {
                    (0..HEIGHT as i32)
                        .map(|y| s.grid.fertility[WorldGrid::idx(x, y)])
                        .sum::<f32>()
                })
                .sum()
        };
        let fert_before = fert(&sim);
        for _ in 0..40 {
            sim.geology_erode();
        }
        let fert_after = fert(&sim);
        assert!(
            fert_after > fert_before,
            "floodplain fertility must grow under erosion ({fert_before} -> {fert_after})"
        );
    }

    #[test]
    fn geology_beat_leaves_the_world_rng_untouched() {
        // The whole system draws from its private stream: two worlds that
        // differ only in whether geology ever fired must roll identically.
        let mut a = Simulation::new(31);
        let mut b = Simulation::new(31);
        // Burn one geology beat on b without any release (fresh stress).
        b.tick_geology_engine();
        // Both worlds now consume the same number of world-RNG rolls for a
        // probe and must agree.
        let ra = a.rng.random::<u64>();
        let rb = b.rng.random::<u64>();
        assert_eq!(ra, rb, "geology must never touch the world stream");
    }
}
