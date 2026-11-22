//! Atmosphere: a coarse moisture field advected by the wind makes rain a
//! consequence of geography instead of a flat coin flip.
//!
//! Ported from the Pixel World Box "nature" design, adapted here:
//!   * the moisture grid is 8-tile cells over the map (100 x 50), transient
//!     and rebuilt from the live grid, never persisted;
//!   * every ATMOSPHERE_STEP ticks the field advects upwind, gains evaporation
//!     over water and loses rain-out over land — so windward coasts soak and
//!     leeward interiors sit in a rain shadow, exactly like the real thing;
//!   * two couplings into the existing engine:
//!       1. `tick_weather` multiplies its rain probability by the atmosphere's
//!          moisture supply (dry spells actually dry out, wet coasts actually
//!          rain) — droughts and monsoons fall out of the feedback loop;
//!       2. a cell that saturates while the sky is clear can force a
//!          rain/storm event on its own, storms sweeping in off the sea.
//!   * all randomness comes from a private stream seeded from the world seed,
//!     so the world RNG rolls stay bit-identical with and without this system.

use super::*;
use crate::sim::simulation::Simulation;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

/// Ticks between atmosphere beats. Ten beats a day keeps the field cheap
/// (5000 cells) and slow enough that weather keeps its own rhythm.
pub const ATMOSPHERE_STEP: u64 = 60;
/// Cell size in tiles.
pub const CELL: usize = 8;
pub const AW: usize = WIDTH / CELL;
pub const AH: usize = HEIGHT / CELL;
/// Water fraction below which a cell counts as land for the supply average.
const LAND_WATER_CUT: f32 = 0.35;
/// Evaporation gained per beat over open water.
const EVAP: f32 = 0.06;
/// Rain-out lost per beat over land (doubled while it is actually raining).
const RAINOUT: f32 = 0.004;
/// A cell above this while the sky is clear can break into a storm.
const SATURATION: f32 = 2.5;

/// Transient atmospheric state, rebuilt from the live grid.
pub struct AtmosphereState {
    /// Moisture per coarse cell.
    moisture: Vec<f32>,
    /// Water fraction per coarse cell (refreshed from the grid each beat).
    water: Vec<f32>,
    /// Supply multiplier handed to `tick_weather`: how much moisture the
    /// air over the land actually carries.
    supply: f32,
    /// Private stream: seeded from the world seed, never the world RNG.
    rng: ChaCha8Rng,
}

impl AtmosphereState {
    pub fn build(seed: u64, grid: &WorldGrid) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x0bad_f00d);
        let mut moisture = vec![0.0f32; AW * AH];
        let mut water = vec![0.0f32; AW * AH];
        for cy in 0..AH {
            for cx in 0..AW {
                let mut wet = 0.0f32;
                for ty in 0..CELL {
                    for tx in 0..CELL {
                        let x = (cx * CELL + tx) as i32;
                        let y = (cy * CELL + ty) as i32;
                        if matches!(grid.get(x, y), Tile::Water | Tile::Flooded) {
                            wet += 1.0;
                        }
                    }
                }
                let frac = wet / (CELL * CELL) as f32;
                water[cy * AW + cx] = frac;
                moisture[cy * AW + cx] = 0.35 + frac * 1.6;
            }
        }
        let _ = rng.random::<f32>(); // keep the stream warm even if unused yet
        AtmosphereState {
            moisture,
            water,
            supply: 1.0,
            rng,
        }
    }

    /// How wet the air over the land is, as a rain-probability multiplier.
    pub fn supply(&self) -> f32 {
        self.supply
    }

    fn recompute_supply(&mut self) {
        let mut sum = 0.0f32;
        let mut land = 0.0f32;
        for c in 0..AW * AH {
            if self.water[c] < LAND_WATER_CUT {
                let w = 1.0 - self.water[c];
                sum += self.moisture[c] * w;
                land += w;
            }
        }
        if land < 1.0 {
            land = 1.0;
        }
        let mean = sum / land;
        // Mean land moisture of ~0.8 (damp baseline) maps to ~1.0; the clamp
        // keeps droughts from shutting rain off entirely and monsoons from
        // making it permanent.
        self.supply = (mean / 0.8).clamp(0.25, 2.2);
    }
}

impl Simulation {
    /// One atmosphere beat. Call every `ATMOSPHERE_STEP` ticks.
    pub(crate) fn tick_atmosphere(&mut self) {
        // Refresh the water map: droughts, floods and river meander move
        // coastlines, and the atmosphere must follow the real geography.
        for cy in 0..AH {
            for cx in 0..AW {
                let mut wet = 0.0f32;
                for ty in 0..CELL {
                    for tx in 0..CELL {
                        let x = (cx * CELL + tx) as i32;
                        let y = (cy * CELL + ty) as i32;
                        if matches!(self.grid.get(x, y), Tile::Water | Tile::Flooded) {
                            wet += 1.0;
                        }
                    }
                }
                self.atmosphere.water[cy * AW + cx] = wet / (CELL * CELL) as f32;
            }
        }

        // Advect: each cell takes moisture from upwind, gains evaporation
        // over its own water and rains out over its own land.
        let (wx, wy) = (self.weather.wind_x, self.weather.wind_y);
        let raining = self.weather.kind != 0;
        let mut next = vec![0.0f32; AW * AH];
        for cy in 0..AH {
            for cx in 0..AW {
                // Sample the cell the wind blows from, scaled so a full cell
                // of drift happens at wind magnitude 1.
                let src_x = (cx as f32 - wx).round() as i32;
                let src_y = (cy as f32 - wy).round() as i32;
                let incoming = if src_x >= 0
                    && src_x < AW as i32
                    && src_y >= 0
                    && src_y < AH as i32
                {
                    self.atmosphere.moisture[src_y as usize * AW + src_x as usize]
                } else {
                    // Air from off the map edge starts over the ocean.
                    1.2
                };
                let w = self.atmosphere.water[cy * AW + cx];
                let mut m = incoming * (1.0 - 0.02) + w * EVAP;
                let rainout = if raining { RAINOUT * 2.0 } else { RAINOUT };
                m -= (1.0 - w) * rainout * m.max(0.0);
                next[cy * AW + cx] = m.clamp(0.0, 4.0);
            }
        }
        self.atmosphere.moisture = next;
        self.atmosphere.recompute_supply();

        // Saturation: a soaked coastal cell under a clear sky breaks into
        // weather on its own. The private stream keeps the world RNG clean.
        if !raining && self.tick_count >= self.weather.wet_until {
            let mut peak = (0usize, 0.0f32);
            for c in 0..AW * AH {
                if self.atmosphere.moisture[c] > peak.1 {
                    peak = (c, self.atmosphere.moisture[c]);
                }
            }
            if peak.1 >= SATURATION {
                let p = 0.05 * self.goals.difficulty.disaster_mult();
                if self.atmosphere.rng.random::<f32>() < p {
                    let storm = peak.1 > SATURATION + 0.6;
                    self.weather.kind = if storm { 2 } else { 1 };
                    self.weather.start_tick = self.tick_count;
                    self.weather.duration = self.atmosphere.rng.random_range(300..800);
                    self.weather.intensity = 0.4 + self.atmosphere.rng.random::<f32>() * 0.6;
                    let cx = (peak.0 % AW) as i32 * CELL as i32 + CELL as i32 / 2;
                    let cy = (peak.0 / AW) as i32 * CELL as i32 + CELL as i32 / 2;
                    let what = if storm { "a storm" } else { "rain" };
                    push_event(
                        &mut self.events,
                        self.tick_count,
                        "weather",
                        "the sky",
                        &format!("{what} sweeps in off the sea near ({cx},{cy})"),
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_cells_start_wetter_than_land() {
        let mut sim = Simulation::new_blank(8);
        for y in 0..HEIGHT as i32 {
            for x in 0..WIDTH as i32 {
                sim.grid.set(x, y, Tile::Water);
            }
        }
        // Blank grid has no water at build time; rebuild the atmosphere.
        sim.atmosphere = AtmosphereState::build(8, &sim.grid);
        let all_wet = sim.atmosphere.water.iter().all(|&w| w > 0.9);
        assert!(all_wet);
        let wet_moist: f32 = sim.atmosphere.moisture.iter().sum();
        assert!(wet_moist / (AW * AH) as f32 > 1.0);
    }

    #[test]
    fn dry_land_starves_the_supply_and_water_soaks_it() {
        let mut dry = Simulation::new_blank(10);
        dry.atmosphere = AtmosphereState::build(10, &dry.grid);
        dry.atmosphere.water.iter_mut().for_each(|w| *w = 0.0);
        dry.atmosphere.moisture.iter_mut().for_each(|m| *m = 0.3);
        dry.atmosphere.recompute_supply();
        let dry_supply = dry.atmosphere.supply();

        let mut wet = Simulation::new_blank(10);
        wet.atmosphere = AtmosphereState::build(10, &wet.grid);
        // Half the world is ocean; the land half carries soaked air.
        for c in 0..AW * AH {
            let ocean = c % 2 == 0;
            wet.atmosphere.water[c] = if ocean { 0.9 } else { 0.0 };
            wet.atmosphere.moisture[c] = if ocean { 2.2 } else { 2.2 };
        }
        wet.atmosphere.recompute_supply();
        assert!(
            wet.atmosphere.supply() > dry_supply * 3.0,
            "wet air must rain far more readily ({} vs {})",
            wet.atmosphere.supply(),
            dry_supply
        );
        assert!(dry_supply < 0.5);
    }

    #[test]
    fn advection_with_east_wind_wets_the_downwind_column() {
        let mut sim = Simulation::new_blank(12);
        // A ocean strip on the west edge, land everywhere else.
        for y in 0..HEIGHT as i32 {
            for x in 0..WIDTH as i32 {
                sim.grid.set(x, y, if x < 40 { Tile::Water } else { Tile::Grass });
            }
        }
        sim.atmosphere = AtmosphereState::build(12, &sim.grid);
        sim.weather.wind_x = 1.0; // blowing east: moisture travels with it
        sim.weather.wind_y = 0.0;
        // Land starts bone dry.
        for c in 0..AW * AH {
            if sim.atmosphere.water[c] < LAND_WATER_CUT {
                sim.atmosphere.moisture[c] = 0.05;
            }
        }
        for _ in 0..30 {
            sim.tick_atmosphere();
        }
        // Far downwind stays drier than the coast behind it: classic rain
        // shadow gradient.
        let col = |x: usize| -> f32 {
            (0..AH)
                .map(|y| sim.atmosphere.moisture[y * AW + x])
                .sum::<f32>()
                / AH as f32
        };
        let coast = col(6);
        let downwind = col(AW - 2);
        assert!(
            coast > downwind,
            "windward coast ({coast}) must hold more moisture than the far interior ({downwind})"
        );
    }

    #[test]
    fn atmosphere_beat_leaves_the_world_rng_untouched() {
        let mut a = Simulation::new(17);
        let mut b = Simulation::new(17);
        b.tick_atmosphere();
        let ra = a.rng.random::<u64>();
        let rb = b.rng.random::<u64>();
        assert_eq!(ra, rb, "atmosphere must never touch the world stream");
    }
}
