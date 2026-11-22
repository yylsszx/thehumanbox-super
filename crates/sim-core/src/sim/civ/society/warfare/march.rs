//! Warbands march. When a raid or a border war starts, the attackers who are not already at the
//! place walk to it: each one plans its way along the roads (a road costs less to walk, so the
//! route keeps to them where it can) and sets out, instead of standing where it was picked.
//! A few marchers per battle, so the walk reads as a band on the road and does not clear a town.
//! The defenders picked for the fight hurry there too: only those at the place fight.

use super::Battle;
use crate::hashing::FxHashMap as HashMap;
use crate::sim::agents::family_outings::chebyshev;
use crate::sim::simulation::Simulation;
use crate::sim::world_events::push_event;

/// A bigger war fields a bigger column: the marching band scales with
/// the battle (pwb's army-size discipline — a war is not walked to by a
/// raiding party's worth of men).
fn column_for_scale(scale: &super::BattleScale) -> usize {
    match scale {
        super::BattleScale::Skirmish | super::BattleScale::Raid => 6,
        super::BattleScale::Siege => 10,
        super::BattleScale::Battle => 14,
        super::BattleScale::War => 18,
    }
}
/// An attacker this close to the battle's place is already there and does not walk.
const AT_PLACE: i32 = 6;

impl Simulation {
    /// Sends the attackers of each new battle who are not yet at its place to march there along the roads.
    pub(crate) fn march_warbands(&mut self, battles: &[Battle]) {
        if battles.is_empty() {
            return;
        }
        let tick = self.tick_count;
        let index: HashMap<String, usize> = self
            .organisms
            .iter()
            .enumerate()
            .filter(|(_, o)| o.alive)
            .map(|(i, o)| (o.id.to_string(), i))
            .collect();
        for battle in battles {
            let target = battle.location;
            if !self.grid.get(target.0, target.1).walkable() {
                continue;
            }
            // The column this battle walks in: a war marches a wider band
            // down the roads than a raid.
            let band_cap = column_for_scale(&battle.scale);
            // Attackers march to the place; defenders hurry there to meet them.
            let mut marched = 0usize;
            for (ids, thought) in [
                (&battle.attacker_orgs, "marching to the battle"),
                (&battle.defender_orgs, "hurrying to defend the tribe"),
            ] {
                let mut side = 0usize;
                for id in ids {
                    if side >= band_cap {
                        break;
                    }
                    let Some(&i) = index.get(id) else { continue };
                    let o = &self.organisms[i];
                    if o.journey.is_some()
                        || o.wander_target.is_some()
                        || o.energy < 0.35
                        || o.hydration < 0.35
                        || o.fear_level > 0.5
                    {
                        continue;
                    }
                    let here = (o.x as i32, o.y as i32);
                    if chebyshev(here, target) <= AT_PLACE {
                        continue;
                    }
                    if self.organisms[i].march_along_roads(&self.grid, target, thought, tick) {
                        side += 1;
                        if thought == "marching to the battle" {
                            marched += 1;
                        }
                    }
                }
            }
            if marched > 0 {
                let name = battle
                    .attackers
                    .first()
                    .and_then(|lid| self.lineage_names.get(lid))
                    .cloned()
                    .unwrap_or_else(|| "A tribe".to_string());
                let detail = format!(
                    "a band of {marched} marched along the roads to ({}, {})",
                    target.0, target.1
                );
                push_event(&mut self.events, tick, "warfare", &name, &detail);
            }
        }
    }
}
