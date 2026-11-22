//! The village forge: smiths slowly equip their own.
//!
//! A completed forge or smithy owned by a lineage is that lineage's gear
//! shop: every `GEAR_FORGE_STEP` ticks, each of its adults within reach of
//! the anvil has a small chance to be fitted with the era's local pattern
//! (a rarity-weighted roll, era-gated by the lineage's armament tier —
//! see `organism::items`). The equip rank rule keeps the ratchet one-way,
//! so a village's fighters climb toward the era's best local kit without
//! ever trading down.

use super::*;

/// Cadence of the forge pass, in ticks.
pub(super) const GEAR_FORGE_STEP: u64 = 500;

/// Tiles a smithy's reach covers (the village around the anvil).
const FORGE_REACH: i32 = 30;
/// Chance per eligible adult per pass to be fitted with new gear.
const FIT_CHANCE: f32 = 0.02;

pub(super) fn tick_gear_forging(sim: &mut Simulation) {
    // The lineage's working forges and smithies, by owner.
    let mut anvils: Vec<(String, i32, i32)> = Vec::new();
    for b in sim.buildings.iter() {
        if !matches!(b.kind, BuildingKind::Forge | BuildingKind::Smithy | BuildingKind::Goldsmith)
            || !b.is_complete()
        {
            continue;
        }
        if let Some(owner) = b.owner_lineage.as_ref().map(|s| s.as_str().to_string()) {
            anvils.push((owner, b.x, b.y));
        }
    }
    if anvils.is_empty() {
        return;
    }
    // Each adult near their own lineage's anvil draws one roll; a hit fits
    // them with the era's local pattern (rank rule keeps it one-way).
    let mut to_fit: Vec<(usize, &'static crate::organism::items::ItemDef)> = Vec::new();
    for (oi, o) in sim.organisms.iter().enumerate() {
        if !o.alive
            || !matches!(
                crate::sim::age_stage::AgeStage::from_age(o.age, o.max_age),
                crate::sim::age_stage::AgeStage::Adult | crate::sim::age_stage::AgeStage::Elder
            )
        {
            continue;
        }
        let lid = o.lineage_id.as_str().to_string();
        let tier = sim
            .lineage_eras
            .get(&o.lineage_id)
            .copied()
            .map(|e| crate::organism::items::era_tier(e as u8))
            .unwrap_or(0);
        if tier == 0 {
            // A pre-metal lineage has no patterns worth forging.
            continue;
        }
        let near_anvil = anvils.iter().any(|(owner, ax, ay)| {
            owner == &lid && (o.x as i32 - ax).abs().max((o.y as i32 - ay).abs()) <= FORGE_REACH
        });
        if !near_anvil || sim.rng.random::<f32>() >= FIT_CHANCE {
            continue;
        }
        if let Some(def) = crate::organism::items::roll_item(&mut sim.rng, tier) {
            to_fit.push((oi, def));
        }
    }
    for (oi, def) in to_fit {
        if crate::organism::items::equip(&mut sim.organisms[oi].gear, def).is_some() {
            // A rare+ fitting makes the smith's news; everyday fittings
            // stay quiet so the log does not flood.
            if def.rarity >= 2 {
                let name = sim.organisms[oi].name.clone();
                push_event(
                    &mut sim.events,
                    sim.tick_count,
                    "loot",
                    &name,
                    &format!("was fitted with {} by the village forge", def.name),
                );
            }
        }
    }
}
