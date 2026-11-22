//! Gear integration: the village forge equips its lineage, trophies come
//! off the kill, and gear rides the save.

use super::*;
use crate::sim::buildings::{Building, BuildingKind};
use crate::sim::era::Era;

#[test]
fn the_village_forge_equips_its_lineage() {
    let mut sim = Simulation::new(0x6EA3);
    // A bronze-era lineage with a working smithy by its own hearth.
    let (lid, ox, oy) = {
        let o = sim.organisms.iter().find(|o| o.alive).unwrap();
        (o.lineage_id, o.x as i32, o.y as i32)
    };
    let mut smithy =
        Building::new(9001, BuildingKind::Smithy, ox + 2, oy + 2, Some(lid.as_str().to_string()), 0);
    smithy.condition = 1.0;
    sim.buildings.push(smithy);
    sim.lineage_eras.insert(lid, Era::Bronze);
    // Run the civ tick on the forge pass cadence until the smith has
    // fitted the villagers (2% per adult per pass — forty passes fits
    // almost everyone).
    for i in 1..=40u64 {
        sim.tick_count = 500 * i;
        crate::sim::civ::civ_tick::tick_civ(&mut sim, None);
    }
    let geared = sim
        .organisms
        .iter()
        .filter(|o| o.alive && !o.gear.is_empty())
        .count();
    assert!(geared > 0, "the forge equips the village ({geared} geared)");
    // Bronze tier only: no firearms ever come off an bronze anvil.
    for o in sim.organisms.iter().filter(|o| o.alive) {
        for key in o.gear.values() {
            let def = crate::organism::items::item_def(key).unwrap();
            assert!(def.tier <= 1, "bronze forge leaked {} (tier {})", def.key, def.tier);
        }
    }
}

#[test]
fn gear_rides_the_save() {
    let mut path = std::env::temp_dir();
    path.push(format!("thehumanbox-gear-test-{}.json", std::process::id()));
    let path_s = path.to_string_lossy().to_string();
    let _ = std::fs::remove_file(&path_s);
    let _ = std::fs::remove_file(format!("{}.tmp", path_s));

    let mut sim = Simulation::new(0x6EA4);
    let org_id = sim.organisms[0].id.clone();
    // Dress them from the table directly.
    let def = crate::organism::items::item_def("sword3").unwrap();
    crate::organism::items::equip(&mut sim.organisms[0].gear, def);
    let def = crate::organism::items::item_def("iron_shield").unwrap();
    crate::organism::items::equip(&mut sim.organisms[0].gear, def);

    sim.save_result(&path_s).unwrap();
    let loaded = Simulation::load_or_new(999, &path_s);
    let org = loaded
        .organisms
        .iter()
        .find(|o| o.id == org_id)
        .expect("the organism rides the save");
    assert_eq!(
        org.gear.get("wep").map(String::as_str),
        Some("sword3"),
        "the golden holy sword rides the save"
    );
    assert_eq!(
        org.gear.get("shd").map(String::as_str),
        Some("iron_shield"),
        "the iron shield rides the save"
    );
    // The aggregate survives the roundtrip too.
    let g = crate::organism::items::GearStats::of(&org.gear);
    assert!((g.dmg - 18.0 * crate::organism::items::DMG_SCALE).abs() < 1e-6);
    let _ = std::fs::remove_file(&path_s);
    let _ = std::fs::remove_file(format!("{}.tmp", path_s));
}

#[test]
fn a_geared_fighter_strikes_harder_than_a_bare_one() {
    // The battle exchange reads GearStats::of directly; pin the aggregate
    // the rolls ride on both ways: edge on the weapon, mitigation on the
    // armour, both capped and scaled as documented.
    use crate::organism::items::{GearStats, DMG_SCALE, MITIGATION_CAP};
    let mut bare = std::collections::BTreeMap::new();
    let mut armed = std::collections::BTreeMap::new();
    crate::organism::items::equip(&mut armed, crate::organism::items::item_def("excalibur").unwrap());
    let bare_s = GearStats::of(&bare);
    let armed_s = GearStats::of(&armed);
    assert_eq!(bare_s.dmg, 0.0);
    assert!((armed_s.dmg - 34.0 * DMG_SCALE).abs() < 1e-6);
    assert!(armed_s.crit > 0.0);
    assert!(armed_s.mitigation <= MITIGATION_CAP);
    // Mitigation caps when the armour stacks.
    let mut towered = std::collections::BTreeMap::new();
    for k in [
        "dragon_shield",
        "mithril_shield",
        "dragon_scale",
        "gold_armor",
        "knight_helm",
    ] {
        crate::organism::items::equip(&mut towered, crate::organism::items::item_def(k).unwrap());
    }
    let towered_s = GearStats::of(&towered);
    assert_eq!(towered_s.mitigation, MITIGATION_CAP, "armour caps at 60%");
    assert_eq!(towered.len(), 3, "one piece per slot: {:?}", towered);
}
