//! Integration tests for the multi-layer Q-learning hierarchy.

use super::*;
use crate::organism::organism::QRowExt;
use crate::sim::simulation::Simulation;

/// Force a tight kin cluster: the first living organisms become one lineage
/// standing together, so squad tests do not depend on world-gen layout.
fn force_cluster(sim: &mut Simulation, count: usize) -> String {
    let lineage = sim.organisms[0].lineage_id.to_string();
    let (bx, by) = {
        let o = &sim.organisms[0];
        (o.x as i32, o.y as i32)
    };
    let mut placed = 0;
    for i in 0..sim.organisms.len() {
        if placed >= count {
            break;
        }
        let o = &mut sim.organisms[i];
        if !o.alive {
            continue;
        }
        o.lineage_id = lineage.clone().into();
        o.x = bx as f32 + (placed % 3) as f32;
        o.y = by as f32 + (placed / 3) as f32;
        o.energy = 0.8;
        o.hydration = 0.8;
        o.health = 0.9;
        placed += 1;
    }
    assert!(placed >= count, "world too small for the cluster");
    lineage
}

#[test]
fn hierarchy_rolls_are_deterministic_and_spread() {
    let a = hierarchy_roll(120, 7, "Nevjom");
    let b = hierarchy_roll(120, 7, "Nevjom");
    assert_eq!(a, b);
    assert_ne!(a, hierarchy_roll(121, 7, "Nevjom"));
    let mut lo = 0usize;
    let mut hi = 0usize;
    for k in 0..200u64 {
        let r = hierarchy_roll(k, k ^ 0x51, "spread");
        assert!((0.0..=1.0).contains(&r));
        if r < 0.5 {
            lo += 1;
        } else {
            hi += 1;
        }
    }
    assert!(lo > 40 && hi > 40, "rolls should cover the unit range: {lo}/{hi}");
}

#[test]
fn squads_form_and_dispatch_intents() {
    let mut sim = Simulation::new(0x5CAD);
    force_cluster(&mut sim, 6);
    assert!(sim.hierarchy.squads.is_empty());
    sim.tick_count = squad::SQUAD_REBUILD_EVERY;
    sim.hierarchy_rebuild_squads();
    let squad_count: usize = sim.hierarchy.squads.values().map(Vec::len).sum();
    assert!(squad_count > 0, "expected squads after clustering kin");
    for brains in sim.hierarchy.squads.values() {
        for brain in brains {
            for &m in &brain.members {
                let o = &sim.organisms[m];
                assert!(o.squad_intent_until > sim.tick_count);
                assert!(o.squad_confidence >= 0.0);
            }
        }
    }
}

#[test]
fn squad_epoch_learns_and_backpropagates() {
    let mut sim = Simulation::new(0xBAC0);
    let lineage = force_cluster(&mut sim, 5);
    sim.tick_count = squad::SQUAD_REBUILD_EVERY;
    sim.hierarchy_rebuild_squads();

    // The cluster's squad: an epoch already dispatched Gather, members
    // obeyed, and the band's fortunes then improved.
    let mut brains = sim.hierarchy.squads.swap_remove(&lineage).unwrap_or_default();
    assert!(!brains.is_empty(), "the forced cluster must have a squad");
    let members = brains[0].members.clone();
    {
        let brain = &mut brains[0];
        brain.last_state = Some(0);
        brain.last_action = Some(SquadIntent::Gather);
        brain.epoch_stats = SquadStats {
            members: members.iter().filter(|&&m| sim.organisms[m].alive).count() as u32,
            avg_energy: 0.2,
            avg_health: 0.4,
            cohesion: 0.1,
            ..Default::default()
        };
    }
    for &m in &members {
        let o = &mut sim.organisms[m];
        if o.alive {
            o.squad_ledger[SquadIntent::Gather.ledger_slot()] = 20;
            o.learning_trace = vec![("0123456789abcdefghijk".to_string(), 10)];
            o.q_table
                .insert("0123456789abcdefghijk".to_string(), vec![(10, 0.0)]);
            o.hope = 0.0;
            o.joy_ticks = 0;
            o.energy = 0.95;
            o.health = 0.95;
        }
    }
    sim.hierarchy.squads.insert(lineage, brains);
    sim.tick_count = squad::SQUAD_REBUILD_EVERY + squad::SQUAD_EPOCH;
    sim.hierarchy_squad_epoch();

    let paid = members
        .iter()
        .filter(|&&m| sim.organisms[m].alive && sim.organisms[m].hope > 0.0)
        .count();
    assert!(paid > 0, "obedience should be rewarded when the band fares well");
    let credited = members
        .iter()
        .filter(|&&m| {
            let o = &sim.organisms[m];
            o.alive
                && o.q_table
                    .get("0123456789abcdefghijk")
                    .map_or(0.0, |row| row.get_q(10))
                    > 0.0
        })
        .count();
    assert!(
        credited > 0,
        "the squad's verdict must reach the members' tabular Q"
    );
    assert!(
        members
            .iter()
            .all(|&m| sim.organisms[m].squad_ledger == [0; LEDGER_SLOTS]),
        "the ledger clears at the epoch boundary"
    );
}

#[test]
fn tribe_epoch_broadcasts_a_posture() {
    let mut sim = Simulation::new(0x7A1E);
    let lineage = force_cluster(&mut sim, 5);
    sim.tick_count = tribe::TRIBE_EPOCH;
    sim.hierarchy_tribe_epoch();

    let brain = sim
        .hierarchy
        .tribes
        .get(&lineage)
        .expect("a lineage of five gets a tribe brain");
    assert!(brain.last_state.is_some());
    assert!(matches!(
        brain.action,
        TribeAction::War | TribeAction::Ally | TribeAction::Missionary | TribeAction::Research
    ));
    let policy = TribePolicy::of(brain.action);
    for o in sim
        .organisms
        .iter()
        .filter(|o| o.alive && o.lineage_id.as_str() == lineage.as_str())
    {
        assert_eq!(o.tribe_attack_mult, policy.attack_mult);
    }
}

#[test]
fn war_posture_pays_fight_rewards_and_charges_the_baseline() {
    let mut sim = Simulation::new(0x0FF1);
    let lineage = force_cluster(&mut sim, 3);
    sim.hierarchy
        .tribes
        .insert(lineage.clone(), TribeBrain::default());
    sim.hierarchy.tribes.get_mut(&lineage).unwrap().action = TribeAction::War;

    let idx = sim
        .organisms
        .iter()
        .position(|o| o.alive && o.lineage_id.as_str() == lineage.as_str())
        .unwrap();
    let fighting = sim.hierarchy_member_shaping(&lineage, idx, 12, 0.0, 0.0);
    let foraging = sim.hierarchy_member_shaping(&lineage, idx, 8, 0.0, 0.0);
    assert!(fighting > foraging, "war posture must pay combat above foraging");
    // The baseline is a (small) cost everyone carries.
    let idle = sim.hierarchy_member_shaping(&lineage, idx, 1000, 0.0, 0.0);
    assert!(idle < 0.0);
}

#[test]
fn gather_intent_pays_cohesion_and_expires() {
    let mut sim = Simulation::new(0x6A7E);
    let lineage = force_cluster(&mut sim, 3);
    let idx = sim
        .organisms
        .iter()
        .position(|o| o.alive && o.lineage_id.as_str() == lineage.as_str())
        .unwrap();
    sim.tick_count = 100;
    {
        let o = &mut sim.organisms[idx];
        o.squad_intent = SquadIntent::Gather;
        o.squad_intent_until = 200;
        o.squad_confidence = 1.0;
        o.squad_centroid = (o.x, o.y);
    }
    let at_centroid =
        sim.hierarchy_member_shaping(&lineage, idx, 10, sim.organisms[idx].x, sim.organisms[idx].y);
    assert!(at_centroid > 0.0, "standing with the band should pay");

    let far = sim.hierarchy_member_shaping(
        &lineage,
        idx,
        10,
        sim.organisms[idx].x + 50.0,
        sim.organisms[idx].y + 50.0,
    );
    assert!(far < at_centroid);

    sim.organisms[idx].squad_intent_until = 50;
    let expired = sim.hierarchy_member_shaping(&lineage, idx, 10, sim.organisms[idx].x, sim.organisms[idx].y);
    assert_eq!(expired, 0.0);
}

#[test]
fn hierarchy_survives_a_few_full_ticks() {
    let mut sim = Simulation::new(0x1234);
    force_cluster(&mut sim, 4);
    for _ in 0..12 {
        sim.tick();
    }
    for o in sim.organisms.iter().filter(|o| o.alive).take(10) {
        let b = o.hierarchy_choice_boost(12, sim.tick_count);
        assert!(b.abs() < 1.0);
    }
}

#[test]
fn squad_epoch_runs_inside_a_full_tick() {
    let mut sim = Simulation::new(0x2B1E);
    force_cluster(&mut sim, 5);
    // 120 ticks covers a rebuild (30) and a squad epoch (120).
    for _ in 0..120 {
        sim.tick();
    }
    for brains in sim.hierarchy.squads.values() {
        for brain in brains {
            assert!(brain.epochs >= 1, "squad epochs must fire at tick 120");
            assert!(brain.epoch_started > 0);
        }
    }
}

#[test]
fn state_vector_features_track_the_body() {
    let mut sim = Simulation::new(0x9D2);
    let idx = sim.organisms.iter().position(|o| o.alive).unwrap();
    {
        let o = &mut sim.organisms[idx];
        o.energy = 0.1;
        o.hydration = 0.9;
        o.fear_level = 0.3;
    }
    let nearby: Vec<usize> = Vec::new();
    let sv = sim.organisms[idx].compute_state_vector(&sim.organisms, &nearby, Some(3.0), 0);
    assert!(sv.hunger() > 0.85);
    assert!(sv.thirst() < 0.2);
    assert!(sv.enemy_proximity() > 0.7);
    assert_eq!(sv.threat_level(), 0.3);
    assert_eq!(sv.ally_count(), 0.0);
}

#[test]
fn obedience_payback_credits_the_tabular_q() {
    let mut sim = Simulation::new(0x1ED6);
    let idx = sim.organisms.iter().position(|o| o.alive).unwrap();
    // 21-byte state keys, so the generalisation row participates too.
    let follower = "0123456789abcdefghijk";
    let defied = "9876543210zyxwvutsrqp";
    {
        let o = &mut sim.organisms[idx];
        // The member obeyed a Gather dispatch (followed), once duplicating a
        // trace entry to prove the dedupe, and once defied (foraged).
        o.learning_trace = vec![
            (follower.to_string(), 10),
            (follower.to_string(), 10),
            (defied.to_string(), 8),
        ];
        o.q_table.insert(follower.to_string(), vec![(10, 0.01)]);
        o.q_table.insert(defied.to_string(), vec![(8, 0.01)]);
        o.squad_ledger[SquadIntent::Gather.ledger_slot()] = 20;
    }
    sim.organisms[idx].collect_squad_payback(SquadIntent::Gather, 0.4, 1.0);

    let o = &sim.organisms[idx];
    let follow_q = o.q_table.get(follower).unwrap().get_q(10);
    let expected = 0.01 + 0.06 * (0.05 - 0.01);
    assert!(
        (follow_q - expected).abs() < 1e-6,
        "follow Q should be pulled toward the squad's credit once: {follow_q}"
    );
    let defied_q = o.q_table.get(defied).unwrap().get_q(8);
    assert!(
        (defied_q - 0.01).abs() < 1e-9,
        "the defied slot must not be credited"
    );
    let coarse = crate::organism::organism::generalization_key(follower).unwrap();
    assert!(
        o.q_table.get(&coarse).unwrap().get_q(10) > 0.0,
        "the coarse row shares the credit"
    );
    // Morale still rides along with the tabular payback.
    assert!(o.hope > 0.0);
}

#[test]
fn payback_needs_the_ledger_to_vouch() {
    let mut sim = Simulation::new(0xD0D0);
    let idx = sim.organisms.iter().position(|o| o.alive).unwrap();
    let state = "0123456789abcdefghijk";
    {
        let o = &mut sim.organisms[idx];
        o.learning_trace = vec![(state.to_string(), 10)];
        o.q_table.insert(state.to_string(), vec![(10, 0.01)]);
        o.hope = 0.0;
        o.squad_ledger = [0; LEDGER_SLOTS];
    }
    sim.organisms[idx].collect_squad_payback(SquadIntent::Gather, 0.4, 1.0);

    let o = &sim.organisms[idx];
    assert_eq!(o.hope, 0.0, "no morale without ledger obedience");
    assert_eq!(
        o.q_table.get(state).unwrap().get_q(10),
        0.01,
        "no tabular credit either"
    );
}

#[test]
fn migrate_payback_credits_plain_steps() {
    let mut sim = Simulation::new(0x3757);
    let idx = sim.organisms.iter().position(|o| o.alive).unwrap();
    let state = "0123456789abcdefghijk";
    {
        let o = &mut sim.organisms[idx];
        o.learning_trace = vec![(state.to_string(), 3)];
        o.q_table.insert(state.to_string(), vec![(3, 0.0)]);
        o.squad_ledger[SquadIntent::Migrate.ledger_slot()] = 12;
    }
    sim.organisms[idx].collect_squad_payback(SquadIntent::Migrate, 0.3, 1.0);
    let q = sim.organisms[idx].q_table.get(state).unwrap().get_q(3);
    assert!(q > 0.0, "walking the march should be credited in the table: {q}");
}
