use super::*;

pub(super) fn lookahead_ticks_for_values(look_ms: Option<&str>, tick_ms: Option<&str>) -> f32 {
    let look_ms = look_ms
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(150.0)
        .max(0.0);
    // Keep prediction cadence aligned with the server's bounded runtime
    // interval. Invalid or zero TICK_MS used to disable lookahead while the
    // server silently ran at its 100ms fallback, making movement stutter.
    let tick_ms = tick_ms
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(100.0)
        .clamp(16.0, 5_000.0);
    look_ms / tick_ms
}

pub(super) static LOOKAHEAD_TICKS: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| {
    // The browser build simulates in the same tab that renders, so there is
    // no network latency to hide. Predicting ahead there made delta frames
    // lead the true position while every full frame (true positions) pulled
    // people back, which read as jittering in place.
    if cfg!(target_arch = "wasm32") {
        return 0.0;
    }
    lookahead_ticks_for_values(
        std::env::var("LOOKAHEAD_MS").ok().as_deref(),
        std::env::var("TICK_MS").ok().as_deref(),
    )
});

pub(super) fn lineage_strategy_payload(sim: &Simulation) -> serde_json::Value {
    let active_strategies: HashMap<String, serde_json::Value> = sim
        .lineage_strategies
        .iter()
        .filter(|(_, (_, expires_tick))| *expires_tick > sim.tick_count)
        .map(|(lineage_id, (strategy, expires_tick))| {
            let objective = sim
                .lineage_strategy_objectives
                .get(lineage_id)
                .filter(|objective| objective.strategy == strategy.as_str());
            (
                lineage_id.clone(),
                json!({
                    "strategy": strategy,
                    "expires_tick": expires_tick,
                    "started_tick": objective.map(|objective| objective.started_tick).unwrap_or(sim.tick_count),
                    "progress": objective.map(|objective| objective.progress).unwrap_or(0),
                    "target": objective.map(|objective| objective.target).unwrap_or(0),
                    "completed": objective.and_then(|objective| objective.completed_tick).is_some(),
                    "completed_tick": objective.and_then(|objective| objective.completed_tick),
                    "status": if objective.and_then(|objective| objective.completed_tick).is_some() {
                        "completed"
                    } else {
                        "active"
                    },
                }),
            )
        })
        .collect();
    serde_json::to_value(active_strategies).unwrap()
}

/// Each tribe's own history, oldest first: `{lineage: [{tick, kind, other, word, x, y}]}`. A tribe that has
/// died out keeps its history too (its last entry is `extinct`), so its card and end summary stay reachable.
pub(super) fn tribe_chronicle_payload(sim: &Simulation) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (lid, entries) in sim.tribe_chronicle.tribes() {
        let rows: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| {
                json!({
                    "tick": e.tick,
                    "kind": e.kind,
                    "other": e.other,
                    "word": e.word,
                    "x": e.place.map(|p| p.0),
                    "y": e.place.map(|p| p.1),
                })
            })
            .collect();
        out.insert(lid.to_string(), serde_json::Value::Array(rows));
    }
    serde_json::Value::Object(out)
}

/// What each tribe's people are at work on right now, counted by trade: `{lineage: {skill: people}}`.
/// The tribe card reads it to say what a tribe is doing. Read only; a stint that has run out is not counted.
pub(super) fn lineage_work_payload(sim: &Simulation) -> serde_json::Value {
    let mut work: std::collections::BTreeMap<&str, std::collections::BTreeMap<&str, u32>> =
        std::collections::BTreeMap::new();
    for org in sim.organisms.iter().filter(|o| o.alive) {
        let Some(plan) = org.work else {
            continue;
        };
        if plan.until <= sim.tick_count {
            continue;
        }
        let trades = work.entry(org.lineage_id.as_str()).or_default();
        *trades.entry(plan.skill.name()).or_insert(0) += 1;
    }
    serde_json::to_value(work).unwrap_or_default()
}

pub(super) fn lineage_strategy_history_payload(sim: &Simulation) -> serde_json::Value {
    serde_json::to_value(
        sim.lineage_strategy_history
            .iter()
            .rev()
            .take(20)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

/// Each tribe's broadcast posture, for the tribe card's badge: `{lineage:
/// {posture, since, epochs, war_epochs, attack_mult}}`. The `attack_mult`
/// is the war-weariness-faded weight the posture actually broadcasts (1.0
/// outside `War`), rounded to two decimals. Sorted by lineage id; read only.
pub(super) fn tribe_posture_payload(sim: &Simulation) -> serde_json::Value {
    let mut postures: std::collections::BTreeMap<String, serde_json::Value> =
        std::collections::BTreeMap::new();
    for (lineage, brain) in &sim.hierarchy.tribes {
        let round2 = |v: f32| (f64::from(v) * 100.0).round() / 100.0;
        let attack_mult = if brain.action == crate::sim::hierarchy::TribeAction::War {
            crate::sim::hierarchy::tribe::effective_attack_mult(brain.war_epochs)
        } else {
            1.0
        };
        postures.insert(
            lineage.clone(),
            json!({
                "posture": brain.action.name(),
                "since": brain.action_started,
                "epochs": brain.epochs,
                "war_epochs": brain.war_epochs,
                "attack_mult": round2(attack_mult),
            }),
        );
    }
    serde_json::Value::Object(postures.into_iter().collect())
}

/// Each lineage's squads and the intent each currently dispatches, for the
/// squad panel and per-organism badges: `{lineage: [{id, intent, members,
/// ids, confidence, target}]}`. `ids` carries the alive members' organism
/// ids so the client can badge them without a membership join; dead members
/// (stale between rebuilds) are skipped. Squads sorted by id within a
/// lineage, lineages sorted by id; read only.
pub(super) fn squad_intents_payload(sim: &Simulation) -> serde_json::Value {
    let mut squads: std::collections::BTreeMap<String, Vec<serde_json::Value>> =
        std::collections::BTreeMap::new();
    for (lineage, brains) in &sim.hierarchy.squads {
        let mut listed: Vec<serde_json::Value> = Vec::new();
        for brain in brains.iter() {
            let mut ids: Vec<String> = Vec::with_capacity(brain.members.len());
            for &m in &brain.members {
                let Some(o) = sim.organisms.get(m) else {
                    continue;
                };
                if o.alive {
                    ids.push(o.id.to_string());
                }
            }
            if ids.is_empty() {
                continue;
            }
            let round2 = |v: f32| (f64::from(v) * 100.0).round() / 100.0;
            listed.push(json!({
                "id": brain.id,
                "intent": brain.intent.name(),
                "members": ids.len(),
                "ids": ids,
                "confidence": round2(brain.confidence),
                "target": [brain.target.0, brain.target.1],
            }));
        }
        listed.sort_by_key(|s| s["id"].as_u64().unwrap_or(0));
        squads.insert(lineage.clone(), listed);
    }
    serde_json::to_value(squads).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::hierarchy::{SquadBrain, SquadIntent, TribeAction, TribeBrain};

    #[test]
    fn tribe_posture_lists_lineages_sorted_with_faded_attack_weight() {
        let mut sim = Simulation::new(0x5CAD);
        let war = TribeBrain {
            action: TribeAction::War,
            action_started: 7,
            epochs: 9,
            war_epochs: 4,
            ..Default::default()
        };
        let ally = TribeBrain {
            action: TribeAction::Ally,
            ..Default::default()
        };
        sim.hierarchy.tribes.insert("line-z".to_string(), war);
        sim.hierarchy.tribes.insert("line-a".to_string(), ally);

        let payload = tribe_posture_payload(&sim);
        let keys: Vec<&String> = payload.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["line-a", "line-z"], "sorted by lineage id");
        assert_eq!(payload["line-z"]["posture"], "war");
        assert_eq!(payload["line-z"]["since"], 7);
        assert_eq!(payload["line-z"]["epochs"], 9);
        assert_eq!(payload["line-z"]["war_epochs"], 4);
        let mult = payload["line-z"]["attack_mult"].as_f64().unwrap();
        assert!(
            mult > 1.0 && mult < 1.6,
            "weariness must fade the broadcast below the fresh 1.6: {mult}"
        );
        assert_eq!(payload["line-a"]["posture"], "ally");
        assert_eq!(payload["line-a"]["attack_mult"], 1.0, "no war, no boost");
    }

    #[test]
    fn squad_intents_list_alive_member_ids_sorted_by_squad_id() {
        let mut sim = Simulation::new(0x60D5);
        let lineage = sim.organisms[0].lineage_id.to_string();
        let mut members = Vec::new();
        for (i, o) in sim.organisms.iter().enumerate() {
            if o.alive {
                members.push(i);
            }
            if members.len() == 3 {
                break;
            }
        }
        assert_eq!(members.len(), 3, "a fresh world has people");
        sim.organisms[members[2]].alive = false;
        let expected: Vec<String> = [members[0], members[1]]
            .iter()
            .map(|&i| sim.organisms[i].id.to_string())
            .collect();

        let band = SquadBrain {
            id: 11,
            intent: SquadIntent::Migrate,
            confidence: 0.5,
            target: (12, 34),
            members,
            ..Default::default()
        };
        let empty = SquadBrain {
            id: 3,
            members: Vec::new(), // no alive members → not listed
            ..Default::default()
        };
        sim.hierarchy.squads.insert(lineage.clone(), vec![band, empty]);

        let payload = squad_intents_payload(&sim);
        let arr = payload[&lineage].as_array().unwrap();
        assert_eq!(arr.len(), 1, "a squad with no alive members is not listed");
        assert_eq!(arr[0]["id"], 11, "squads sorted by id within the lineage");
        assert_eq!(arr[0]["intent"], "migrate");
        assert_eq!(arr[0]["members"], 2, "the dead member is skipped");
        let ids: Vec<&str> = arr[0]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(ids, expected);
        assert_eq!(arr[0]["confidence"], 0.5);
        assert_eq!(arr[0]["target"], serde_json::json!([12, 34]));
    }
}
