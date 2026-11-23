use super::*;

/// Ground a battle stands on: walkable and dry — a war band neither fights
/// in open water nor on sheer rock (swimming is travel, not a front line).
pub(super) fn dry_ground(grid: &crate::world::grid::WorldGrid, x: i32, y: i32) -> bool {
    !matches!(
        grid.get(x, y),
        crate::world::tiles::Tile::Water
            | crate::world::tiles::Tile::Void
            | crate::world::tiles::Tile::Rock
            | crate::world::tiles::Tile::Lava
    )
}

/// The ground a border war is fought on: the shared border itself.
///
/// Returns the centroid of the attacker-side tiles that touch the defender's
/// territory, snapped to the nearest such tile that stands on walkable land
/// (a border traced along a river's banks must not drop the battle in the
/// water). None when the two territories do not actually touch — a border war
/// without a border is not a border war.
pub(super) fn shared_front_point(
    grid: &crate::world::grid::WorldGrid,
    a_terr: &HashSet<(i32, i32)>,
    b_terr: &HashSet<(i32, i32)>,
) -> Option<(i32, i32)> {
    let mut front: Vec<(i32, i32)> = Vec::new();
    for &(x, y) in a_terr.iter() {
        for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1), (0, 1), (0, 0)] {
            if b_terr.contains(&(x + dx, y + dy)) {
                front.push((x, y));
                break;
            }
        }
    }
    if front.is_empty() {
        return None;
    }
    let (sx, sy): (i64, i64) = front
        .iter()
        .fold((0, 0), |(ax, ay), &(x, y)| (ax + x as i64, ay + y as i64));
    let (cx, cy) = ((sx / front.len() as i64) as i32, (sy / front.len() as i64) as i32);
    if crate::world::grid::WorldGrid::in_bounds(cx, cy) && dry_ground(grid, cx, cy) {
        return Some((cx, cy));
    }
    // The centroid missed the dry land (a river border, say): stand on the
    // contact tile nearest the centroid that is dry. Ties break by
    // coordinates so the choice is deterministic.
    front.sort_unstable_by_key(|&(x, y)| ((x - cx).abs() + (y - cy).abs(), x, y));
    front.iter().copied().find(|&(x, y)| dry_ground(grid, x, y))
}

/// Where a raid strikes: the defender's border tile nearest the raider's
/// people — an invasion crosses the frontier and lands inside the defender's
/// claimed ground, not at the empty midpoint between two tribes. None if the
/// defender holds no walkable land.
pub(super) fn invasion_point(
    grid: &crate::world::grid::WorldGrid,
    d_terr: &HashSet<(i32, i32)>,
    from: (i32, i32),
) -> Option<(i32, i32)> {
    let mut ranked: Vec<(i32, i32, i32)> = d_terr
        .iter()
        .copied()
        .map(|(x, y)| ((x - from.0).abs() + (y - from.1).abs(), x, y))
        .collect();
    ranked.sort_unstable();
    ranked
        .into_iter()
        .find(|&(_, x, y)| dry_ground(grid, x, y))
        .map(|(_, x, y)| (x, y))
}

pub fn scale_for_participants(n: usize) -> BattleScale {
    if n >= BattleScale::War.min_participants() {
        BattleScale::War
    } else if n >= BattleScale::Battle.min_participants() {
        BattleScale::Battle
    } else if n >= BattleScale::Siege.min_participants() {
        BattleScale::Siege
    } else if n >= BattleScale::Raid.min_participants() {
        BattleScale::Raid
    } else {
        BattleScale::Skirmish
    }
}

pub const MAX_BATTLE_TICKS: u64 = 200;
pub const STRENGTH_BREAK_FRAC: f32 = 0.30;
pub const RAID_CHECK_INTERVAL: u64 = 240;
pub const RAID_ATTITUDE_THRESHOLD: f32 = -0.45;

/// The ground a war spawner reads: the map it fights over, the claims drawn
/// on it, and the treaties and battles that already bind the tribes.
pub struct WarGround<'a> {
    pub grid: &'a crate::world::grid::WorldGrid,
    pub territory: &'a HashMap<crate::intern::Sym, HashSet<(i32, i32)>>,
    pub treaties: &'a [Treaty],
    pub active_battles: &'a [Battle],
}

/// How a tribe as a whole feels about each other tribe: the mean of its living
/// members' attitudes, with someone who has no opinion counting as neutral.
/// Before, the first member in the list spoke for everyone, so a single
/// person's grudge could send the whole tribe raiding.
pub fn tribe_mood(
    organisms: &[crate::organism::organism::Organism],
    members: &[usize],
) -> std::collections::BTreeMap<crate::intern::Sym, f32> {
    let mut sums: crate::hashing::FxHashMap<usize, (crate::intern::Sym, f32)> = Default::default();
    let alive: Vec<&crate::organism::organism::Organism> = members
        .iter()
        .map(|&i| &organisms[i])
        .filter(|o| o.alive)
        .collect();
    for o in &alive {
        for (lid, &att) in &o.lineage_attitudes {
            sums.entry(lid.addr()).or_insert((*lid, 0.0)).1 += att;
        }
    }
    let n = alive.len().max(1) as f32;
    sums.into_values().map(|(lid, v)| (lid, v / n)).collect()
}
pub const RAID_MIN_POP: usize = 8;

pub fn try_spawn_raids(
    tick: u64,
    rng: &mut rand_chacha::ChaCha8Rng,
    organisms: &[crate::organism::organism::Organism],
    ground: &WarGround<'_>,
    events: &mut std::collections::VecDeque<crate::sim::simulation::Event>,
    minds: &WarMinds<'_>,
    reasons: &mut Vec<(String, CasusBelli)>,
) -> Vec<Battle> {
    let grid = ground.grid;
    let territory = ground.territory;
    let treaties = ground.treaties;
    let active_battles = ground.active_battles;
    use rand::RngExt;
    let mut out = Vec::new();
    if !tick.is_multiple_of(RAID_CHECK_INTERVAL) {
        return out;
    }

    let mut pop_per_lineage: HashMap<String, Vec<usize>> = HashMap::default();
    for (i, o) in organisms.iter().enumerate() {
        if !o.alive {
            continue;
        }
        pop_per_lineage
            .entry(o.lineage_id.to_string())
            .or_default()
            .push(i);
    }

    let mut already_engaged: HashSet<String> = HashSet::default();
    for b in active_battles {
        if b.ended_tick.is_some() {
            continue;
        }
        for l in b.attackers.iter().chain(b.defenders.iter()) {
            already_engaged.insert(l.clone());
        }
    }
    already_engaged.extend(minds.at_war.iter().cloned());

    // Sorted: the loop below draws from `rng` per lineage, so HashMap order
    // decided the raid outcome and shifted the shared RNG stream.
    let mut lineage_ids: Vec<String> = pop_per_lineage.keys().cloned().collect();
    lineage_ids.sort();
    for a_lid in &lineage_ids {
        let a_pop = pop_per_lineage.get(a_lid).map(|v| v.len()).unwrap_or(0);
        if a_pop < RAID_MIN_POP {
            continue;
        }
        if already_engaged.contains(a_lid) {
            continue;
        }

        let mood = tribe_mood(organisms, &pop_per_lineage[a_lid]);
        let mut known: std::collections::BTreeSet<&str> =
            mood.keys().map(crate::intern::Sym::as_str).collect();
        known.extend(minds.relations.others_of(a_lid));
        let mut candidates: Vec<(String, f32, CasusBelli)> = known
            .into_iter()
            .map(|lid| {
                (
                    lid,
                    minds.opinion(a_lid, lid, mood.get(lid).copied().unwrap_or(0.0)),
                )
            })
            .filter(|(lid, att)| {
                *att <= RAID_ATTITUDE_THRESHOLD
                    && pop_per_lineage.get(*lid).map(|v| v.len()).unwrap_or(0) >= RAID_MIN_POP
                    && !already_engaged.contains(*lid)
                    && !has_active_treaty(treaties, a_lid, lid, tick)
                    && *lid != a_lid
                    && !minds.same_side(a_lid, lid)
            })
            .filter_map(|(lid, att)| {
                minds
                    .casus(a_lid, lid, false)
                    .map(|why| (lid.to_string(), att, why))
            })
            .collect();
        if candidates.is_empty() {
            continue;
        }

        // `lineage_attitudes` is a std `HashMap`, so equal attitudes (very
        // common once values saturate) were tie-broken by per-process hash
        // order — i.e. the raid target changed run to run. Break ties on the
        // lineage id so the choice is reproducible.
        candidates.sort_by(|x, y| {
            x.1.partial_cmp(&y.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| x.0.cmp(&y.0))
        });
        let (b_lid, _, why) = candidates.remove(0);

        if rng.random::<f32>() > 0.35 {
            continue;
        }

        let a_centroid = lineage_centroid(organisms, a_lid);
        let b_centroid = lineage_centroid(organisms, &b_lid);
        // Strike at the defender's frontier: the raid lands on the defender's
        // claimed ground facing the raider's people, so the fight (and the
        // occupation front it presses) happens on real, disputed land.
        let loc = territory
            .get(b_lid.as_str())
            .and_then(|d_terr| invasion_point(grid, d_terr, a_centroid))
            .unwrap_or_else(|| {
                let (ax, ay) = a_centroid;
                let (bx, by) = b_centroid;
                ((ax + bx) / 2, (ay + by) / 2)
            });

        let n_per_side = 3 + (rng.random::<u32>() % 3) as usize;
        let attacker_orgs = pick_combatants(organisms, &pop_per_lineage, a_lid, n_per_side, loc, rng);
        let defender_orgs = pick_combatants(organisms, &pop_per_lineage, &b_lid, n_per_side, loc, rng);
        if attacker_orgs.is_empty() || defender_orgs.is_empty() {
            continue;
        }

        let id = format!("battle_{}_{}_{}", tick, a_lid, b_lid);
        let initial_a = attacker_orgs.len() as u32;
        let initial_d = defender_orgs.len() as u32;
        let battle = Battle {
            id: id.clone(),
            attackers: vec![a_lid.clone()],
            defenders: vec![b_lid.clone()],
            attacker_orgs,
            defender_orgs,
            scale: BattleScale::Raid,
            location: loc,
            started_tick: tick,
            ended_tick: None,
            casualties_a: 0,
            casualties_d: 0,
            outcome: None,
            initial_a,
            initial_d,
            routed_a: false,
            routed_d: false,
        };
        already_engaged.insert(a_lid.clone());
        already_engaged.insert(b_lid.clone());
        reasons.push((id.clone(), why));
        push_event(
            events,
            tick,
            "raid_started",
            a_lid,
            &format!("raid against {} at ({},{})", b_lid, loc.0, loc.1),
        );
        push_event(
            events,
            tick,
            "battle_began",
            &id,
            &format!("{} attacks {} ({})", a_lid, b_lid, BattleScale::Raid.name()),
        );
        out.push(battle);
    }
    out
}

pub const BORDER_WAR_CHECK_INTERVAL: u64 = 480;
pub const BORDER_WAR_ATTITUDE_THRESHOLD: f32 = -0.20;
pub const BORDER_WAR_MIN_POP: usize = 18;
pub const BORDER_WAR_BORDER_OVERLAP_MIN: usize = 6;

pub fn try_spawn_border_wars(
    tick: u64,
    rng: &mut rand_chacha::ChaCha8Rng,
    organisms: &[crate::organism::organism::Organism],
    ground: &WarGround<'_>,
    events: &mut std::collections::VecDeque<crate::sim::simulation::Event>,
    minds: &WarMinds<'_>,
    reasons: &mut Vec<(String, CasusBelli)>,
) -> Vec<Battle> {
    let grid = ground.grid;
    let territory = ground.territory;
    let treaties = ground.treaties;
    let active_battles = ground.active_battles;
    use rand::RngExt;
    let mut out = Vec::new();
    if tick == 0 || !tick.is_multiple_of(BORDER_WAR_CHECK_INTERVAL) {
        return out;
    }

    let mut pop_per_lineage: HashMap<String, Vec<usize>> = HashMap::default();
    for (i, o) in organisms.iter().enumerate() {
        if !o.alive {
            continue;
        }
        pop_per_lineage
            .entry(o.lineage_id.to_string())
            .or_default()
            .push(i);
    }

    let mut already_engaged: HashSet<String> = HashSet::default();
    for b in active_battles {
        if b.ended_tick.is_some() {
            continue;
        }
        for l in b.attackers.iter().chain(b.defenders.iter()) {
            already_engaged.insert(l.clone());
        }
    }
    already_engaged.extend(minds.at_war.iter().cloned());

    // Sorted: the pair loop draws from `rng` and `break`s after the first
    // match, so HashMap order decided which pair went to war.
    let mut lineages: Vec<&String> = pop_per_lineage.keys().collect();
    lineages.sort();
    for i in 0..lineages.len() {
        let a_lid = lineages[i];
        let a_pop = pop_per_lineage.get(a_lid).map(|v| v.len()).unwrap_or(0);
        if a_pop < BORDER_WAR_MIN_POP || already_engaged.contains(a_lid) {
            continue;
        }
        let Some(a_terr) = territory.get(a_lid.as_str()) else {
            continue;
        };
        if a_terr.is_empty() {
            continue;
        }
        let mood = tribe_mood(organisms, &pop_per_lineage[a_lid]);

        for j in (i + 1)..lineages.len() {
            let b_lid = lineages[j];
            if already_engaged.contains(b_lid) {
                continue;
            }
            if has_active_treaty(treaties, a_lid, b_lid, tick) {
                continue;
            }
            let b_pop = pop_per_lineage.get(b_lid).map(|v| v.len()).unwrap_or(0);
            if b_pop < BORDER_WAR_MIN_POP {
                continue;
            }
            let Some(b_terr) = territory.get(b_lid.as_str()) else {
                continue;
            };
            if b_terr.is_empty() {
                continue;
            }
            if minds.same_side(a_lid, b_lid) {
                continue;
            }
            let att_ab = minds.opinion(a_lid, b_lid, mood.get(b_lid.as_str()).copied().unwrap_or(0.0));
            if att_ab > BORDER_WAR_ATTITUDE_THRESHOLD {
                continue;
            }
            let mut overlap = 0usize;
            for (x, y) in a_terr.iter() {
                for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1), (0, 1), (0, 0)] {
                    if b_terr.contains(&(x + dx, y + dy)) {
                        overlap += 1;
                        break;
                    }
                }
                if overlap >= BORDER_WAR_BORDER_OVERLAP_MIN {
                    break;
                }
            }
            if overlap < BORDER_WAR_BORDER_OVERLAP_MIN {
                continue;
            }
            let Some(why) = minds.casus(a_lid, b_lid, true) else {
                continue;
            };
            if rng.random::<f32>() > 0.55 {
                continue;
            }
            // The war is fought where the two countries actually meet: the
            // battle stands on the shared border, so the front it presses
            // bites into both sides' claimed ground and conquest visibly
            // moves the line on the map.
            let loc = shared_front_point(grid, a_terr, b_terr).unwrap_or_else(|| {
                let (ax, ay) = lineage_centroid(organisms, a_lid);
                let (bx, by) = lineage_centroid(organisms, b_lid);
                ((ax + bx) / 2, (ay + by) / 2)
            });
            let n_per_side = (8 + (rng.random::<u32>() % 6) as usize)
                .min(a_pop / 2)
                .min(b_pop / 2);
            let attacker_orgs = pick_combatants(organisms, &pop_per_lineage, a_lid, n_per_side, loc, rng);
            let defender_orgs = pick_combatants(organisms, &pop_per_lineage, b_lid, n_per_side, loc, rng);
            if attacker_orgs.is_empty() || defender_orgs.is_empty() {
                continue;
            }
            let total = attacker_orgs.len() + defender_orgs.len();
            let scale = scale_for_participants(total);
            let id = format!("border_war_{}_{}_{}", tick, a_lid, b_lid);
            let initial_a = attacker_orgs.len() as u32;
            let initial_d = defender_orgs.len() as u32;
            out.push(Battle {
                id: id.clone(),
                attackers: vec![a_lid.clone()],
                defenders: vec![b_lid.clone()],
                attacker_orgs,
                defender_orgs,
                scale,
                location: loc,
                started_tick: tick,
                ended_tick: None,
                casualties_a: 0,
                casualties_d: 0,
                outcome: None,
                initial_a,
                initial_d,
                routed_a: false,
                routed_d: false,
            });
            already_engaged.insert(a_lid.clone());
            already_engaged.insert(b_lid.clone());
            reasons.push((id.clone(), why));
            push_event(
                events,
                tick,
                "war_declared",
                a_lid,
                &format!(
                    "{} declared {} on {} over the border at ({},{})",
                    a_lid,
                    scale.name(),
                    b_lid,
                    loc.0,
                    loc.1
                ),
            );
            push_event(
                events,
                tick,
                "battle_began",
                &id,
                &format!(
                    "{} attacks {} ({}, {} v {})",
                    a_lid,
                    b_lid,
                    scale.name(),
                    initial_a,
                    initial_d
                ),
            );
            break;
        }
    }
    out
}

pub(super) fn lineage_centroid(organisms: &[crate::organism::organism::Organism], lid: &str) -> (i32, i32) {
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut n = 0.0f32;
    for o in organisms {
        if !o.alive || o.lineage_id != lid {
            continue;
        }
        sx += o.x;
        sy += o.y;
        n += 1.0;
    }
    if n < 0.5 {
        (0, 0)
    } else {
        ((sx / n) as i32, (sy / n) as i32)
    }
}

pub(super) fn pick_combatants(
    organisms: &[crate::organism::organism::Organism],
    pop_per_lineage: &HashMap<String, Vec<usize>>,
    lid: &str,
    n: usize,
    near: (i32, i32),
    rng: &mut rand_chacha::ChaCha8Rng,
) -> Vec<String> {
    use rand::seq::SliceRandom;
    let mut ids: Vec<(String, f32)> = match pop_per_lineage.get(lid) {
        Some(v) => v
            .iter()
            .filter(|&&i| organisms[i].alive && organisms[i].age_stage().can_combat())
            .map(|&i| {
                let dx = organisms[i].x - near.0 as f32;
                let dy = organisms[i].y - near.1 as f32;
                (organisms[i].id.to_string(), (dx * dx + dy * dy).sqrt())
            })
            .collect(),
        None => return Vec::new(),
    };
    ids.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let take = n.min(ids.len());
    let mut chosen: Vec<String> = ids.into_iter().take(take * 2).map(|(s, _)| s).collect();
    chosen.shuffle(rng);
    chosen.truncate(take);
    chosen
}
