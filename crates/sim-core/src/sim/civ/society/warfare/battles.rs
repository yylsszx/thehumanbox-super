use super::*;
use super::neuro::{self, Stance};
pub struct BattleInstitutions<'a> {
    pub lineage_eras: &'a HashMap<crate::intern::Sym, Era>,
    pub governments: &'a HashMap<String, Government>,
    pub buildings: &'a [Building],
    pub field_fortifications: &'a [FieldFortification],
    pub grid: &'a WorldGrid,
    pub in_war: &'a HashSet<String>,
    /// Forged weapon designs, for the emergent-arsenal damage multiplier.
    pub weapon_registry: &'a [super::armory::EmergentWeapon],
    /// Invented military techs / warlike ideas, as damage multipliers.
    pub emergent_combat: &'a HashMap<String, f32>,
}

/// A person killed in a battle. Battle deaths skip the mortality phase, so
/// this does the bookkeeping that phase does: the cause, the life story, and
/// the death counts (totals and the year the stats panel charts them in).
fn fall_in_battle(
    org: &mut crate::organism::organism::Organism,
    tick: u64,
    history: &mut crate::sim::simulation::History,
) {
    org.alive = false;
    org.death_cause = "war".to_string();
    org.log_life(tick, "death", format!("fell in battle at age {}", org.age));
    history.deaths_combat += 1;
    history.record_death(tick);
}

/// How close (in tiles) a fighter must be to a battle's place to fight in it: a battle ranges over the land around
/// it, but someone at home across the island is not in it.
pub const AT_BATTLE: i32 = 25;

/// Health below which a fighter may leave the battle.
const WITHDRAW_AT: f32 = 0.25;

// ---- Formation discipline (pwb's ring phalanx / marching column) ----

/// Same-side fighters standing within this many tiles of one another
/// fight as a formation.
const FORMATION_RADIUS: f32 = 3.0;
/// Shoulder-to-shoulder count that saturates the formation bonuses: the
/// seventh shoulder adds nothing a shield wall has not.
const COHESION_CAP: f32 = 6.0;
/// Outgoing damage bonus per shoulder-to-shoulder mate (up to +24%).
const FORMATION_STRIKE: f32 = 0.04;
/// Incoming damage the formation covers, per mate (up to 12%).
const FORMATION_COVER: f32 = 0.02;
/// A routed side barely strikes back while it runs.
const ROUTED_STRIKE: f32 = 0.25;

// ---- Battle morale (pwb's army morale) ----

/// Losses a side takes before its morale breaks, before the stiffeners:
/// cohesion (each full shoulder-to-shoulder line buys 6% more losses)
/// and the era (a gun line breaks later than a brawl).
const MORALE_BREAK_BASE: f32 = 0.45;
const MORALE_COHESION_STIFFEN: f32 = 0.06;
const MORALE_ERA_STIFFEN: f32 = 0.015;
/// Fear that floods a broken line, carrying the survivors home through
/// the flight machinery.
const ROUT_FEAR: f32 = 0.8;

/// Formation cohesion per fighter: how many same-side mates stand within
/// `FORMATION_RADIUS` (capped at `COHESION_CAP`).
pub(super) fn formation_cohesion(here: &[usize], organisms: &[crate::organism::organism::Organism]) -> Vec<f32> {
    here.iter()
        .map(|&i| {
            let (x, y) = (organisms[i].x, organisms[i].y);
            here.iter()
                .filter(|&&j| j != i)
                .filter(|&&j| {
                    (organisms[j].x - x).abs() + (organisms[j].y - y).abs() <= FORMATION_RADIUS
                })
                .count() as f32
        })
        .map(|c| c.min(COHESION_CAP))
        .collect()
}

/// The loss fraction at which a side's morale breaks: stiffer lines (in
/// formation) and later eras take more losses before they run.
pub(super) fn morale_break_threshold(avg_cohesion: f32, era: Era) -> f32 {
    MORALE_BREAK_BASE
        + avg_cohesion * MORALE_COHESION_STIFFEN
        + (era as u8 as f32) * MORALE_ERA_STIFFEN
}

/// A fighter just wounded below `WITHDRAW_AT` in this exchange may leave the
/// fight and make for home, instead of fighting on until they fall: the
/// fearful often, the bold seldom. Each fighter weighs it once, when the wound
/// comes; whoever stays fights on. Their side is weaker for every one who
/// goes, and a side that loses enough of its fighters, dead or gone, breaks.
fn withdraw_wounded(
    ids: &mut Vec<String>,
    index: &HashMap<String, usize>,
    organisms: &mut [crate::organism::organism::Organism],
    health_before: &HashMap<String, f32>,
    tick: u64,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> u32 {
    use rand::RngExt;
    let mut left = 0;
    ids.retain(|id| {
        let Some(&i) = index.get(id) else {
            return true;
        };
        let o = &mut organisms[i];
        let just_wounded = health_before.get(id).is_some_and(|&h| h >= WITHDRAW_AT) && o.health < WITHDRAW_AT;
        if !o.alive || !just_wounded {
            return true;
        }
        if rng.random::<f32>() >= 0.1 + 0.4 * o.traits.fear {
            return true;
        }
        o.think("wounded, leaving the fight", tick);
        o.log_life(tick, "battle", "came away wounded from a battle".to_string());
        o.wander_target = Some((o.home_x as i32, o.home_y as i32));
        left += 1;
        false
    });
    left
}

/// How a stance changes the damage a fighter deals.
#[inline]
fn stance_outgoing(s: Stance) -> f32 {
    match s {
        Stance::Advance | Stance::Hold => 1.0,
        // Kiting trades output for safety; a fighter breaking off barely
        // fights on the way out.
        Stance::Kite => 0.85,
        Stance::Withdraw => 0.5,
    }
}

/// How a stance changes the damage a fighter takes.
#[inline]
fn stance_incoming(s: Stance) -> f32 {
    match s {
        Stance::Advance => 1.0,
        // Held ground braces; kiting and withdrawal both present a moving,
        // sparse target.
        Stance::Hold => 0.92,
        Stance::Kite | Stance::Withdraw => 0.75,
    }
}

/// Fighters the neural stance pass has judged beaten — still healthy but
/// badly outmatched — break off before the wound forces it, mirroring pwb's
/// "散兵劣势撤整" rule. A side that pulls its doomed out early keeps them
/// alive for the next war.
fn disengage_outmatched(
    ids: &mut Vec<String>,
    index: &HashMap<String, usize>,
    organisms: &mut [crate::organism::organism::Organism],
    stances: &[(usize, Stance)],
    tick: u64,
) -> u32 {
    let beaten: HashSet<usize> = stances
        .iter()
        .filter(|&&(_, s)| s == Stance::Withdraw)
        .map(|&(i, _)| i)
        .collect();
    if beaten.is_empty() {
        return 0;
    }
    let mut left = 0;
    ids.retain(|id| {
        let Some(&i) = index.get(id) else {
            return true;
        };
        // Only those the network flags AND whose health is already sagging
        // go: a fresh fighter with a bad matchup presses on.
        if !beaten.contains(&i) || !organisms[i].alive || organisms[i].health >= 0.6 {
            return true;
        }
        let o = &mut organisms[i];
        o.think("outmatched, falling back", tick);
        o.log_life(tick, "battle", "broke off a losing battle".to_string());
        o.wander_target = Some((o.home_x as i32, o.home_y as i32));
        left += 1;
        false
    });
    left
}

pub fn tick_battles(
    tick: u64,
    rng: &mut rand_chacha::ChaCha8Rng,
    battles: &mut Vec<Battle>,
    treaties: &mut Vec<Treaty>,
    organisms: &mut [crate::organism::organism::Organism],
    events: &mut std::collections::VecDeque<crate::sim::simulation::Event>,
    history: &mut crate::sim::simulation::History,
    institutions: BattleInstitutions<'_>,
) {
    use rand::RngExt;
    let mut signed: Vec<(String, String, TreatyKind)> = Vec::new();
    let index: HashMap<String, usize> = {
        let wanted: HashSet<&str> = battles
            .iter()
            .filter(|battle| battle.ended_tick.is_none())
            .flat_map(|battle| {
                battle
                    .attacker_orgs
                    .iter()
                    .chain(&battle.defender_orgs)
                    .map(String::as_str)
            })
            .collect();
        let mut index = HashMap::default();
        for (i, o) in organisms.iter().enumerate() {
            if wanted.contains(o.id.as_str()) {
                index.entry(o.id.to_string()).or_insert(i);
            }
        }
        index
    };
    for battle in battles.iter_mut() {
        if battle.ended_tick.is_some() {
            continue;
        }

        let att_alive: Vec<usize> = battle
            .attacker_orgs
            .iter()
            .filter_map(|id| index.get(id).copied())
            .filter(|&i| organisms[i].alive)
            .collect();
        let def_alive: Vec<usize> = battle
            .defender_orgs
            .iter()
            .filter_map(|id| index.get(id).copied())
            .filter(|&i| organisms[i].alive)
            .collect();

        let defender_wall_bonus =
            nearby_operational_defense(institutions.buildings, &battle.defenders, battle.location)
                || nearby_field_fortification(
                    institutions.field_fortifications,
                    &battle.defenders,
                    battle.location,
                    institutions.grid,
                );

        let att_era = battle
            .attackers
            .first()
            .and_then(|lid| institutions.lineage_eras.get(lid.as_str()).copied())
            .unwrap_or(Era::PreStone);
        let def_era = battle
            .defenders
            .first()
            .and_then(|lid| institutions.lineage_eras.get(lid.as_str()).copied())
            .unwrap_or(Era::PreStone);
        // The emergent arsenal: fighters fielding their lineage's best forged
        // design hit up to 35% harder at the damage cap.
        let att_wmul = super::armory::lineage_weapon_multiplier(
            institutions.weapon_registry,
            battle.attackers.first().map(String::as_str),
            super::armory::armament_tier(att_era),
        ) * battle
            .attackers
            .first()
            .and_then(|lid| institutions.emergent_combat.get(lid.as_str()).copied())
            .unwrap_or(1.0);
        let def_wmul = super::armory::lineage_weapon_multiplier(
            institutions.weapon_registry,
            battle.defenders.first().map(String::as_str),
            super::armory::armament_tier(def_era),
        ) * battle
            .defenders
            .first()
            .and_then(|lid| institutions.emergent_combat.get(lid.as_str()).copied())
            .unwrap_or(1.0);
        let att_policy_bonus = battle
            .attackers
            .first()
            .map(|lid| military_policy_bonus(institutions.governments.get(lid)))
            .unwrap_or(0.0);
        let def_policy_bonus = battle
            .defenders
            .first()
            .map(|lid| military_policy_bonus(institutions.governments.get(lid)))
            .unwrap_or(0.0);

        let base_dmg = 0.06f32;
        let health_before: HashMap<String, f32> = att_alive
            .iter()
            .chain(def_alive.iter())
            .map(|&i| (organisms[i].id.to_string(), organisms[i].health))
            .collect();

        // A combatant can die to fire, illness, or another system between
        // battle ticks. Resolve the empty side below instead of indexing an
        // empty force while trying to manufacture one combat pair.
        //
        // Only those at the battle trade blows. A fighter still on the way (or
        // never set out) is not hurt where they stand: before, someone resting
        // at home a hundred tiles off "fell in battle".
        let at_battle = |i: &usize| {
            let o = &organisms[*i];
            (o.x as i32 - battle.location.0)
                .abs()
                .max((o.y as i32 - battle.location.1).abs())
                <= AT_BATTLE
        };
        let att_here: Vec<usize> = att_alive.iter().copied().filter(at_battle).collect();
        let def_here: Vec<usize> = def_alive.iter().copied().filter(at_battle).collect();

        // Neural stance pass: every fighter present sizes up their opposite
        // number (advance / hold / kite / withdraw) before blows land. The
        // outmatched disengage after the exchange instead of dying in place.
        let stances_for = |here: &[usize], foes: &[usize], era: Era| {
            here.iter()
                .enumerate()
                .map(|(k, &i)| {
                    let stance = if foes.is_empty() {
                        Stance::Advance
                    } else {
                        let foe = &organisms[foes[k % foes.len()]];
                        let o = &organisms[i];
                        let dist = (o.x - foe.x).abs().max(o.y - foe.y).abs();
                        let f = neuro::stance_features(
                            o,
                            foe,
                            combat_style_for(o, era),
                            dist,
                            here.len().saturating_sub(1),
                        );
                        neuro::neuro_stance(&f)
                    };
                    (i, stance)
                })
                .collect::<Vec<_>>()
        };
        let att_stances = stances_for(&att_here, &def_here, att_era);
        let def_stances = stances_for(&def_here, &att_here, def_era);
        let stance_by_org = |i: usize, stances: &[(usize, Stance)]| {
            stances
                .iter()
                .find(|&&(j, _)| j == i)
                .map(|&(_, s)| s)
                .unwrap_or(Stance::Advance)
        };

        // Formation discipline (pwb's ring phalanx): fighters of a side
        // standing shoulder to shoulder fight as a formation. March there
        // as a column and you arrive as one; straggle in alone and you
        // fight alone.
        let cohesion_att = formation_cohesion(&att_here, organisms);
        let cohesion_def = formation_cohesion(&def_here, organisms);

        let avg_cohesion_att = cohesion_att.iter().sum::<f32>() / att_here.len().max(1) as f32;
        let avg_cohesion_def = cohesion_def.iter().sum::<f32>() / def_here.len().max(1) as f32;

        let pairs = att_here.len().min(def_here.len());
        for k in 0..pairs {
            let ai = att_here[k];
            let di = def_here[k % def_here.len()];
            let att_style = combat_style_for(&organisms[ai], att_era);
            let def_style = combat_style_for(&organisms[di], def_era);
            let a_bonus = soldier_bonus(&organisms[ai]) + att_policy_bonus;
            let d_bonus = soldier_bonus(&organisms[di]) + def_policy_bonus;
            // Neural hit rolls: one probability draw and one damage-variance
            // draw per side, taken unconditionally so the RNG stream stays
            // dense whatever the network says. A hit lands 1.16x so the
            // expected damage of an exchange matches the old always-hit
            // variance formula (E[1.16 * p] ~= E[0.7 + 0.6u] = 1.0).
            let a_f = neuro::acc_features(&organisms[ai], &organisms[di], institutions.grid);
            let d_f = neuro::acc_features(&organisms[di], &organisms[ai], institutions.grid);
            let p_a = neuro::neuro_acc(&a_f);
            let p_d = neuro::neuro_acc(&d_f);
            let (roll_a, var_a) = (rng.random::<f32>(), rng.random::<f32>());
            let (roll_d, var_d) = (rng.random::<f32>(), rng.random::<f32>());
            // Gear crit draws, also taken unconditionally for the stream.
            let (crit_a, crit_d) = (rng.random::<f32>(), rng.random::<f32>());
            let a_stance = stance_by_org(ai, &att_stances);
            let d_stance = stance_by_org(di, &def_stances);
            // Personal gear (see organism::items): the attacker's weapon
            // adds its edge and the defender's mitigation blunts the blow.
            let g_a = crate::organism::items::GearStats::of(&organisms[ai].gear);
            let g_d = crate::organism::items::GearStats::of(&organisms[di].gear);
            // A defender behind a completed wall/field fortification takes
            // reduced damage. This multiplier has to sit on `a_dmg` (the
            // damage dealt *to* the defender); putting it on `d_dmg` made a
            // defensive structure boost the defender's own output by 50%
            // while granting no mitigation at all.
            let mut a_dmg = base_dmg
                * damage_multiplier(att_style)
                * (1.0 + a_bonus)
                * att_wmul
                * stance_outgoing(a_stance)
                * stance_incoming(d_stance)
                * (0.8 + var_a * 0.4)
                * if roll_a < p_a { 1.16 } else { 0.0 }
                * if defender_wall_bonus { 0.65 } else { 1.0 };
            let mut d_dmg = base_dmg
                * damage_multiplier(def_style)
                * (1.0 + d_bonus)
                * def_wmul
                * stance_outgoing(d_stance)
                * stance_incoming(a_stance)
                * (0.8 + var_d * 0.4)
                * if roll_d < p_d { 1.16 } else { 0.0 };
            // Gear: the weapon's edge rides the hit, the wearer's
            // mitigation blunts it, and a bowman's counter-party cannot
            // reach him for the full reply.
            a_dmg = (a_dmg + g_a.dmg) * (1.0 - g_d.mitigation);
            d_dmg = (d_dmg + g_d.dmg) * (1.0 - g_a.mitigation);
            // Formation: each shoulder-to-shoulder mate strengthens the
            // line's strike and covers it against the counter.
            let coh_a = cohesion_att
                .get(k)
                .copied()
                .unwrap_or(0.0);
            let coh_d = cohesion_def
                .get(k % def_here.len().max(1))
                .copied()
                .unwrap_or(0.0);
            a_dmg *= 1.0 + FORMATION_STRIKE * coh_a;
            a_dmg *= 1.0 - (FORMATION_COVER * coh_d).min(0.12);
            d_dmg *= 1.0 + FORMATION_STRIKE * coh_d;
            d_dmg *= 1.0 - (FORMATION_COVER * coh_a).min(0.12);
            // A routed side barely strikes back while it runs.
            if battle.routed_a {
                a_dmg *= ROUTED_STRIKE;
            }
            if battle.routed_d {
                d_dmg *= ROUTED_STRIKE;
            }
            if g_a.ranged && !g_d.ranged {
                d_dmg *= 0.5;
            }
            if g_d.ranged && !g_a.ranged {
                a_dmg *= 0.5;
            }
            if crit_a < g_a.crit {
                a_dmg *= crate::organism::items::CRIT_MULT;
            }
            if crit_d < g_d.crit {
                d_dmg *= crate::organism::items::CRIT_MULT;
            }

            organisms[di].health = (organisms[di].health - a_dmg).max(0.0);
            organisms[di].mark_harm(crate::organism::organism::Harm::War, tick);
            if organisms[di].health <= 0.0 && organisms[di].alive {
                fall_in_battle(&mut organisms[di], tick, history);
                battle.casualties_d += 1;
            }
            organisms[ai].health = (organisms[ai].health - d_dmg).max(0.0);
            organisms[ai].mark_harm(crate::organism::organism::Harm::War, tick);
            if organisms[ai].health <= 0.0 && organisms[ai].alive {
                fall_in_battle(&mut organisms[ai], tick, history);
                battle.casualties_a += 1;
            }
            // Gear: lifesteal heals the wielder for a cut of the damage it
            // dealt; thorns wound the attacker that landed the blow.
            if a_dmg > 0.0 && g_a.lifesteal > 0.0 && organisms[ai].alive {
                organisms[ai].health = (organisms[ai].health + a_dmg * g_a.lifesteal).min(1.0);
            }
            if a_dmg > 0.0 && g_d.thorns > 0.0 && organisms[ai].alive {
                organisms[ai].health = (organisms[ai].health - g_d.thorns).max(0.0);
                organisms[ai].mark_harm(crate::organism::organism::Harm::War, tick);
                if organisms[ai].health <= 0.0 && organisms[ai].alive {
                    fall_in_battle(&mut organisms[ai], tick, history);
                    battle.casualties_a += 1;
                }
            }
            if d_dmg > 0.0 && g_d.lifesteal > 0.0 && organisms[di].alive {
                organisms[di].health = (organisms[di].health + d_dmg * g_d.lifesteal).min(1.0);
            }
            if d_dmg > 0.0 && g_a.thorns > 0.0 && organisms[di].alive {
                organisms[di].health = (organisms[di].health - g_a.thorns).max(0.0);
                organisms[di].mark_harm(crate::organism::organism::Harm::War, tick);
                if organisms[di].health <= 0.0 && organisms[di].alive {
                    fall_in_battle(&mut organisms[di], tick, history);
                    battle.casualties_d += 1;
                }
            }
        }

        // Battle morale (pwb's army morale): losses bleed it, formation
        // cohesion and the era stiffen it. When it breaks the side routs —
        // fear floods the line and the flight machinery carries the
        // survivors home, and a routed side barely strikes back from the
        // moment it breaks.
        {
            let a_loss = battle.casualties_a as f32 / battle.initial_a.max(1) as f32;
            let d_loss = battle.casualties_d as f32 / battle.initial_d.max(1) as f32;
            if !battle.routed_a
                && !att_here.is_empty()
                && a_loss >= morale_break_threshold(avg_cohesion_att, att_era)
            {
                battle.routed_a = true;
                let name = battle
                    .attackers
                    .first()
                    .map(String::as_str)
                    .unwrap_or("A tribe");
                push_event(
                    events,
                    tick,
                    "battle_route",
                    name,
                    "the attackers' line broke — the survivors ran",
                );
                for &i in &att_alive {
                    if organisms[i].alive {
                        organisms[i].fear_level = organisms[i].fear_level.max(ROUT_FEAR);
                    }
                }
            }
            if !battle.routed_d
                && !def_here.is_empty()
                && d_loss >= morale_break_threshold(avg_cohesion_def, def_era)
            {
                battle.routed_d = true;
                let name = battle
                    .defenders
                    .first()
                    .map(String::as_str)
                    .unwrap_or("A tribe");
                push_event(
                    events,
                    tick,
                    "battle_route",
                    name,
                    "the defenders' line broke — the survivors ran",
                );
                for &i in &def_alive {
                    if organisms[i].alive {
                        organisms[i].fear_level = organisms[i].fear_level.max(ROUT_FEAR);
                    }
                }
            }
        }

        withdraw_wounded(
            &mut battle.attacker_orgs,
            &index,
            organisms,
            &health_before,
            tick,
            rng,
        );
        withdraw_wounded(
            &mut battle.defender_orgs,
            &index,
            organisms,
            &health_before,
            tick,
            rng,
        );
        disengage_outmatched(&mut battle.attacker_orgs, &index, organisms, &att_stances, tick);
        disengage_outmatched(&mut battle.defender_orgs, &index, organisms, &def_stances, tick);

        let att_now = battle
            .attacker_orgs
            .iter()
            .filter_map(|id| index.get(id).copied())
            .filter(|&i| organisms[i].alive)
            .count() as f32;
        let def_now = battle
            .defender_orgs
            .iter()
            .filter_map(|id| index.get(id).copied())
            .filter(|&i| organisms[i].alive)
            .count() as f32;
        let a_frac = att_now / battle.initial_a.max(1) as f32;
        let d_frac = def_now / battle.initial_d.max(1) as f32;
        let elapsed = tick.saturating_sub(battle.started_tick);

        let timed_out = elapsed >= MAX_BATTLE_TICKS;
        let a_broken = a_frac < STRENGTH_BREAK_FRAC;
        let d_broken = d_frac < STRENGTH_BREAK_FRAC;
        if a_broken || d_broken || timed_out {
            let outcome = if a_broken && d_broken {
                BattleOutcome::Stalemate
            } else if d_broken {
                BattleOutcome::AttackerVictory
            } else if a_broken {
                BattleOutcome::DefenderVictory
            } else if a_frac > d_frac {
                BattleOutcome::AttackerVictory
            } else if d_frac > a_frac {
                BattleOutcome::DefenderVictory
            } else {
                BattleOutcome::Stalemate
            };
            battle.ended_tick = Some(tick);
            battle.outcome = Some(outcome);

            let a_lid = battle.attackers.first().cloned().unwrap_or_default();
            let b_lid = battle.defenders.first().cloned().unwrap_or_default();
            push_event(
                events,
                tick,
                "battle_ended",
                &battle.id,
                &format!(
                    "{} vs {}: {} (cas {}/{})",
                    a_lid,
                    b_lid,
                    outcome.name(),
                    battle.casualties_a,
                    battle.casualties_d
                ),
            );

            let loser = match outcome {
                BattleOutcome::AttackerVictory => Some((b_lid.clone(), a_lid.clone())),
                BattleOutcome::DefenderVictory => Some((a_lid.clone(), b_lid.clone())),
                BattleOutcome::Stalemate => None,
            };
            // Spoils: the victors strip the fallen, era-gated by their own
            // lineage (see organism::items). One draw per surviving victor,
            // taken unconditionally for the stream.
            let victors: Vec<usize> = match outcome {
                BattleOutcome::AttackerVictory => att_alive.clone(),
                BattleOutcome::DefenderVictory => def_alive.clone(),
                BattleOutcome::Stalemate => Vec::new(),
            };
            let victor_era = match outcome {
                BattleOutcome::DefenderVictory => def_era,
                _ => att_era,
            };
            let tier = crate::organism::items::era_tier(victor_era as u8);
            for &vi in &victors {
                let roll = rng.random::<f32>();
                if !organisms[vi].alive || roll >= 0.06 {
                    continue;
                }
                if let Some(def) = crate::organism::items::roll_item(rng, tier) {
                    if crate::organism::items::equip(&mut organisms[vi].gear, def).is_some() {
                        let name = organisms[vi].name.clone();
                        push_event(
                            events,
                            tick,
                            "loot",
                            &name,
                            &format!("stripped {} from the fallen", def.name),
                        );
                    }
                }
            }
            if let Some((loser_lid, winner_lid)) = loser.filter(|_| !institutions.in_war.contains(&battle.id))
            {
                let dominance = if outcome == BattleOutcome::AttackerVictory {
                    a_frac
                } else {
                    d_frac
                };
                let kind = if dominance > 0.75 && rng.random::<f32>() < 0.35 {
                    TreatyKind::Vassalage
                } else if rng.random::<f32>() < 0.5 {
                    TreatyKind::NonAggression
                } else {
                    TreatyKind::Trade
                };
                signed.push((loser_lid, winner_lid, kind));
            }
        }
    }

    for (a, b, kind) in signed {
        let expires = tick.saturating_add(2000 + (rng.random::<u64>() % 4000));
        if !establish_treaty(treaties, organisms, &a, &b, kind, tick, expires) {
            continue;
        }
        push_event(
            events,
            tick,
            "treaty_signed",
            &a,
            &format!("{} signs {} with {}", a, kind.name(), b),
        );
        if matches!(kind, TreatyKind::Vassalage) {
            push_event(
                events,
                tick,
                "vassal_state",
                &a,
                &format!("{} becomes vassal of {}", a, b),
            );
        }
    }

    consolidate_treaties(treaties, tick);

    let mut active_finished: Vec<usize> = Vec::new();
    for (i, b) in battles.iter().enumerate() {
        if b.ended_tick.is_some() {
            active_finished.push(i);
        }
    }
    if battles.len() > 50 {
        let to_remove = battles.len() - 50;
        let mut removed = 0;
        let mut i = 0;
        while i < battles.len() && removed < to_remove {
            if battles[i].ended_tick.is_some() {
                battles.remove(i);
                removed += 1;
            } else {
                i += 1;
            }
        }
    }
    let _ = active_finished;
}

pub(super) fn combat_style_for(org: &crate::organism::organism::Organism, era: Era) -> CombatStyle {
    if org.has_tool("rifle") {
        if era >= Era::Modern {
            CombatStyle::Modern
        } else {
            CombatStyle::Rifle
        }
    } else if org.has_tool("musket") {
        CombatStyle::Musket
    } else if org.has_tool("sword") || org.has_tool("iron_sword") || org.has_tool("bow") {
        CombatStyle::Sword
    } else if org.has_tool("spear") || org.has_tool("bronze_spear") || org.has_tool("stone_spear") {
        CombatStyle::Spear
    } else {
        CombatStyle::Brawl
    }
}

pub(super) fn military_policy_bonus(government: Option<&Government>) -> f32 {
    let Some(government) = government else {
        return 0.0;
    };
    if !government.has_law(LawKind::MilitaryService) {
        return 0.0;
    }
    // An enacted service law improves coordination. A funded treasury adds a
    // smaller logistics bonus, capped so equipment and individual skill still
    // decide battles.
    0.08 + (government.treasury as f32 / 500.0).min(0.12)
}

pub(super) fn soldier_bonus(o: &crate::organism::organism::Organism) -> f32 {
    let training: f32 = match o.specialty.as_deref() {
        Some("officer") => 0.38,
        Some("soldier") => 0.24,
        _ => 0.0,
    };
    let equipment: f32 = if o.has_tool("rifle") {
        0.34
    } else if o.has_tool("musket") {
        0.25
    } else if o.has_tool("sword") || o.has_tool("bow") {
        0.16
    } else if o.has_tool("spear") || o.discoveries.contains("spear") {
        0.10
    } else {
        0.0
    };
    let readiness = (0.55 + 0.25 * o.energy + 0.20 * o.health).clamp(0.4, 1.0);
    (training + equipment) * readiness
}

pub(super) fn is_defensive_building(kind: BuildingKind) -> bool {
    matches!(
        kind,
        BuildingKind::Wall
            | BuildingKind::Tower
            | BuildingKind::Watchtower
            | BuildingKind::Barracks
            | BuildingKind::Castle
            | BuildingKind::Gate
            | BuildingKind::Fence
    )
}

pub(super) fn nearby_operational_defense(
    buildings: &[Building],
    defenders: &[String],
    loc: (i32, i32),
) -> bool {
    buildings.iter().any(|building| {
        if !building.is_operational() || !is_defensive_building(building.kind) {
            return false;
        }
        let Some(owner) = building.owner_lineage.as_deref() else {
            return false;
        };
        if !defenders.iter().any(|lineage| lineage == owner) {
            return false;
        }
        let (width, height) = building.footprint();
        let nearest_x = loc.0.clamp(building.x, building.x + i32::from(width) - 1);
        let nearest_y = loc.1.clamp(building.y, building.y + i32::from(height) - 1);
        (loc.0 - nearest_x).abs() <= 3 && (loc.1 - nearest_y).abs() <= 3
    })
}

pub(super) fn nearby_field_fortification(
    fortifications: &[FieldFortification],
    defenders: &[String],
    loc: (i32, i32),
    grid: &WorldGrid,
) -> bool {
    fortifications.iter().any(|fortification| {
        grid.structure_at(fortification.x, fortification.y) > 0.0
            && defenders
                .iter()
                .any(|lineage| lineage == &fortification.lineage_id)
            && (loc.0 - fortification.x).abs() <= 3
            && (loc.1 - fortification.y).abs() <= 3
    })
}
