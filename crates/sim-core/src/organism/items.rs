//! Personal gear: pwb's item system ported into the organism layer.
//!
//! A person can carry one item per slot — weapon, armour, helmet, shield,
//! boots, ring, amulet — out of a hundred-item table with five rarities
//! (common / fine / rare / epic / legendary). Items come from three
//! channels, all era-gated by the lineage's armament tier:
//!
//!   * hunting: a rare beast occasionally yields trophy gear (a wolf-fang
//!     amulet off a wolf, a bearskin cloak off a bear);
//!   * spoils: the survivors of a victorious battle strip the fallen;
//!   * forging: a village with a working smithy slowly equips its own
//!     soldiers with the era's best local pattern.
//!
//! A pickup replaces the slot's current item only when the newcomer is the
//! stronger pattern (pwb's rank rule), so a person's gear only ratchets up.
//!
//! The stats fold into the existing combat rolls (battle exchanges, duels,
//! raids) and the health tick:
//!
//!   * `dmg` adds to outgoing damage (scaled by `DMG_SCALE`);
//!   * `armor` (+ the toughness folded out of pwb's `hp`) is percentage
//!     mitigation on incoming damage, capped at 60% over all pieces;
//!   * `crit` is a percentage chance to strike for 2.2x;
//!   * `lifesteal` heals the wielder for a percentage of damage dealt;
//!   * `thorns` wounds an attacker that lands a blow on the wearer;
//!   * `regen` adds to the natural health recovery each tick;
//!   * `ranged` (bows, staves, firearms): the wielder strikes from beyond
//!     the counter's reach — a melee-only foe's reply lands half-strength.
//!
//! pwb's `speedMul` has no THB counterpart (people have no per-person
//! speed stat) and is dropped; the ranged stat maps to first-strike reach
//! rather than a raw tile distance. Everything else is pwb's table,
//! translated item for item.

use std::collections::BTreeMap;

/// The seven equipment slots.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GearSlot {
    Weapon,
    Armor,
    Helm,
    Shield,
    Boots,
    Ring,
    Amulet,
}

impl GearSlot {
    pub const ALL: [GearSlot; 7] = [
        GearSlot::Weapon,
        GearSlot::Armor,
        GearSlot::Helm,
        GearSlot::Shield,
        GearSlot::Boots,
        GearSlot::Ring,
        GearSlot::Amulet,
    ];
    pub fn key(self) -> &'static str {
        match self {
            GearSlot::Weapon => "wep",
            GearSlot::Armor => "arm",
            GearSlot::Helm => "hlm",
            GearSlot::Shield => "shd",
            GearSlot::Boots => "bts",
            GearSlot::Ring => "rng",
            GearSlot::Amulet => "amu",
        }
    }
    pub fn from_key(key: &str) -> Option<Self> {
        GearSlot::ALL.into_iter().find(|s| s.key() == key)
    }
}

/// One item pattern from the hundred-item table.
#[derive(Clone, Copy, Debug)]
pub struct ItemDef {
    pub key: &'static str,
    pub name: &'static str,
    pub slot: GearSlot,
    /// Strength order within the slot: a pickup replaces weaker gear.
    pub rank: u8,
    /// 0 common .. 4 legendary (drop weights 46/26/15/9/4).
    pub rarity: u8,
    /// Armament tier the lineage must have reached before this pattern
    /// appears in loot or forging (see `era_tier`).
    pub tier: u8,
    pub dmg: f32,
    /// pwb's max-HP bonus, folded into toughness mitigation here.
    pub hp: f32,
    pub regen: f32,
    /// Percentage mitigation on incoming damage.
    pub armor: f32,
    /// Percentage chance to deal 2.2x damage.
    pub crit: f32,
    /// Percentage of damage dealt, healed back.
    pub lifesteal: f32,
    /// Fixed counter-wound on an attacker that lands a blow.
    pub thorns: f32,
    /// Bows, staves and firearms strike from beyond the counter's reach.
    pub ranged: bool,
}

const fn item(
    key: &'static str,
    name: &'static str,
    slot: GearSlot,
    rank: u8,
    rarity: u8,
    tier: u8,
) -> ItemDef {
    ItemDef {
        key,
        name,
        slot,
        rank,
        rarity,
        tier,
        dmg: 0.0,
        hp: 0.0,
        regen: 0.0,
        armor: 0.0,
        crit: 0.0,
        lifesteal: 0.0,
        thorns: 0.0,
        ranged: false,
    }
}

/// The hundred-item table, translated from pwb item for item. Tiers:
/// 0 always, 1 Bronze+, 2 Classical+, 3 Medieval+, 4 Industrial+,
/// 5 Modern+, 6 Information+.
pub const ITEM_TABLE: &[ItemDef] = &[
    // ---------- weapons: swords ----------
    item("sword1", "a rust-eaten shortsword", GearSlot::Weapon, 1, 0, 0)
        .with_dmg(5.0),
    item("sword2", "an iron-cast longsword", GearSlot::Weapon, 2, 0, 1)
        .with_dmg(11.0),
    item("sword3", "a golden holy sword", GearSlot::Weapon, 3, 3, 2)
        .with_dmg(18.0)
        .with_crit(5.0),
    item("wood_club", "a rough wooden club", GearSlot::Weapon, 0, 0, 0)
        .with_dmg(2.0),
    item("iron_dagger", "an iron dagger", GearSlot::Weapon, 1, 0, 1)
        .with_dmg(6.0)
        .with_crit(8.0),
    item("katana", "a katana", GearSlot::Weapon, 3, 2, 3)
        .with_dmg(16.0)
        .with_crit(12.0),
    item("elf_blade", "an elven blade", GearSlot::Weapon, 3, 2, 3)
        .with_dmg(15.0),
    item("flame_sword", "a flame-edged sword", GearSlot::Weapon, 4, 3, 3)
        .with_dmg(22.0)
        .with_thorns(6.0),
    item("ice_sword", "an ice-etched sword", GearSlot::Weapon, 4, 3, 3)
        .with_dmg(21.0)
        .with_armor(6.0),
    item("excalibur", "Excalibur", GearSlot::Weapon, 6, 4, 3)
        .with_dmg(34.0)
        .with_crit(10.0),
    item("holy_sword", "a sword of holy light", GearSlot::Weapon, 5, 3, 3)
        .with_dmg(26.0)
        .with_regen(0.4),
    // ---------- weapons: axes and hammers ----------
    item("battle_axe", "a battle axe", GearSlot::Weapon, 2, 0, 1).with_dmg(14.0),
    item("warhammer", "a war hammer", GearSlot::Weapon, 3, 1, 2).with_dmg(17.0),
    item("berserker_axe", "a berserker's great axe", GearSlot::Weapon, 4, 2, 2)
        .with_dmg(24.0),
    item("executioner_axe", "an executioner's axe", GearSlot::Weapon, 5, 2, 3)
        .with_dmg(28.0)
        .with_crit(8.0),
    item("thunder_hammer", "a thunder hammer", GearSlot::Weapon, 5, 4, 3)
        .with_dmg(30.0)
        .with_crit(15.0),
    item("storm_hammer", "a storm hammer", GearSlot::Weapon, 4, 3, 3)
        .with_dmg(23.0)
        .with_thorns(8.0),
    item("pickaxe", "a miner's pick", GearSlot::Weapon, 1, 0, 1).with_dmg(6.0),
    // ---------- weapons: spears ----------
    item("hunting_spear", "a hunting spear", GearSlot::Weapon, 2, 0, 0)
        .with_dmg(12.0),
    item("dragon_fang", "a dragon-fang spear", GearSlot::Weapon, 4, 3, 3)
        .with_dmg(23.0)
        .with_crit(6.0),
    item("dual_blades", "twin blades", GearSlot::Weapon, 3, 2, 3)
        .with_dmg(18.0)
        .with_crit(10.0),
    item("scythe", "a reaper's scythe", GearSlot::Weapon, 5, 4, 3)
        .with_dmg(26.0)
        .with_lifesteal(20.0),
    item("dragon_slayer", "a dragonslayer blade", GearSlot::Weapon, 5, 4, 3)
        .with_dmg(30.0),
    // ---------- weapons: bows ----------
    item("short_bow", "a short bow", GearSlot::Weapon, 1, 0, 0)
        .with_dmg(3.0)
        .as_ranged(),
    item("long_bow", "a long bow", GearSlot::Weapon, 2, 1, 2)
        .with_dmg(6.0)
        .as_ranged(),
    item("crossbow", "a heavy crossbow", GearSlot::Weapon, 3, 1, 3)
        .with_dmg(9.0)
        .as_ranged(),
    item("elf_bow", "an elven long bow", GearSlot::Weapon, 3, 2, 3)
        .with_dmg(8.0)
        .as_ranged(),
    item("eagle_bow", "an eagle-eyed bow", GearSlot::Weapon, 4, 3, 3)
        .with_dmg(11.0)
        .with_crit(8.0)
        .as_ranged(),
    item("hunt_bow", "a hunter's horn bow", GearSlot::Weapon, 2, 0, 0)
        .with_dmg(5.0)
        .as_ranged(),
    // ---------- weapons: firearms ----------
    item("musket", "a flintlock musket", GearSlot::Weapon, 4, 2, 4)
        .with_dmg(19.0)
        .with_crit(6.0)
        .as_ranged(),
    item("rifle", "a rifled rifle", GearSlot::Weapon, 5, 3, 4)
        .with_dmg(20.0)
        .with_crit(8.0)
        .as_ranged(),
    // ---------- weapons: future arms ----------
    item("laser_rifle", "a phase beam rifle", GearSlot::Weapon, 6, 4, 6)
        .with_dmg(28.0)
        .with_crit(10.0)
        .as_ranged(),
    item("plasma_caster", "a plasma caster", GearSlot::Weapon, 7, 4, 6)
        .with_dmg(34.0)
        .with_crit(12.0)
        .as_ranged(),
    // ---------- weapons: staves ----------
    item("wood_staff", "an apprentice's staff", GearSlot::Weapon, 1, 0, 0)
        .with_dmg(3.0)
        .as_ranged(),
    item("mage_staff", "a mage's staff", GearSlot::Weapon, 2, 1, 2)
        .with_dmg(6.0)
        .as_ranged(),
    item("storm_staff", "a storm staff", GearSlot::Weapon, 4, 3, 3)
        .with_dmg(12.0)
        .with_crit(6.0)
        .as_ranged(),
    item("archmage_staff", "an archmage's staff", GearSlot::Weapon, 5, 4, 3)
        .with_dmg(15.0)
        .with_crit(10.0)
        .as_ranged(),
    // ---------- armour ----------
    item("armor1", "leather armour", GearSlot::Armor, 1, 0, 0).with_hp(18.0),
    item("armor2", "heavy steel armour", GearSlot::Armor, 2, 1, 2).with_hp(36.0),
    item("cloth_robe", "a coarse cloth robe", GearSlot::Armor, 0, 0, 0).with_hp(6.0),
    item("studded_leather", "studded leather", GearSlot::Armor, 2, 0, 1)
        .with_hp(26.0)
        .with_armor(4.0),
    item("chain_mail", "a chainmail shirt", GearSlot::Armor, 3, 1, 2)
        .with_hp(40.0)
        .with_armor(8.0),
    item("iron_plate", "plate armour", GearSlot::Armor, 3, 1, 2)
        .with_hp(48.0)
        .with_armor(10.0),
    // ---------- armour: future ----------
    item("energy_shield", "an energy shield", GearSlot::Armor, 6, 4, 6)
        .with_hp(68.0)
        .with_armor(20.0),
    item("gold_armor", "golden armour", GearSlot::Armor, 4, 2, 3)
        .with_hp(55.0)
        .with_armor(12.0),
    item("dragon_scale", "a dragonscale coat", GearSlot::Armor, 5, 4, 3)
        .with_hp(75.0)
        .with_armor(18.0)
        .with_thorns(6.0),
    item("mithril_mail", "a mithril hauberk", GearSlot::Armor, 5, 3, 3)
        .with_hp(62.0)
        .with_armor(16.0),
    item("shadow_cloak", "a cloak of shadow", GearSlot::Armor, 3, 2, 3).with_hp(30.0),
    item("berserker_hide", "a berserker's hide", GearSlot::Armor, 2, 0, 0)
        .with_hp(30.0)
        .with_dmg(4.0),
    item("elf_robe", "an elven robe", GearSlot::Armor, 3, 2, 3)
        .with_hp(28.0)
        .with_regen(0.4),
    item("bone_armor", "a bone-warplate", GearSlot::Armor, 3, 1, 1)
        .with_hp(44.0)
        .with_armor(6.0)
        .with_thorns(5.0),
    item("obsidian_shell", "an obsidian shell", GearSlot::Armor, 4, 3, 2)
        .with_hp(60.0)
        .with_armor(15.0),
    item("holy_vestment", "a holy vestment", GearSlot::Armor, 4, 2, 3)
        .with_hp(38.0)
        .with_regen(0.8),
    item("emperor_robe", "an emperor's regalia", GearSlot::Armor, 5, 3, 3)
        .with_hp(70.0)
        .with_armor(10.0)
        .with_crit(5.0),
    // ---------- helmets ----------
    item("leather_cap", "a leather cap", GearSlot::Helm, 1, 0, 0).with_hp(6.0),
    item("archer_cap", "a huntsman's cap", GearSlot::Helm, 1, 0, 0).with_hp(5.0),
    item("iron_helm", "an iron helm", GearSlot::Helm, 2, 0, 1)
        .with_hp(14.0)
        .with_armor(4.0),
    item("horned_helm", "a horned helm", GearSlot::Helm, 2, 0, 1)
        .with_hp(12.0)
        .with_dmg(3.0),
    item("wizard_hat", "a wizard's pointed hat", GearSlot::Helm, 2, 1, 2).with_crit(6.0),
    item("knight_helm", "a knight's full helm", GearSlot::Helm, 3, 1, 3)
        .with_hp(22.0)
        .with_armor(8.0),
    item("warlock_hood", "a warlock's hood", GearSlot::Helm, 3, 2, 3).with_crit(10.0),
    item("viking_helm", "a viking war helm", GearSlot::Helm, 3, 1, 2)
        .with_hp(20.0)
        .with_armor(5.0)
        .with_dmg(4.0),
    item("skull_mask", "a skull mask", GearSlot::Helm, 3, 1, 1)
        .with_hp(18.0)
        .with_armor(6.0)
        .with_thorns(6.0),
    item("gold_crown", "a golden crown", GearSlot::Helm, 3, 2, 2)
        .with_hp(16.0)
        .with_dmg(4.0),
    item("dragon_skull", "a dragon-skull crown", GearSlot::Helm, 4, 3, 3)
        .with_hp(30.0)
        .with_armor(10.0)
        .with_dmg(6.0),
    item("crown_kings", "a crown of kings", GearSlot::Helm, 5, 4, 3)
        .with_hp(26.0)
        .with_dmg(8.0)
        .with_crit(8.0),
    // ---------- shields ----------
    item("round_shield", "a round wooden shield", GearSlot::Shield, 0, 0, 0)
        .with_armor(4.0),
    item("wood_shield", "a wooden shield", GearSlot::Shield, 1, 0, 0).with_armor(6.0),
    item("kite_shield", "a kite shield", GearSlot::Shield, 2, 0, 2)
        .with_armor(10.0)
        .with_hp(8.0),
    item("spiked_shield", "a spiked shield", GearSlot::Shield, 2, 0, 1)
        .with_armor(8.0)
        .with_thorns(12.0),
    item("iron_shield", "an iron shield", GearSlot::Shield, 2, 1, 2)
        .with_armor(12.0)
        .with_hp(10.0),
    item("tower_shield", "a tower shield", GearSlot::Shield, 3, 1, 3)
        .with_armor(18.0)
        .with_hp(16.0),
    item("gold_shield", "a golden shield", GearSlot::Shield, 4, 2, 3)
        .with_armor(20.0)
        .with_hp(12.0),
    item("holy_aegis", "a holy aegis", GearSlot::Shield, 4, 3, 3)
        .with_armor(22.0)
        .with_regen(0.5),
    item("mithril_shield", "a mithril shield", GearSlot::Shield, 5, 3, 3)
        .with_armor(24.0),
    item("dragon_shield", "a dragon-crested shield", GearSlot::Shield, 5, 4, 3)
        .with_armor(26.0)
        .with_hp(20.0)
        .with_thorns(8.0),
    // ---------- boots ----------
    item("worn_sandal", "a worn sandal", GearSlot::Boots, 0, 0, 0),
    item("leather_boots", "leather boots", GearSlot::Boots, 1, 0, 0),
    item("boots", "swift boots", GearSlot::Boots, 2, 1, 2),
    item("mountain_boots", "mountain boots", GearSlot::Boots, 2, 0, 1).with_hp(6.0),
    item("war_greaves", "war greaves", GearSlot::Boots, 2, 1, 2)
        .with_hp(14.0)
        .with_armor(3.0),
    item("iron_boots", "iron boots", GearSlot::Boots, 2, 0, 1)
        .with_hp(10.0)
        .with_armor(4.0),
    item("swift_greaves", "greaves of the gale", GearSlot::Boots, 3, 2, 3),
    item("lava_boots", "boots of lava", GearSlot::Boots, 4, 3, 3).with_armor(8.0),
    item("winged_sandals", "winged sandals", GearSlot::Boots, 4, 3, 3),
    item("phantom_boots", "phantom boots", GearSlot::Boots, 5, 4, 3).with_crit(6.0),
    // ---------- rings ----------
    item("copper_ring", "a copper ring", GearSlot::Ring, 1, 0, 0).with_dmg(3.0),
    item("silver_ring", "a silver ring", GearSlot::Ring, 1, 0, 0).with_hp(8.0),
    item("ring", "a ring of regeneration", GearSlot::Ring, 2, 1, 1).with_regen(0.4),
    item("gold_ring", "a golden ring", GearSlot::Ring, 2, 0, 2).with_hp(16.0),
    item("ruby_ring", "a ruby ring", GearSlot::Ring, 3, 1, 2).with_dmg(8.0),
    item("sapphire_ring", "a sapphire ring", GearSlot::Ring, 3, 1, 2).with_armor(8.0),
    item("ring_power", "a ring of power", GearSlot::Ring, 3, 2, 3)
        .with_dmg(10.0)
        .with_hp(10.0),
    item("ring_fire", "a ring of flame", GearSlot::Ring, 4, 3, 3)
        .with_dmg(12.0)
        .with_thorns(8.0),
    item("ring_ice", "a ring of ice", GearSlot::Ring, 4, 3, 3)
        .with_armor(10.0)
        .with_crit(8.0),
    item("ring_sage", "a sage's ring", GearSlot::Ring, 4, 3, 3)
        .with_crit(12.0)
        .with_regen(0.3),
    item("ring_void", "a ring of the void", GearSlot::Ring, 5, 4, 3)
        .with_dmg(14.0)
        .with_crit(10.0),
    item("ring_immortal", "the immortal's ring", GearSlot::Ring, 6, 4, 3)
        .with_regen(1.2)
        .with_hp(20.0),
    // ---------- amulets ----------
    item("lucky_charm", "a lucky charm", GearSlot::Amulet, 1, 0, 0).with_crit(5.0),
    item("wolf_fang", "a wolf-fang pendant", GearSlot::Amulet, 1, 0, 0).with_dmg(4.0),
    item("bear_claw", "a bear-claw chain", GearSlot::Amulet, 2, 0, 0)
        .with_dmg(6.0)
        .with_hp(8.0),
    item("totem_stone", "a totem stone", GearSlot::Amulet, 2, 1, 0)
        .with_armor(6.0)
        .with_thorns(6.0),
    item("eagle_eye", "an eagle-eye stone", GearSlot::Amulet, 2, 1, 0).with_crit(8.0),
    item("vampire_fang", "a vampire's fang", GearSlot::Amulet, 3, 2, 2)
        .with_lifesteal(15.0),
    item("holy_symbol", "a holy symbol", GearSlot::Amulet, 3, 1, 2).with_regen(0.5),
    item("moon_pendant", "a moon pendant", GearSlot::Amulet, 3, 2, 2).with_hp(20.0),
    item("demon_heart", "a demon's heart", GearSlot::Amulet, 4, 3, 3)
        .with_dmg(14.0)
        .with_lifesteal(10.0),
    item("phoenix_feather", "a phoenix feather", GearSlot::Amulet, 4, 3, 3)
        .with_regen(1.0),
    item("doom_amulet", "an amulet of doom", GearSlot::Amulet, 5, 4, 3)
        .with_dmg(16.0)
        .with_hp(16.0),
    item(
        "god_shard",
        "a shard of a god",
        GearSlot::Amulet,
        6,
        4,
        3,
    )
    .with_dmg(12.0)
    .with_hp(24.0)
    .with_regen(0.8)
    .with_crit(10.0)
    .with_lifesteal(10.0),
];

impl ItemDef {
    const fn with_dmg(mut self, v: f32) -> Self {
        self.dmg = v;
        self
    }
    const fn with_hp(mut self, v: f32) -> Self {
        self.hp = v;
        self
    }
    const fn with_regen(mut self, v: f32) -> Self {
        self.regen = v;
        self
    }
    const fn with_armor(mut self, v: f32) -> Self {
        self.armor = v;
        self
    }
    const fn with_crit(mut self, v: f32) -> Self {
        self.crit = v;
        self
    }
    const fn with_lifesteal(mut self, v: f32) -> Self {
        self.lifesteal = v;
        self
    }
    const fn with_thorns(mut self, v: f32) -> Self {
        self.thorns = v;
        self
    }
    const fn as_ranged(mut self) -> Self {
        self.ranged = true;
        self
    }
}

/// Look an item up by its save key.
pub fn item_def(key: &str) -> Option<&'static ItemDef> {
    ITEM_TABLE.iter().find(|d| d.key == key)
}

/// pwb's rarity weights: common 46, fine 26, rare 15, epic 9, legendary 4.
pub const RARITY_WEIGHTS: [f32; 5] = [46.0, 26.0, 15.0, 9.0, 4.0];

/// The armament tier an era supports, on the same ladder as the item
/// tiers: 0 for Pre-Stone and Stone, 1 Bronze/Iron, 2 Classical, 3
/// Medieval/Renaissance, 4 Industrial, 5 Modern, 6 Information and beyond.
pub fn era_tier(era: u8) -> u8 {
    match era {
        0 | 1 => 0,
        2 | 3 => 1,
        4 => 2,
        5 | 6 => 3,
        7 => 4,
        8 => 5,
        _ => 6,
    }
}

/// Roll a drop for a lineage of tier `tier`: rarity-weighted, uniform within
/// the rarity, and restricted to patterns the tier can field. Returns the
/// chosen pattern, or `None` when nothing of the tier fits the roll.
pub fn roll_item<R: rand::RngExt>(rng: &mut R, tier: u8) -> Option<&'static ItemDef> {
    let total: f32 = RARITY_WEIGHTS.iter().sum();
    let pick = rng.random::<f32>() * total;
    let mut acc = 0.0;
    let mut rarity = 0usize;
    for (i, w) in RARITY_WEIGHTS.iter().enumerate() {
        acc += w;
        if pick < acc {
            rarity = i;
            break;
        }
    }
    // Within the rolled rarity, prefer patterns the tier can actually
    // field; if the tier's whole basket of that rarity is empty, fall
    // back to any pattern of the tier (the hunt still yields something).
    let pool: Vec<&ItemDef> = ITEM_TABLE
        .iter()
        .filter(|d| d.tier <= tier && d.rarity as usize == rarity)
        .collect();
    if pool.is_empty() {
        let fallback: Vec<&ItemDef> = ITEM_TABLE.iter().filter(|d| d.tier <= tier).collect();
        if fallback.is_empty() {
            return None;
        }
        return Some(fallback[rng.random_range(0..fallback.len())]);
    }
    Some(pool[rng.random_range(0..pool.len())])
}

/// The aggregate of everything a person wears, in the scales the combat
/// rolls use.
#[derive(Clone, Copy, Debug, Default)]
pub struct GearStats {
    /// Outgoing damage added per blow (pwb dmg * `DMG_SCALE`).
    pub dmg: f32,
    /// Percentage mitigation on incoming blows (armor + toughness), capped.
    pub mitigation: f32,
    /// Percentage chance of a 2.2x critical strike.
    pub crit: f32,
    /// Percentage of damage dealt, healed back.
    pub lifesteal: f32,
    /// Counter-wound on an attacker that lands a blow.
    pub thorns: f32,
    /// Health recovered extra each tick.
    pub regen: f32,
    /// The wearer strikes from beyond a melee counter's reach.
    pub ranged: bool,
    /// Highest rank worn (for looting comparisons).
    pub best_rank: u8,
}

/// pwb dmg numbers (2..34) against THB's ~0.06 baseline per blow.
pub const DMG_SCALE: f32 = 0.004;
/// pwb hp numbers (5..75) folded into toughness mitigation, percent.
pub const HP_TO_MITIGATION: f32 = 0.4;
/// Total mitigation cap over all pieces (pwb's 60% armour cap).
pub const MITIGATION_CAP: f32 = 0.60;
/// Extra health per tick from regen stats.
pub const REGEN_SCALE: f32 = 0.0002;
/// Counter-wound per thorns point.
pub const THORNS_SCALE: f32 = 0.004;
/// Critical strikes deal this multiple.
pub const CRIT_MULT: f32 = 2.2;

impl GearStats {
    /// Sum a person's gear. `gear` maps slot key ("wep" .. "amu") to the
    /// item key; unknown keys (a future table change) count for nothing.
    pub fn of(gear: &BTreeMap<String, String>) -> GearStats {
        let mut s = GearStats::default();
        for key in gear.values() {
            let Some(d) = item_def(key) else { continue };
            s.dmg += d.dmg * DMG_SCALE;
            s.mitigation += (d.armor + d.hp * HP_TO_MITIGATION) / 100.0;
            s.crit += d.crit / 100.0;
            s.lifesteal += d.lifesteal / 100.0;
            s.thorns += d.thorns * THORNS_SCALE;
            s.regen += d.regen * REGEN_SCALE;
            s.ranged |= d.ranged;
            s.best_rank = s.best_rank.max(d.rank);
        }
        s.mitigation = s.mitigation.min(MITIGATION_CAP);
        s
    }
}

/// Equip an item, pwb's rank rule: a pickup replaces the slot's current
/// item only when it is the stronger pattern. Returns the newly worn key
/// when the item went on, `None` when the current gear was kept.
pub fn equip(gear: &mut BTreeMap<String, String>, def: &ItemDef) -> Option<&'static str> {
    let slot_key = def.slot.key();
    let keep_current = gear
        .get(slot_key)
        .and_then(|cur| item_def(cur))
        .is_some_and(|cur| cur.rank >= def.rank);
    if keep_current {
        return None;
    }
    gear.insert(slot_key.to_string(), def.key.to_string());
    Some(def.key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn the_table_spans_all_seven_slots_and_five_rarities() {
        assert!(ITEM_TABLE.len() >= 100, "the hundred-item table survived the port");
        for slot in GearSlot::ALL {
            assert!(
                ITEM_TABLE.iter().any(|d| d.slot == slot),
                "slot {:?} has patterns",
                slot
            );
        }
        assert!(
            ITEM_TABLE.iter().filter(|d| d.rarity == 4).count() >= 10,
            "legendary patterns survive the port"
        );
        // Keys are unique (they are save identities).
        let mut keys: Vec<&str> = ITEM_TABLE.iter().map(|d| d.key).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), ITEM_TABLE.len(), "item keys are unique");
    }

    #[test]
    fn gear_aggregates_and_the_rank_rule_ratchets_up() {
        let mut gear = BTreeMap::new();
        assert!(equip(&mut gear, item_def("sword1").unwrap()).is_some());
        // A weaker club never displaces the shortsword.
        assert!(equip(&mut gear, item_def("wood_club").unwrap()).is_none());
        assert_eq!(gear.get("wep").map(String::as_str), Some("sword1"));
        // The holy sword ratchets up.
        assert!(equip(&mut gear, item_def("sword3").unwrap()).is_some());
        // Stack a shield and a ring, then read the aggregate.
        equip(&mut gear, item_def("iron_shield").unwrap());
        equip(&mut gear, item_def("vampire_fang").unwrap());
        let s = GearStats::of(&gear);
        assert!((s.dmg - (18.0 * DMG_SCALE)).abs() < 1e-6, "dmg: {}", s.dmg);
        assert!((s.mitigation - (12.0 + 10.0 * HP_TO_MITIGATION) / 100.0).abs() < 1e-6);
        assert_eq!(s.best_rank, 3);
        assert!(!s.ranged, "no ranged piece worn");
        assert!((s.lifesteal - 0.15).abs() < 1e-6, "lifesteal: {}", s.lifesteal);
    }

    #[test]
    fn mitigation_caps_at_sixty_percent() {
        let mut gear = BTreeMap::new();
        for k in ["dragon_shield", "mithril_shield", "dragon_scale", "obsidian_shell"] {
            equip(&mut gear, item_def(k).unwrap());
        }
        // Only one piece per slot counts: dragon_shield (armour=26+hp20)
        // wins the shield slot, dragon_scale the body, obsidian is weaker
        // in rank than dragon_scale so it never goes on.
        let s = GearStats::of(&gear);
        assert!(s.mitigation <= MITIGATION_CAP, "capped at 60%: {}", s.mitigation);
        assert_eq!(gear.len(), 2, "one piece per slot: {:?}", gear);
    }

    #[test]
    fn rolls_respect_the_tier_and_the_rarity_weights() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        // A stone-age tier can never roll a firearm or future arm.
        for _ in 0..500 {
            if let Some(d) = roll_item(&mut rng, 0) {
                assert!(d.tier <= 0, "tier-0 roll leaked {} (tier {})", d.key, d.tier);
            }
        }
        // An Information-tier lineage can roll everything.
        let mut seen_rarity = [false; 5];
        for _ in 0..2000 {
            if let Some(d) = roll_item(&mut rng, 6) {
                seen_rarity[d.rarity as usize] = true;
            }
        }
        assert!(seen_rarity.iter().all(|&s| s), "every rarity is reachable");
    }
}
