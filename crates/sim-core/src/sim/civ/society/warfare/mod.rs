use crate::hashing::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde::{Deserialize, Serialize};

use super::government::{Government, LawKind};
use crate::sim::era::Era;
use crate::sim::tech::buildings::{Building, BuildingKind};
use crate::sim::world_events::push_event;
use crate::world::grid::WorldGrid;

mod armory;
mod battles;
mod march;
pub(crate) mod neuro;
mod spawning;
#[cfg(test)]
mod tests;
mod treaties;
mod types;
mod wars;
#[cfg(test)]
mod wars_tests;

pub use armory::{EmergentWeapon, WeaponDefense, WeaponDoctrine};
pub use battles::*;
pub use spawning::*;
pub use treaties::*;
pub use types::*;
pub use wars::*;
