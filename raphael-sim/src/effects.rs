use crate::{Combo, Settings};

#[bitfield_struct::bitfield(u64, default = false)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Effects {
    #[bits(4)]
    pub inner_quiet: u8,
    #[bits(4)]
    pub waste_not: u8,
    #[bits(3)]
    pub innovation: u8,
    #[bits(3)]
    pub veneration: u8,
    #[bits(3)]
    pub great_strides: u8,
    #[bits(3)]
    pub muscle_memory: u8,
    #[bits(4)]
    pub manipulation: u8,
    #[bits(3)]
    pub final_appraisal: u8,
    #[bits(2)]
    pub careful_observation_charges: u8,

    pub trained_perfection_available: bool,
    pub heart_and_soul_available: bool,
    pub quick_innovation_available: bool,
    pub trained_perfection_active: bool,
    pub heart_and_soul_active: bool,

    #[bits(2)]
    /// This effect does not exist in-game and is only used by the solver.
    pub special_quality_state: SpecialQualityState,

    #[bits(2)]
    pub combo: Combo,

    // 32-bit boundary
    /// Remaining usages of the Stellar Steady Hand action.
    pub stellar_steady_hand_charges: u8,
    #[bits(2)]
    /// Remaining duration of the effect from the Stellar Steady Hand action.
    pub stellar_steady_hand: u8,

    /// Combo effect from Hasty Touch that enables usage of Daring Touch.
    pub expedience: bool,

    #[bits(3)]
    /// Remaining Crafter's Delineations shared by specialist actions.
    pub crafter_delineations: u8,

    /// Cosmic/Splendorous tools increase Good-condition quality to 175%.
    pub splendor_cosmic: bool,

    #[bits(11)]
    pub _padding: u32,
}

impl Effects {
    /// Canonical representation of the shared Crafter's Delineation resource.
    ///
    /// Heart and Soul and Quick Innovation are one-shot actions sharing the same
    /// consumable.  Keeping impossible availability/resource combinations out of
    /// solver keys is critical: otherwise equivalent live roots miss precomputed
    /// tables and trigger expensive dynamic solving.
    #[must_use]
    pub const fn canonicalize_specialist_resources(self) -> Self {
        let usable_actions = self.heart_and_soul_available() as u8
            + self.quick_innovation_available() as u8
            + self.careful_observation_charges();
        let delineations = if self.crafter_delineations() < usable_actions {
            self.crafter_delineations()
        } else {
            usable_actions
        };
        let effects = self
            .with_crafter_delineations(delineations)
            .with_careful_observation_charges(if self.careful_observation_charges() < delineations {
                self.careful_observation_charges()
            } else {
                delineations
            });
        if delineations == 0 {
            // Heart and Soul may already be active after its delineation was spent.
            effects
                .with_heart_and_soul_available(false)
                .with_quick_innovation_available(false)
        } else {
            effects
        }
    }

    /// Effects at synthesis begin
    pub fn initial(settings: &Settings) -> Self {
        let special_quality_state = match settings.adversarial {
            true => SpecialQualityState::AdversarialGuard2,
            false => SpecialQualityState::Normal,
        };
        Self::new()
            .with_special_quality_state(special_quality_state)
            .with_trained_perfection_available(
                settings.is_action_allowed::<crate::actions::TrainedPerfection>(),
            )
            .with_heart_and_soul_available(
                settings.is_action_allowed::<crate::actions::HeartAndSoul>(),
            )
            .with_quick_innovation_available(
                settings.is_action_allowed::<crate::actions::QuickInnovation>(),
            )
            .with_crafter_delineations(
                u8::from(settings.is_action_allowed::<crate::actions::HeartAndSoul>())
                    + u8::from(settings.is_action_allowed::<crate::actions::QuickInnovation>()),
            )
            .with_combo(Combo::SynthesisBegin)
            .with_stellar_steady_hand_charges(settings.stellar_steady_hand_charges)
    }

    pub(crate) const fn progress_modifier(self) -> u32 {
        let mm_mod = if self.muscle_memory() != 0 { 10 } else { 0 };
        let vene_mod = if self.veneration() != 0 { 5 } else { 0 };
        10 + mm_mod + vene_mod
    }

    pub(crate) const fn quality_modifier(self) -> u32 {
        let gs_mod = if self.great_strides() != 0 { 10 } else { 0 };
        let inno_mod = if self.innovation() != 0 { 5 } else { 0 };
        (self.inner_quiet() as u32 + 10) * (10 + gs_mod + inno_mod)
    }

    pub const fn adversarial_guard_active(self) -> bool {
        matches!(
            self.special_quality_state(),
            SpecialQualityState::AdversarialGuard | SpecialQualityState::AdversarialGuard2
        )
    }

    pub const fn quality_actions_allowed(self) -> bool {
        !matches!(self.special_quality_state(), SpecialQualityState::Forbidden)
    }

    #[must_use]
    pub const fn tick_down(self) -> Self {
        // Calculate the decrement bits for all ticking effects.
        // The decrement contains the least-significant bit of all active ticking effects.
        let effects_tick = {
            let mask_0 = self.into_bits() & EFFECTS_BIT_0;
            let mask_1 = (self.into_bits() & EFFECTS_BIT_1) >> 1;
            let mask_2 = (self.into_bits() & EFFECTS_BIT_2) >> 2;
            let mask_3 = (self.into_bits() & EFFECTS_BIT_3) >> 3;
            mask_0 | mask_1 | mask_2 | mask_3
        };
        Self::from_bits(self.into_bits() - effects_tick)
    }

    /// Removes all effects that are only relevant for Quality.
    #[must_use]
    pub const fn strip_quality_effects(self) -> Self {
        self.with_special_quality_state(SpecialQualityState::Forbidden)
            .with_inner_quiet(0)
            .with_innovation(0)
            .with_great_strides(0)
            .with_expedience(false)
            .with_splendor_cosmic(false)
            .with_quick_innovation_available(false)
            .with_final_appraisal(0)
            .with_careful_observation_charges(0)
            .canonicalize_specialist_resources()
    }
}

const EFFECTS_BIT_0: u64 = Effects::new()
    .with_waste_not(1)
    .with_innovation(1)
    .with_veneration(1)
    .with_great_strides(1)
    .with_muscle_memory(1)
    .with_manipulation(1)
    .with_final_appraisal(1)
    .with_stellar_steady_hand(1)
    .with_expedience(true)
    .into_bits();

const EFFECTS_BIT_1: u64 = Effects::new()
    .with_special_quality_state(SpecialQualityState::AdversarialGuard)
    .with_waste_not(2)
    .with_innovation(2)
    .with_veneration(2)
    .with_great_strides(2)
    .with_muscle_memory(2)
    .with_manipulation(2)
    .with_stellar_steady_hand(2)
    .into_bits();

const EFFECTS_BIT_2: u64 = Effects::new()
    .with_waste_not(4)
    .with_innovation(4)
    .with_veneration(4)
    .with_muscle_memory(4)
    .with_manipulation(4)
    .into_bits();

const EFFECTS_BIT_3: u64 = Effects::new()
    .with_waste_not(8)
    .with_manipulation(8)
    .into_bits();

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpecialQualityState {
    Forbidden,         // Quality-increasing actions are forbidden
    Normal,            // Default mode of operation
    AdversarialGuard,  // Guarded from adversarial mode (expires in 1 turn)
    AdversarialGuard2, // Guard from adversarial mode (expires in 2 turns)
}

impl SpecialQualityState {
    pub const fn into_bits(self) -> u8 {
        match self {
            Self::Forbidden => 0,
            Self::Normal => 1,
            Self::AdversarialGuard => 2,
            Self::AdversarialGuard2 => 3,
        }
    }

    pub const fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::Forbidden,
            1 => Self::Normal,
            2 => Self::AdversarialGuard,
            _ => Self::AdversarialGuard2,
        }
    }
}

#[cfg(test)]
mod specialist_resource_tests {
    use super::*;

    #[test]
    fn canonicalization_clamps_and_preserves_active_heart_and_soul() {
        let effects = Effects::new()
            .with_crafter_delineations(7)
            .with_heart_and_soul_available(true)
            .with_quick_innovation_available(false)
            .canonicalize_specialist_resources();
        assert_eq!(effects.crafter_delineations(), 1);

        let effects = effects
            .with_crafter_delineations(0)
            .with_quick_innovation_available(true)
            .with_heart_and_soul_active(true)
            .canonicalize_specialist_resources();
        assert_eq!(effects.crafter_delineations(), 0);
        assert!(!effects.heart_and_soul_available());
        assert!(!effects.quick_innovation_available());
        assert!(effects.heart_and_soul_active());
    }

    #[test]
    fn quality_stripping_removes_cosmic_good_condition_bonus() {
        let effects = Effects::new().with_splendor_cosmic(true);
        assert!(!effects.strip_quality_effects().splendor_cosmic());
    }
}
