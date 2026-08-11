use strum::EnumCount;

use raphael_sim::*;

use crate::SolverSettings;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionCombo {
    TricksOfTheTrade,   // Heart and Soul + Tricks of the Trade
    IntensiveSynthesis, // Heart and Soul + Intensive Synthesis
    PreciseTouch,       // Heart and Soul + Precise Touch
    StandardTouch,      // Basic Touch + Standard Touch
    AdvancedTouch,      // Basic Touch + Standard Touch + Advanced Touch
    FocusedTouch,       // Observe + AdvancedTouch
    RefinedTouch,       // Basic Touch + Refined Touch
    None,               // No action
    Single(Action),
}

impl ActionCombo {
    pub const fn into_bits(self) -> u8 {
        const LEGACY_ACTION_COUNT: u8 = Action::COUNT as u8 - 1;
        match self {
            Self::Single(Action::FinalAppraisal) => LEGACY_ACTION_COUNT + 8,
            Self::Single(action) => action.into_bits(),
            Self::TricksOfTheTrade => LEGACY_ACTION_COUNT,
            Self::IntensiveSynthesis => LEGACY_ACTION_COUNT + 1,
            Self::PreciseTouch => LEGACY_ACTION_COUNT + 2,
            Self::StandardTouch => LEGACY_ACTION_COUNT + 3,
            Self::AdvancedTouch => LEGACY_ACTION_COUNT + 4,
            Self::FocusedTouch => LEGACY_ACTION_COUNT + 5,
            Self::RefinedTouch => LEGACY_ACTION_COUNT + 6,
            Self::None => LEGACY_ACTION_COUNT + 7,
        }
    }

    pub const fn from_bits(bits: u8) -> Self {
        const N: u8 = Action::COUNT as u8 - 1;
        if bits < N {
            Self::Single(Action::from_bits(bits))
        } else if bits == N {
            Self::TricksOfTheTrade
        } else if bits == N + 1 {
            Self::IntensiveSynthesis
        } else if bits == N + 2 {
            Self::PreciseTouch
        } else if bits == N + 3 {
            Self::StandardTouch
        } else if bits == N + 4 {
            Self::AdvancedTouch
        } else if bits == N + 5 {
            Self::FocusedTouch
        } else if bits == N + 6 {
            Self::RefinedTouch
        } else if bits == N + 8 {
            Self::Single(Action::FinalAppraisal)
        } else {
            Self::None
        }
    }

    pub const fn actions(self) -> &'static [Action] {
        match self {
            Self::TricksOfTheTrade => &[Action::HeartAndSoul, Action::TricksOfTheTrade],
            Self::IntensiveSynthesis => &[Action::HeartAndSoul, Action::IntensiveSynthesis],
            Self::PreciseTouch => &[Action::HeartAndSoul, Action::PreciseTouch],
            Self::StandardTouch => &[Action::BasicTouch, Action::StandardTouch],
            Self::AdvancedTouch => &[
                Action::BasicTouch,
                Action::StandardTouch,
                Action::AdvancedTouch,
            ],
            Self::FocusedTouch => &[Action::Observe, Action::AdvancedTouch],
            Self::RefinedTouch => &[Action::BasicTouch, Action::RefinedTouch],
            Self::None => &[],
            Self::Single(action) => match action {
                Action::BasicSynthesis => &[Action::BasicSynthesis],
                Action::BasicTouch => &[Action::BasicTouch],
                Action::MasterMend => &[Action::MasterMend],
                Action::Observe => &[Action::Observe],
                Action::TricksOfTheTrade => &[Action::TricksOfTheTrade],
                Action::WasteNot => &[Action::WasteNot],
                Action::Veneration => &[Action::Veneration],
                Action::StandardTouch => &[Action::StandardTouch],
                Action::GreatStrides => &[Action::GreatStrides],
                Action::Innovation => &[Action::Innovation],
                Action::WasteNot2 => &[Action::WasteNot2],
                Action::ByregotsBlessing => &[Action::ByregotsBlessing],
                Action::PreciseTouch => &[Action::PreciseTouch],
                Action::MuscleMemory => &[Action::MuscleMemory],
                Action::CarefulSynthesis => &[Action::CarefulSynthesis],
                Action::Manipulation => &[Action::Manipulation],
                Action::PrudentTouch => &[Action::PrudentTouch],
                Action::AdvancedTouch => &[Action::AdvancedTouch],
                Action::Reflect => &[Action::Reflect],
                Action::PreparatoryTouch => &[Action::PreparatoryTouch],
                Action::Groundwork => &[Action::Groundwork],
                Action::DelicateSynthesis => &[Action::DelicateSynthesis],
                Action::IntensiveSynthesis => &[Action::IntensiveSynthesis],
                Action::TrainedEye => &[Action::TrainedEye],
                Action::HeartAndSoul => &[Action::HeartAndSoul],
                Action::PrudentSynthesis => &[Action::PrudentSynthesis],
                Action::TrainedFinesse => &[Action::TrainedFinesse],
                Action::RefinedTouch => &[Action::RefinedTouch],
                Action::QuickInnovation => &[Action::QuickInnovation],
                Action::ImmaculateMend => &[Action::ImmaculateMend],
                Action::TrainedPerfection => &[Action::TrainedPerfection],
                Action::StellarSteadyHand => &[Action::StellarSteadyHand],
                Action::RapidSynthesis => &[Action::RapidSynthesis],
                Action::HastyTouch => &[Action::HastyTouch],
                Action::DaringTouch => &[Action::DaringTouch],
                Action::FinalAppraisal => &[Action::FinalAppraisal],
            },
        }
    }

    pub const fn steps(self) -> u8 {
        self.actions().len() as u8
    }

    pub fn duration(self) -> u8 {
        self.actions().iter().map(|action| action.time_cost()).sum()
    }
}

pub const FULL_SEARCH_ACTIONS: [ActionCombo; 37] = [
    ActionCombo::AdvancedTouch,
    ActionCombo::TricksOfTheTrade,
    ActionCombo::IntensiveSynthesis,
    ActionCombo::PreciseTouch,
    ActionCombo::StandardTouch,
    ActionCombo::FocusedTouch,
    ActionCombo::RefinedTouch,
    // progress
    ActionCombo::Single(Action::BasicSynthesis),
    ActionCombo::Single(Action::Veneration),
    ActionCombo::Single(Action::MuscleMemory),
    ActionCombo::Single(Action::CarefulSynthesis),
    ActionCombo::Single(Action::Groundwork),
    ActionCombo::Single(Action::PrudentSynthesis),
    ActionCombo::Single(Action::RapidSynthesis),
    // quality
    ActionCombo::Single(Action::BasicTouch),
    ActionCombo::Single(Action::StandardTouch),
    ActionCombo::Single(Action::GreatStrides),
    ActionCombo::Single(Action::Innovation),
    ActionCombo::Single(Action::ByregotsBlessing),
    ActionCombo::Single(Action::PrudentTouch),
    ActionCombo::Single(Action::Reflect),
    ActionCombo::Single(Action::PreparatoryTouch),
    ActionCombo::Single(Action::AdvancedTouch),
    ActionCombo::Single(Action::TrainedFinesse),
    ActionCombo::Single(Action::TrainedEye),
    ActionCombo::Single(Action::QuickInnovation),
    ActionCombo::Single(Action::HastyTouch),
    ActionCombo::Single(Action::DaringTouch),
    // durability
    ActionCombo::Single(Action::MasterMend),
    ActionCombo::Single(Action::WasteNot),
    ActionCombo::Single(Action::WasteNot2),
    ActionCombo::Single(Action::Manipulation),
    ActionCombo::Single(Action::ImmaculateMend),
    ActionCombo::Single(Action::TrainedPerfection),
    // misc
    ActionCombo::Single(Action::DelicateSynthesis),
    ActionCombo::Single(Action::StellarSteadyHand),
    ActionCombo::Single(Action::FinalAppraisal),
];

pub const PROGRESS_ONLY_SEARCH_ACTIONS: [ActionCombo; 16] = [
    ActionCombo::IntensiveSynthesis,
    ActionCombo::TricksOfTheTrade,
    // progress
    ActionCombo::Single(Action::BasicSynthesis),
    ActionCombo::Single(Action::Veneration),
    ActionCombo::Single(Action::MuscleMemory),
    ActionCombo::Single(Action::CarefulSynthesis),
    ActionCombo::Single(Action::Groundwork),
    ActionCombo::Single(Action::PrudentSynthesis),
    ActionCombo::Single(Action::RapidSynthesis),
    // durability
    ActionCombo::Single(Action::MasterMend),
    ActionCombo::Single(Action::WasteNot),
    ActionCombo::Single(Action::WasteNot2),
    ActionCombo::Single(Action::Manipulation),
    ActionCombo::Single(Action::ImmaculateMend),
    ActionCombo::Single(Action::TrainedPerfection),
    // misc
    ActionCombo::Single(Action::StellarSteadyHand),
];

pub fn use_action_combo(
    settings: &SolverSettings,
    mut state: SimulationState,
    action_combo: ActionCombo,
) -> Result<SimulationState, ActionError> {
    for action in action_combo.actions() {
        state = state.use_action(*action, Condition::Normal, &settings.simulator_settings)?;
    }
    // All combos are already implemented as ActionCombo, so we reset the combo to None for cases
    // where the full combo is not used, reducing state space.
    if state.effects.combo() != Combo::SynthesisBegin {
        state.effects.set_combo(Combo::None);
    }
    // Expedience enables Daring Touch, but Daring Touch can only be used if Stellar Steady Hand is active.
    if state.effects.stellar_steady_hand() == 0 {
        state.effects.set_expedience(false);
    }
    Ok(state)
}

/// Execute actions using an already-observed deterministic condition prefix. Unlike the Normal
/// solver convenience, this preserves the real combo state after an individual prefix action.
pub fn use_action_combo_with_condition(
    settings: &SolverSettings,
    mut state: SimulationState,
    action_combo: ActionCombo,
    mut condition: Condition,
) -> Result<(SimulationState, Condition), ActionError> {
    for action in action_combo.actions() {
        state = state.use_action(*action, condition, &settings.simulator_settings)?;
        if action.increases_step_count() {
            condition = condition.deterministic_successor();
        }
    }
    Ok((state, condition))
}

#[cfg(test)]
mod condition_prefix_tests {
    use super::*;

    const SETTINGS: Settings = Settings {
        max_cp: 500,
        max_durability: 80,
        max_progress: 5000,
        max_quality: 50000,
        base_progress: 100,
        base_quality: 100,
        job_level: 100,
        allowed_actions: ActionMask::all(),
        adversarial: false,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };

    fn solver_settings() -> SolverSettings {
        SolverSettings {
            simulator_settings: SETTINGS,
            allow_non_max_quality_solutions: true,
        }
    }

    #[test]
    fn zero_step_preserves_excellent_then_actions_advance_to_poor_and_normal() {
        let settings = solver_settings();
        let root = SimulationState {
            effects: Effects::initial(&SETTINGS).with_heart_and_soul_available(true),
            ..SimulationState::new(&SETTINGS)
        };
        let (state, condition) = use_action_combo_with_condition(
            &settings,
            root,
            ActionCombo::Single(Action::HeartAndSoul),
            Condition::Excellent,
        )
        .unwrap();
        assert_eq!(condition, Condition::Excellent);
        let (state, condition) = use_action_combo_with_condition(
            &settings,
            state,
            ActionCombo::Single(Action::BasicTouch),
            condition,
        )
        .unwrap();
        assert_eq!(condition, Condition::Poor);
        assert_eq!(state.effects.combo(), Combo::BasicTouch);
        let cp = state.cp;
        let (state, condition) = use_action_combo_with_condition(
            &settings,
            state,
            ActionCombo::Single(Action::StandardTouch),
            condition,
        )
        .unwrap();
        assert_eq!(condition, Condition::Normal);
        assert_eq!(
            cp - state.cp,
            18,
            "live combo discount must survive prefix replay"
        );
    }

    #[test]
    fn good_omen_advances_to_good_then_normal() {
        let settings = solver_settings();
        let root = SimulationState::new(&SETTINGS);
        let (state, condition) = use_action_combo_with_condition(
            &settings,
            root,
            ActionCombo::Single(Action::BasicSynthesis),
            Condition::GoodOmen,
        )
        .unwrap();
        assert_eq!(condition, Condition::Good);
        let (_, condition) = use_action_combo_with_condition(
            &settings,
            state,
            ActionCombo::Single(Action::BasicSynthesis),
            condition,
        )
        .unwrap();
        assert_eq!(condition, Condition::Normal);
    }

    #[test]
    fn final_appraisal_combo_encoding_round_trips_without_shifting_legacy_codes() {
        let final_appraisal = ActionCombo::Single(Action::FinalAppraisal);
        assert_eq!(
            ActionCombo::from_bits(final_appraisal.into_bits()),
            final_appraisal
        );
        assert_eq!(
            ActionCombo::TricksOfTheTrade.into_bits(),
            Action::COUNT as u8 - 1
        );
    }
}
