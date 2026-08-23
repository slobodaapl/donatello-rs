use raphael_sim::*;
use smallvec::SmallVec;

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
        const LEGACY_ACTION_COUNT: u8 = Action::FinalAppraisal as u8;
        match self {
            Self::Single(Action::FinalAppraisal) => LEGACY_ACTION_COUNT + 8,
            Self::Single(Action::CarefulObservation) => LEGACY_ACTION_COUNT + 9,
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
        const N: u8 = Action::FinalAppraisal as u8;
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
        } else if bits == N + 9 {
            Self::Single(Action::CarefulObservation)
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
                Action::CarefulObservation => &[Action::CarefulObservation],
            },
        }
    }

    pub const fn steps(self) -> u8 {
        self.actions().len() as u8
    }

    pub fn duration(self) -> u8 {
        self.actions().iter().map(|action| action.time_cost()).sum()
    }

    pub const fn is_stellar_window_action(self) -> bool {
        matches!(
            self,
            Self::Single(
                Action::StellarSteadyHand
                    | Action::RapidSynthesis
                    | Action::HastyTouch
                    | Action::DaringTouch
            )
        )
    }
}

pub const FULL_SEARCH_ACTIONS: [ActionCombo; 38] = [
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
    ActionCombo::Single(Action::CarefulObservation),
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
        if action.advances_condition() {
            condition = condition.deterministic_successor();
        }
    }
    Ok((state, condition))
}

pub(crate) fn final_appraisal_is_dominated(
    settings: &SolverSettings,
    state: SimulationState,
    condition: Condition,
) -> bool {
    condition == Condition::Normal
        && state
            .use_action(
                Action::BasicSynthesis,
                condition,
                &settings.simulator_settings,
            )
            .is_ok_and(|next| next.progress >= settings.max_progress())
}

#[derive(Debug, Clone)]
pub struct StellarWindowTransition {
    pub state: SimulationState,
    pub condition: Condition,
    pub actions: SmallVec<[ActionCombo; 4]>,
    pub advancing_steps: u16,
}

/// Collapse one Stellar Steady Hand window into its bounded progress/quality policy frontier.
/// Returned transitions retain concrete actions, exact simulator state, and observed-condition
/// successors; callers may therefore queue them as one search edge without making execution atomic.
pub fn stellar_window_transitions(
    settings: &SolverSettings,
    mut state: SimulationState,
    mut condition: Condition,
    stop_progress: Option<u16>,
) -> SmallVec<[StellarWindowTransition; 8]> {
    let mut actions = SmallVec::<[ActionCombo; 4]>::new();
    if state.effects.stellar_steady_hand() == 0 {
        let activation = ActionCombo::Single(Action::StellarSteadyHand);
        let Ok((activated, next_condition)) =
            use_action_combo_with_condition(settings, state, activation, condition)
        else {
            return SmallVec::new();
        };
        state = activated;
        condition = next_condition;
        actions.push(activation);
    }

    fn expand(
        settings: &SolverSettings,
        state: SimulationState,
        condition: Condition,
        stop_progress: Option<u16>,
        actions: &SmallVec<[ActionCombo; 4]>,
        advancing_steps: u16,
        result: &mut SmallVec<[StellarWindowTransition; 8]>,
    ) {
        if state.is_final(&settings.simulator_settings)
            || state.effects.stellar_steady_hand() == 0
            || stop_progress == Some(state.progress)
        {
            let mut state = state;
            if state.effects.stellar_steady_hand() == 0 {
                state.effects.set_expedience(false);
            }
            result.push(StellarWindowTransition {
                state,
                condition,
                actions: actions.clone(),
                advancing_steps,
            });
            return;
        }

        let quality_action = if state.effects.expedience() {
            Action::DaringTouch
        } else {
            Action::HastyTouch
        };
        for action in [Action::RapidSynthesis, quality_action] {
            let action = ActionCombo::Single(action);
            let Ok((child, child_condition)) =
                use_action_combo_with_condition(settings, state, action, condition)
            else {
                continue;
            };
            let child_advancing_steps =
                advancing_steps.saturating_add(u16::from(child.progress > state.progress));
            let mut child_actions = actions.clone();
            child_actions.push(action);
            expand(
                settings,
                child,
                child_condition,
                stop_progress,
                &child_actions,
                child_advancing_steps,
                result,
            );
        }
    }

    let mut result = SmallVec::new();
    expand(
        settings,
        state,
        condition,
        stop_progress,
        &actions,
        0,
        &mut result,
    );
    result
}

pub fn has_stellar_window_resource(state: &SimulationState) -> bool {
    state.effects.stellar_steady_hand_charges() != 0 || state.effects.stellar_steady_hand() != 0
}

pub fn remove_stellar_window_from_bound_settings(settings: &mut Settings) {
    settings.stellar_steady_hand_charges = 0;
    settings.allowed_actions = settings
        .allowed_actions
        .remove(Action::StellarSteadyHand)
        .remove(Action::RapidSynthesis)
        .remove(Action::HastyTouch)
        .remove(Action::DaringTouch);
}

#[cfg(test)]
mod stellar_window_tests {
    use super::*;

    fn settings(charges: u8) -> SolverSettings {
        SolverSettings {
            simulator_settings: Settings {
                max_cp: 500,
                max_durability: 80,
                max_progress: 5000,
                max_quality: 50_000,
                base_progress: 100,
                base_quality: 100,
                job_level: 100,
                allowed_actions: ActionMask::all(),
                adversarial: false,
                backload_progress: false,
                stellar_steady_hand_charges: charges,
            },
            allow_non_max_quality_solutions: true,
        }
    }

    #[test]
    fn inactive_window_is_eight_exact_progress_quality_branches() {
        let settings = settings(1);
        let root = SimulationState::new(&settings.simulator_settings);
        let transitions = stellar_window_transitions(&settings, root, Condition::Normal, None);
        assert_eq!(transitions.len(), 8);
        assert!(transitions.iter().all(|transition| {
            transition.actions.len() == 4
                && transition.actions[0] == ActionCombo::Single(Action::StellarSteadyHand)
                && transition.state.effects.stellar_steady_hand() == 0
                && !transition.state.effects.expedience()
        }));
        assert!(transitions.iter().any(|transition| {
            transition.actions.as_slice()
                == [
                    ActionCombo::Single(Action::StellarSteadyHand),
                    ActionCombo::Single(Action::HastyTouch),
                    ActionCombo::Single(Action::DaringTouch),
                    ActionCombo::Single(Action::HastyTouch),
                ]
        }));
    }

    #[test]
    fn active_window_replans_only_the_remaining_actions() {
        let settings = settings(0);
        let mut root = SimulationState::new(&settings.simulator_settings);
        root.effects.set_stellar_steady_hand(2);
        root.effects.set_expedience(true);
        root.effects.set_combo(Combo::BasicTouch);
        let transitions = stellar_window_transitions(&settings, root, Condition::Excellent, None);
        assert_eq!(transitions.len(), 4);
        assert!(transitions.iter().all(|transition| {
            transition.actions.len() == 2
                && transition.actions[0] != ActionCombo::Single(Action::StellarSteadyHand)
                && transition.condition == Condition::Normal
        }));
        assert!(transitions.iter().any(|transition| {
            transition.actions[0] == ActionCombo::Single(Action::DaringTouch)
        }));
    }

    #[test]
    fn exact_progress_stop_returns_a_concrete_partial_window() {
        let mut settings = settings(1);
        settings.simulator_settings.max_progress = 1000;
        let root = SimulationState::new(&settings.simulator_settings);
        let transitions = stellar_window_transitions(&settings, root, Condition::Normal, Some(500));
        assert!(transitions.iter().any(|transition| {
            transition.state.progress == 500
                && transition.actions.as_slice()
                    == [
                        ActionCombo::Single(Action::StellarSteadyHand),
                        ActionCombo::Single(Action::RapidSynthesis),
                    ]
                && transition.state.effects.stellar_steady_hand() == 2
        }));
    }
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
    fn careful_observation_is_a_poor_only_zero_step_condition_advance() {
        let settings = solver_settings();
        let mut root = SimulationState::new(&SETTINGS);
        root.effects = root
            .effects
            .with_careful_observation_charges(1)
            .with_crafter_delineations(1)
            .canonicalize_specialist_resources();
        let (state, condition) = use_action_combo_with_condition(
            &settings,
            root,
            ActionCombo::Single(Action::CarefulObservation),
            Condition::Poor,
        )
        .unwrap();
        assert_eq!(condition, Condition::Normal);
        assert_eq!(state.effects.careful_observation_charges(), 0);
        assert_eq!(state.effects.crafter_delineations(), 0);
    }

    #[test]
    fn final_appraisal_is_dominated_at_a_normal_basic_synthesis_finish() {
        let settings = solver_settings();
        let mut root = SimulationState::new(&SETTINGS);
        root.progress = 4_900;

        assert!(final_appraisal_is_dominated(
            &settings,
            root,
            Condition::Normal
        ));
        assert!(!final_appraisal_is_dominated(
            &settings,
            root,
            Condition::Poor
        ));
        root.progress = 0;
        assert!(!final_appraisal_is_dominated(
            &settings,
            root,
            Condition::Normal
        ));
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
            Action::FinalAppraisal as u8
        );
        let careful_observation = ActionCombo::Single(Action::CarefulObservation);
        assert_eq!(
            ActionCombo::from_bits(careful_observation.into_bits()),
            careful_observation
        );
    }
}
