use crate::actions::*;
use crate::effects::*;
use crate::{Condition, Settings};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimulationState {
    pub cp: u16,
    pub durability: u16,
    pub progress: u16,
    pub quality: u16,            // previous unguarded action = Poor
    pub unreliable_quality: u16, // previous unguarded action = Normal, diff with quality
    pub effects: Effects,
}

impl SimulationState {
    pub fn new(settings: &Settings) -> Self {
        Self {
            cp: settings.max_cp,
            durability: settings.max_durability,
            progress: 0,
            quality: 0,
            unreliable_quality: 0,
            effects: Effects::initial(settings),
        }
    }

    pub fn from_macro(settings: &Settings, actions: &[Action]) -> Result<Self, ActionError> {
        Self::from_macro_from_state(settings, Self::new(settings), actions)
    }

    pub fn from_macro_from_state(
        settings: &Settings,
        mut state: Self,
        actions: &[Action],
    ) -> Result<Self, ActionError> {
        for action in actions {
            state = state.use_action(*action, Condition::Normal, settings)?;
        }
        Ok(state)
    }

    pub fn from_macro_continue_on_error(
        settings: &Settings,
        actions: &[Action],
    ) -> (Self, Vec<Result<(), ActionError>>) {
        let mut state = Self::new(settings);
        let mut errors = Vec::new();
        for action in actions {
            state = match state.use_action(*action, Condition::Normal, settings) {
                Ok(new_state) => {
                    errors.push(Ok(()));
                    new_state
                }
                Err(err) => {
                    errors.push(Err(err));
                    state
                }
            };
        }
        (state, errors)
    }

    pub fn is_final(&self, settings: &Settings) -> bool {
        self.durability == 0 || self.progress >= settings.max_progress
    }

    fn check_common_preconditions<A: ActionImpl>(
        &self,
        settings: &Settings,
        condition: Condition,
    ) -> Result<(), ActionError> {
        if settings.job_level < A::LEVEL_REQUIREMENT {
            Err(ActionError::InsufficientLevels)
        } else if !settings.allowed_actions.has_mask(A::ACTION_MASK) {
            Err(ActionError::Disabled)
        } else if self.is_final(settings) {
            Err(ActionError::StateIsFinal)
        } else if A::cp_cost(self, settings, condition) > self.cp {
            Err(ActionError::InsufficientCP)
        } else {
            Ok(())
        }
    }

    pub fn use_action_impl<A: ActionImpl>(
        &self,
        settings: &Settings,
        condition: Condition,
    ) -> Result<Self, ActionError> {
        self.use_action_impl_with_outcome::<A>(settings, condition, true)
    }

    fn use_action_impl_with_outcome<A: ActionImpl>(
        &self,
        settings: &Settings,
        condition: Condition,
        succeeded: bool,
    ) -> Result<Self, ActionError> {
        self.check_common_preconditions::<A>(settings, condition)?;
        A::precondition(self, settings, condition)?;

        let mut state = *self;

        if A::base_durability_cost(&state, settings) != 0 {
            state.durability = state
                .durability
                .saturating_sub(A::durability_cost(self, settings, condition));
        }

        state.cp -= A::cp_cost(self, settings, condition);

        let quality_increase = if succeeded {
            A::quality_increase(self, settings, condition)
        } else {
            0
        };
        if settings.adversarial {
            let guard_active = state.effects.adversarial_guard_active();
            if quality_increase != 0 {
                if guard_active {
                    state.quality = state.quality.saturating_add(quality_increase);
                    state.unreliable_quality = 0;
                } else {
                    let adversarial_quality_increase =
                        A::quality_increase(self, settings, Condition::Poor);
                    let quality_diff = quality_increase - adversarial_quality_increase;
                    state.quality = state
                        .quality
                        .saturating_add(adversarial_quality_increase)
                        .saturating_add(std::cmp::min(state.unreliable_quality, quality_diff));
                    state.unreliable_quality =
                        quality_diff.saturating_sub(state.unreliable_quality);
                }
            } else if A::INCREASES_STEP_COUNT && !guard_active {
                state.unreliable_quality = 0;
            }
        } else {
            state.quality = state.quality.saturating_add(quality_increase);
        }
        if quality_increase != 0 && settings.job_level >= 11 {
            state
                .effects
                .set_inner_quiet(std::cmp::min(10, state.effects.inner_quiet() + 1));
        }

        let progress_increase = match (succeeded, condition) {
            (true, Condition::Malleable) => {
                A::progress_increase(self, settings).saturating_mul(3) / 2
            }
            (true, _) => A::progress_increase(self, settings),
            (false, _) => 0,
        };
        state.progress = state.progress.saturating_add(progress_increase);

        if progress_increase != 0
            && state.progress >= settings.max_progress
            && self.effects.final_appraisal() != 0
        {
            state.progress = settings.max_progress.saturating_sub(1);
            state.effects.set_final_appraisal(0);
        }

        if state.is_final(settings) {
            return Ok(state);
        }

        let is_synthesis_begin = state.effects.combo() == Combo::SynthesisBegin;
        state.effects =
            Effects::from_bits(state.effects.into_bits() & A::EFFECT_RESET_MASK.into_bits());
        if !A::INCREASES_STEP_COUNT && is_synthesis_begin {
            // SynthesisBegin is implemented as a combo but it is in reality not a combo.
            // Actions that require the SynthesisBegin "combo" actually check the step count,
            // which does not increase when using actions such as QuickInnovation.
            state.effects.set_combo(Combo::SynthesisBegin);
        }

        if A::INCREASES_STEP_COUNT {
            if state.effects.manipulation() != 0 {
                state.durability = std::cmp::min(settings.max_durability, state.durability + 5);
            }
            state.effects = state.effects.tick_down();
        } else if !A::PRESERVES_EFFECT_DURATIONS && state.effects.stellar_steady_hand() != 0 {
            state
                .effects
                .set_stellar_steady_hand(state.effects.stellar_steady_hand().saturating_sub(1));
        }

        if succeeded {
            A::transform(&mut state, settings, condition);
            state.effects =
                Effects::from_bits(state.effects.into_bits() | A::EFFECT_SET_MASK.into_bits());
        }

        if condition == Condition::Primed {
            let set = A::EFFECT_SET_MASK;
            if set.waste_not() != 0 {
                state
                    .effects
                    .set_waste_not(state.effects.waste_not().saturating_add(2));
            }
            if set.innovation() != 0 {
                state
                    .effects
                    .set_innovation(state.effects.innovation().saturating_add(2));
            }
            if set.veneration() != 0 {
                state
                    .effects
                    .set_veneration(state.effects.veneration().saturating_add(2));
            }
            if set.great_strides() != 0 {
                state
                    .effects
                    .set_great_strides(state.effects.great_strides().saturating_add(2));
            }
            if set.muscle_memory() != 0 {
                state
                    .effects
                    .set_muscle_memory(state.effects.muscle_memory().saturating_add(2));
            }
            if set.manipulation() != 0 {
                state
                    .effects
                    .set_manipulation(state.effects.manipulation().saturating_add(2));
            }
            if set.final_appraisal() != 0 {
                state
                    .effects
                    .set_final_appraisal(state.effects.final_appraisal().saturating_add(2));
            }
        }

        if progress_increase != 0 && settings.backload_progress {
            state.effects = state.effects.strip_quality_effects();
            state.unreliable_quality = 0;
        } else if settings.adversarial && quality_increase != 0 {
            state
                .effects
                .set_special_quality_state(SpecialQualityState::AdversarialGuard);
        }

        state.effects = state.effects.canonicalize_specialist_resources();

        Ok(state)
    }

    pub fn use_action(
        &self,
        action: Action,
        condition: Condition,
        settings: &Settings,
    ) -> Result<Self, ActionError> {
        if action.success_rate(self, condition) < 100 {
            return Err(ActionError::UnreliableAction);
        }
        self.use_action_with_outcome(action, condition, settings, true)
    }

    /// Applies one already-rolled action outcome. Unlike [`Self::use_action`], this accepts
    /// unreliable actions and models both their success and failure branches.
    pub fn use_action_with_outcome(
        &self,
        action: Action,
        condition: Condition,
        settings: &Settings,
        succeeded: bool,
    ) -> Result<Self, ActionError> {
        match action {
            Action::BasicSynthesis => {
                self.use_action_impl_with_outcome::<BasicSynthesis>(settings, condition, succeeded)
            }
            Action::BasicTouch => {
                self.use_action_impl_with_outcome::<BasicTouch>(settings, condition, succeeded)
            }
            Action::MasterMend => {
                self.use_action_impl_with_outcome::<MasterMend>(settings, condition, succeeded)
            }
            Action::Observe => {
                self.use_action_impl_with_outcome::<Observe>(settings, condition, succeeded)
            }
            Action::TricksOfTheTrade => self
                .use_action_impl_with_outcome::<TricksOfTheTrade>(settings, condition, succeeded),
            Action::WasteNot => {
                self.use_action_impl_with_outcome::<WasteNot>(settings, condition, succeeded)
            }
            Action::Veneration => {
                self.use_action_impl_with_outcome::<Veneration>(settings, condition, succeeded)
            }
            Action::StandardTouch => {
                self.use_action_impl_with_outcome::<StandardTouch>(settings, condition, succeeded)
            }
            Action::GreatStrides => {
                self.use_action_impl_with_outcome::<GreatStrides>(settings, condition, succeeded)
            }
            Action::Innovation => {
                self.use_action_impl_with_outcome::<Innovation>(settings, condition, succeeded)
            }
            Action::WasteNot2 => {
                self.use_action_impl_with_outcome::<WasteNot2>(settings, condition, succeeded)
            }
            Action::ByregotsBlessing => self
                .use_action_impl_with_outcome::<ByregotsBlessing>(settings, condition, succeeded),
            Action::PreciseTouch => {
                self.use_action_impl_with_outcome::<PreciseTouch>(settings, condition, succeeded)
            }
            Action::MuscleMemory => {
                self.use_action_impl_with_outcome::<MuscleMemory>(settings, condition, succeeded)
            }
            Action::CarefulSynthesis => self
                .use_action_impl_with_outcome::<CarefulSynthesis>(settings, condition, succeeded),
            Action::Manipulation => {
                self.use_action_impl_with_outcome::<Manipulation>(settings, condition, succeeded)
            }
            Action::PrudentTouch => {
                self.use_action_impl_with_outcome::<PrudentTouch>(settings, condition, succeeded)
            }
            Action::AdvancedTouch => {
                self.use_action_impl_with_outcome::<AdvancedTouch>(settings, condition, succeeded)
            }
            Action::Reflect => {
                self.use_action_impl_with_outcome::<Reflect>(settings, condition, succeeded)
            }
            Action::PreparatoryTouch => self
                .use_action_impl_with_outcome::<PreparatoryTouch>(settings, condition, succeeded),
            Action::Groundwork => {
                self.use_action_impl_with_outcome::<Groundwork>(settings, condition, succeeded)
            }
            Action::DelicateSynthesis => self
                .use_action_impl_with_outcome::<DelicateSynthesis>(settings, condition, succeeded),
            Action::IntensiveSynthesis => self
                .use_action_impl_with_outcome::<IntensiveSynthesis>(settings, condition, succeeded),
            Action::TrainedEye => {
                self.use_action_impl_with_outcome::<TrainedEye>(settings, condition, succeeded)
            }
            Action::HeartAndSoul => {
                self.use_action_impl_with_outcome::<HeartAndSoul>(settings, condition, succeeded)
            }
            Action::PrudentSynthesis => self
                .use_action_impl_with_outcome::<PrudentSynthesis>(settings, condition, succeeded),
            Action::TrainedFinesse => {
                self.use_action_impl_with_outcome::<TrainedFinesse>(settings, condition, succeeded)
            }
            Action::RefinedTouch => {
                self.use_action_impl_with_outcome::<RefinedTouch>(settings, condition, succeeded)
            }
            Action::QuickInnovation => {
                self.use_action_impl_with_outcome::<QuickInnovation>(settings, condition, succeeded)
            }
            Action::ImmaculateMend => {
                self.use_action_impl_with_outcome::<ImmaculateMend>(settings, condition, succeeded)
            }
            Action::TrainedPerfection => self
                .use_action_impl_with_outcome::<TrainedPerfection>(settings, condition, succeeded),
            Action::StellarSteadyHand => self
                .use_action_impl_with_outcome::<StellarSteadyHand>(settings, condition, succeeded),
            Action::RapidSynthesis => {
                self.use_action_impl_with_outcome::<RapidSynthesis>(settings, condition, succeeded)
            }
            Action::HastyTouch => {
                self.use_action_impl_with_outcome::<HastyTouch>(settings, condition, succeeded)
            }
            Action::DaringTouch => {
                self.use_action_impl_with_outcome::<DaringTouch>(settings, condition, succeeded)
            }
            Action::FinalAppraisal => {
                self.use_action_impl_with_outcome::<FinalAppraisal>(settings, condition, succeeded)
            }
            Action::CarefulObservation => self
                .use_action_impl_with_outcome::<CarefulObservation>(settings, condition, succeeded),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ActionMask;

    #[test]
    fn stellar_steady_hand_counts_zero_step_actions() {
        let settings = Settings {
            max_cp: 500,
            max_durability: 40,
            max_progress: 100,
            max_quality: 1000,
            base_progress: 10,
            base_quality: 10,
            job_level: 100,
            allowed_actions: ActionMask::all(),
            adversarial: false,
            backload_progress: false,
            stellar_steady_hand_charges: 1,
        };
        let state = SimulationState::new(&settings)
            .use_action(Action::StellarSteadyHand, Condition::Normal, &settings)
            .unwrap();
        assert_eq!(state.effects.stellar_steady_hand(), 3);

        let state = state
            .use_action(Action::FinalAppraisal, Condition::Normal, &settings)
            .unwrap();
        assert_eq!(state.effects.stellar_steady_hand(), 2);
    }

    #[test]
    fn careful_observation_preserves_effects_and_consumes_shared_resources() {
        let settings = Settings {
            max_cp: 500,
            max_durability: 40,
            max_progress: 100,
            max_quality: 1000,
            base_progress: 10,
            base_quality: 10,
            job_level: 100,
            allowed_actions: ActionMask::all(),
            adversarial: false,
            backload_progress: false,
            stellar_steady_hand_charges: 1,
        };
        let mut state = SimulationState::new(&settings);
        state.effects = state
            .effects
            .with_inner_quiet(3)
            .with_innovation(2)
            .with_stellar_steady_hand(2)
            .with_expedience(true)
            .with_combo(Combo::BasicTouch)
            .with_careful_observation_charges(2)
            .with_crafter_delineations(4)
            .canonicalize_specialist_resources();

        let observed = state
            .use_action(Action::CarefulObservation, Condition::Poor, &settings)
            .unwrap();
        assert_eq!(observed.effects.inner_quiet(), 3);
        assert_eq!(observed.effects.innovation(), 2);
        assert_eq!(observed.effects.stellar_steady_hand(), 2);
        assert!(observed.effects.expedience());
        assert_eq!(observed.effects.combo(), Combo::BasicTouch);
        assert_eq!(observed.effects.careful_observation_charges(), 1);
        assert_eq!(observed.effects.crafter_delineations(), 3);
        assert_eq!(
            state.use_action(Action::CarefulObservation, Condition::Normal, &settings),
            Err(ActionError::SpecialConditionNotMet)
        );
    }

    #[test]
    fn splendor_cosmic_tool_uses_175_percent_good_quality() {
        let settings = Settings {
            max_cp: 500,
            max_durability: 40,
            max_progress: 100,
            max_quality: 1000,
            base_progress: 10,
            base_quality: 10,
            job_level: 100,
            allowed_actions: ActionMask::all(),
            adversarial: false,
            backload_progress: false,
            stellar_steady_hand_charges: 0,
        };
        let normal_tool = SimulationState::new(&settings)
            .use_action(Action::BasicTouch, Condition::Good, &settings)
            .unwrap();
        let mut cosmic_tool = SimulationState::new(&settings);
        cosmic_tool.effects.set_splendor_cosmic(true);
        let cosmic_tool = cosmic_tool
            .use_action(Action::BasicTouch, Condition::Good, &settings)
            .unwrap();

        assert_eq!(normal_tool.quality, 15);
        assert_eq!(cosmic_tool.quality, 17);
    }

    #[test]
    fn unreliable_failure_consumes_the_action_without_granting_success_effects() {
        let settings = Settings {
            max_cp: 500,
            max_durability: 40,
            max_progress: 100,
            max_quality: 1000,
            base_progress: 10,
            base_quality: 10,
            job_level: 100,
            allowed_actions: ActionMask::all(),
            adversarial: false,
            backload_progress: false,
            stellar_steady_hand_charges: 0,
        };
        let state = SimulationState::new(&settings);

        assert_eq!(
            state.use_action(Action::HastyTouch, Condition::Normal, &settings),
            Err(ActionError::UnreliableAction)
        );
        let failed = state
            .use_action_with_outcome(Action::HastyTouch, Condition::Normal, &settings, false)
            .unwrap();
        assert_eq!(failed.cp, state.cp);
        assert_eq!(failed.durability, state.durability - 10);
        assert_eq!(failed.quality, 0);
        assert_eq!(failed.effects.inner_quiet(), 0);
        assert!(!failed.effects.expedience());

        let succeeded = state
            .use_action_with_outcome(Action::HastyTouch, Condition::Normal, &settings, true)
            .unwrap();
        assert!(succeeded.quality > 0);
        assert_eq!(succeeded.effects.inner_quiet(), 1);
        assert!(succeeded.effects.expedience());
    }
}
