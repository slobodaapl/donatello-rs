use raphael_sim::{Action, ActionError, Condition, Settings, SimulationState};

pub const CONDITION_COUNT: usize = 11;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalStatus {
    Active,
    Complete,
    FailedDurability,
    FailedQuality,
    Horizon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    pub simulation: SimulationState,
    pub condition: Condition,
    pub step: u8,
    pub decisions: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct RecipeModel {
    pub settings: Settings,
    pub required_quality: u16,
    pub condition_probabilities_bps: [u16; CONDITION_COUNT],
    pub max_steps: u8,
    pub max_decisions: u8,
}

impl RecipeModel {
    pub fn validate(&self) -> Result<(), String> {
        if self.required_quality == 0 || self.required_quality > self.settings.max_quality {
            return Err(String::from(
                "required quality must be within 1..=max quality",
            ));
        }
        if self.max_steps == 0 || self.max_decisions < self.max_steps {
            return Err(String::from("invalid Gabriel decision horizon"));
        }
        let probability_sum: u32 = self
            .condition_probabilities_bps
            .iter()
            .map(|value| u32::from(*value))
            .sum();
        if probability_sum != 10_000 {
            return Err(format!(
                "condition probabilities total {probability_sum} basis points; expected 10000"
            ));
        }
        Ok(())
    }

    pub fn status(&self, state: State) -> TerminalStatus {
        if state.simulation.progress >= self.settings.max_progress {
            return if state.simulation.quality >= self.required_quality {
                TerminalStatus::Complete
            } else {
                TerminalStatus::FailedQuality
            };
        }
        if state.simulation.durability == 0 {
            return TerminalStatus::FailedDurability;
        }
        if state.step >= self.max_steps || state.decisions >= self.max_decisions {
            return TerminalStatus::Horizon;
        }
        TerminalStatus::Active
    }

    pub fn apply_outcome(
        &self,
        state: State,
        action: Action,
        succeeded: bool,
        condition_draw: f64,
    ) -> Result<State, ActionError> {
        let specialist_action = matches!(
            action,
            Action::CarefulObservation | Action::HeartAndSoul | Action::QuickInnovation
        );
        if specialist_action && state.simulation.effects.crafter_delineations() == 0 {
            return Err(ActionError::NoRemainingUses);
        }
        let mut simulation = if action == Action::CarefulObservation {
            if state.simulation.effects.careful_observation_charges() == 0 {
                return Err(ActionError::NoRemainingUses);
            }
            state.simulation
        } else {
            state.simulation.use_action_with_outcome(
                action,
                state.condition,
                &self.settings,
                succeeded,
            )?
        };
        let previous_effects = state.simulation.effects;
        let spent_delineation = succeeded
            && matches!(
                action,
                Action::CarefulObservation | Action::HeartAndSoul | Action::QuickInnovation
            );
        simulation.effects = simulation
            .effects
            .with_careful_observation_charges(
                previous_effects
                    .careful_observation_charges()
                    .saturating_sub(u8::from(succeeded && action == Action::CarefulObservation)),
            )
            .with_crafter_delineations(
                previous_effects
                    .crafter_delineations()
                    .saturating_sub(u8::from(spent_delineation)),
            )
            .with_heart_and_soul_available(
                previous_effects.heart_and_soul_available()
                    && !(succeeded && action == Action::HeartAndSoul),
            )
            .with_quick_innovation_available(
                previous_effects.quick_innovation_available()
                    && !(succeeded && action == Action::QuickInnovation),
            );
        let condition = if action.advances_condition() {
            self.next_condition(state.condition, condition_draw)
        } else {
            state.condition
        };
        Ok(State {
            simulation,
            condition,
            step: state
                .step
                .saturating_add(u8::from(action.increases_step_count())),
            decisions: state.decisions.saturating_add(1),
        })
    }

    pub fn apply_counter(
        &self,
        state: State,
        action: Action,
        stream: &mut CounterStream,
    ) -> Result<State, ActionError> {
        let success_rate = action.success_rate(&state.simulation, state.condition);
        let succeeded =
            success_rate == 100 || stream.success_draw() < f64::from(success_rate) / 100.0;
        let condition_draw = if action.advances_condition() && state.condition != Condition::Robust
        {
            stream.condition_draw()
        } else {
            0.0
        };
        self.apply_outcome(state, action, succeeded, condition_draw)
    }

    fn next_condition(&self, current: Condition, draw: f64) -> Condition {
        match current {
            Condition::Excellent => Condition::Poor,
            Condition::Poor => Condition::Normal,
            Condition::GoodOmen => Condition::Good,
            Condition::Robust => Condition::Sturdy,
            _ => {
                let mut threshold = 0.0;
                for (index, probability) in self.condition_probabilities_bps.iter().enumerate() {
                    threshold += f64::from(*probability) / 10_000.0;
                    if draw < threshold {
                        return condition_from_index(index);
                    }
                }
                Condition::Normal
            }
        }
    }
}

fn condition_from_index(index: usize) -> Condition {
    match index {
        1 => Condition::Good,
        2 => Condition::Excellent,
        3 => Condition::Poor,
        4 => Condition::Centered,
        5 => Condition::Sturdy,
        6 => Condition::Pliant,
        7 => Condition::Malleable,
        8 => Condition::Primed,
        9 => Condition::GoodOmen,
        10 => Condition::Robust,
        _ => Condition::Normal,
    }
}

const MIX_1: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_2: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_3: u64 = 0x94D0_49BB_1331_11EB;

#[derive(Debug, Clone, Copy)]
pub struct CounterStream {
    seed: u64,
    success_index: u64,
    condition_index: u64,
}

impl CounterStream {
    pub const fn new(seed: u64) -> Self {
        Self {
            seed,
            success_index: 0,
            condition_index: 0,
        }
    }

    pub fn success_draw(&mut self) -> f64 {
        let value = counter_uniform(self.seed, self.success_index, 0);
        self.success_index += 1;
        value
    }

    pub fn condition_draw(&mut self) -> f64 {
        let value = counter_uniform(self.seed, self.condition_index, 1);
        self.condition_index += 1;
        value
    }
}

pub fn mix64(value: u64) -> u64 {
    let mut mixed = value.wrapping_add(MIX_1);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(MIX_2);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(MIX_3);
    mixed ^ (mixed >> 31)
}

pub fn derive_seed(seed: u64, index: u64) -> u64 {
    mix64(seed.wrapping_add(index.wrapping_add(1).wrapping_mul(MIX_2)))
}

fn counter_uniform(seed: u64, index: u64, kind: u64) -> f64 {
    let bits = mix64(
        seed.wrapping_add(index.wrapping_add(1).wrapping_mul(MIX_1))
            .wrapping_add(kind.wrapping_add(17).wrapping_mul(MIX_3)),
    );
    ((bits >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use raphael_sim::{ActionMask, Effects};

    fn model() -> RecipeModel {
        RecipeModel {
            settings: Settings {
                max_cp: 700,
                max_durability: 60,
                max_progress: 11_250,
                max_quality: 31_520,
                base_progress: 319,
                base_quality: 318,
                job_level: 100,
                allowed_actions: ActionMask::all().remove(Action::TrainedEye),
                adversarial: false,
                backload_progress: false,
                stellar_steady_hand_charges: 0,
            },
            required_quality: 31_520,
            condition_probabilities_bps: [
                2_000, 1_000, 0, 0, 1_500, 1_000, 1_500, 1_000, 1_000, 0, 1_000,
            ],
            max_steps: 55,
            max_decisions: 64,
        }
    }

    #[test]
    fn robust_forces_sturdy_without_using_probability_row() {
        let model = model();
        let state = State {
            simulation: SimulationState {
                cp: 700,
                durability: 60,
                progress: 0,
                quality: 0,
                unreliable_quality: 0,
                effects: Effects::new(),
            },
            condition: Condition::Robust,
            step: 0,
            decisions: 0,
        };
        let next = model
            .apply_outcome(state, Action::Observe, true, 0.99)
            .unwrap();
        assert_eq!(next.condition, Condition::Sturdy);
    }

    #[test]
    fn aqueduct_vector_covers_exactly_one_probability_mass() {
        assert!(model().validate().is_ok());
    }

    #[test]
    fn reference_specialist_counters_remain_raw_until_their_action_consumes_them() {
        let model = model();
        let mut simulation = SimulationState::new(&model.settings);
        simulation.effects = simulation
            .effects
            .with_careful_observation_charges(3)
            .with_crafter_delineations(2)
            .with_heart_and_soul_available(true)
            .with_quick_innovation_available(true);
        let state = State {
            simulation,
            condition: Condition::Normal,
            step: 0,
            decisions: 0,
        };

        let after_synthesis = model
            .apply_outcome(state, Action::BasicSynthesis, true, 0.0)
            .unwrap();
        assert_eq!(
            after_synthesis
                .simulation
                .effects
                .careful_observation_charges(),
            3
        );
        assert_eq!(after_synthesis.simulation.effects.crafter_delineations(), 2);
        assert!(
            after_synthesis
                .simulation
                .effects
                .heart_and_soul_available()
        );
        assert!(
            after_synthesis
                .simulation
                .effects
                .quick_innovation_available()
        );

        let after_observation = model
            .apply_outcome(after_synthesis, Action::CarefulObservation, true, 0.0)
            .unwrap();
        assert_eq!(
            after_observation
                .simulation
                .effects
                .careful_observation_charges(),
            2
        );
        assert_eq!(
            after_observation.simulation.effects.crafter_delineations(),
            1
        );
    }

    #[test]
    fn specialist_actions_are_unusable_without_delineations() {
        let model = model();
        let mut simulation = SimulationState::new(&model.settings);
        simulation.effects = simulation
            .effects
            .with_careful_observation_charges(3)
            .with_crafter_delineations(0)
            .with_heart_and_soul_available(true)
            .with_quick_innovation_available(true);
        let state = State {
            simulation,
            condition: Condition::Normal,
            step: 0,
            decisions: 0,
        };

        for action in [
            Action::CarefulObservation,
            Action::HeartAndSoul,
            Action::QuickInnovation,
        ] {
            assert_eq!(
                model.apply_outcome(state, action, true, 0.0),
                Err(ActionError::NoRemainingUses)
            );
        }
    }
}
