use std::cmp::Reverse;
use std::collections::BinaryHeap;

use raphael_sim::{Action, Condition, Effects, SimulationState};
use rustc_hash::FxHashMap;
use smallvec::{SmallVec, smallvec};

use crate::{
    AtomicFlag, SolverSettings,
    actions::{final_appraisal_is_dominated, stellar_window_transitions},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressTarget {
    Complete,
    OneShort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressPolicy {
    Pareto,
    Fastest,
}

#[derive(Debug, Clone)]
pub struct ProgressEndpoint {
    pub state: SimulationState,
    pub condition: Condition,
    pub actions: Vec<Action>,
    pub advancing_steps: u16,
    pub action_count: u16,
    pub duration: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    progress: u16,
    condition: Condition,
    effects: Effects,
    unreliable_quality: u16,
}

impl Key {
    fn new(state: SimulationState, condition: Condition) -> Self {
        Self {
            progress: state.progress,
            condition,
            effects: state.effects,
            unreliable_quality: state.unreliable_quality,
        }
    }
}

#[derive(Debug, Clone)]
struct Label {
    state: SimulationState,
    condition: Condition,
    parent: Option<usize>,
    actions: SmallVec<[Action; 4]>,
    advancing_steps: u16,
    action_count: u16,
    duration: u16,
    alive: bool,
}

impl Label {
    fn dominates(&self, other: &Self, policy: ProgressPolicy) -> bool {
        self.state.cp >= other.state.cp
            && self.state.durability >= other.state.durability
            && (policy == ProgressPolicy::Fastest || self.state.quality >= other.state.quality)
            && self.action_count <= other.action_count
            && self.duration <= other.duration
    }
}

pub struct ProgressFrontierSolver {
    settings: SolverSettings,
    interrupt: AtomicFlag,
}

impl ProgressFrontierSolver {
    pub fn new(settings: SolverSettings, interrupt: AtomicFlag) -> Self {
        Self {
            settings,
            interrupt,
        }
    }

    pub fn solve(
        &self,
        root: SimulationState,
        condition: Condition,
        target: ProgressTarget,
        policy: ProgressPolicy,
    ) -> Vec<ProgressEndpoint> {
        self.solve_with_expansion_limit(root, condition, target, policy, None)
    }

    pub fn solve_with_expansion_limit(
        &self,
        root: SimulationState,
        condition: Condition,
        target: ProgressTarget,
        policy: ProgressPolicy,
        expansion_limit: Option<usize>,
    ) -> Vec<ProgressEndpoint> {
        let mut labels = vec![Label {
            state: root,
            condition,
            parent: None,
            actions: SmallVec::new(),
            advancing_steps: 0,
            action_count: 0,
            duration: 0,
            alive: true,
        }];
        let mut fronts = FxHashMap::<Key, Vec<usize>>::default();
        fronts.insert(Key::new(root, condition), vec![0]);
        let mut queue = BinaryHeap::new();
        let maximum_progress = maximum_progress_increase(&self.settings);
        queue.push((
            Reverse((
                progress_step_lower_bound(root, &self.settings, maximum_progress),
                0_u16,
                0_u16,
            )),
            0_usize,
        ));
        let mut endpoints = Vec::new();
        let mut expansions = 0;

        while let Some((_, label_index)) = queue.pop() {
            if self.interrupt.is_set() {
                break;
            }
            if !labels[label_index].alive {
                continue;
            }
            if expansion_limit.is_some_and(|limit| expansions >= limit) {
                break;
            }
            expansions += 1;
            if self.reached(labels[label_index].state, target) {
                endpoints.push(label_index);
                if policy == ProgressPolicy::Fastest {
                    break;
                }
                continue;
            }
            if labels[label_index]
                .state
                .is_final(&self.settings.simulator_settings)
            {
                continue;
            }

            let parent_state = labels[label_index].state;
            let parent_condition = labels[label_index].condition;
            let parent_action_count = labels[label_index].action_count;
            let parent_duration = labels[label_index].duration;
            let parent_advancing_steps = labels[label_index].advancing_steps;
            let stellar_active = parent_state.effects.stellar_steady_hand() != 0;
            let mut transitions = Vec::new();
            if !stellar_active {
                for &action in actions(target) {
                    if matches!(
                        action,
                        Action::StellarSteadyHand
                            | Action::RapidSynthesis
                            | Action::HastyTouch
                            | Action::DaringTouch
                    ) {
                        continue;
                    }
                    if action == Action::FinalAppraisal
                        && final_appraisal_is_dominated(
                            &self.settings,
                            parent_state,
                            parent_condition,
                        )
                    {
                        continue;
                    }
                    let Ok(state) = parent_state.use_action(
                        action,
                        parent_condition,
                        &self.settings.simulator_settings,
                    ) else {
                        continue;
                    };
                    let condition = if action.advances_condition() {
                        parent_condition.deterministic_successor()
                    } else {
                        parent_condition
                    };
                    transitions.push((
                        state,
                        condition,
                        smallvec![action],
                        u16::from(state.progress > parent_state.progress),
                    ));
                }
            }
            let stop_progress = Some(match target {
                ProgressTarget::Complete => self.settings.max_progress(),
                ProgressTarget::OneShort => self.settings.max_progress().saturating_sub(1),
            });
            transitions.extend(
                stellar_window_transitions(
                    &self.settings,
                    parent_state,
                    parent_condition,
                    stop_progress,
                )
                .into_iter()
                .map(|transition| {
                    let actions: SmallVec<[Action; 4]> = transition
                        .actions
                        .iter()
                        .flat_map(|action| action.actions().iter().copied())
                        .collect();
                    (
                        transition.state,
                        transition.condition,
                        actions,
                        transition.advancing_steps,
                    )
                }),
            );
            for (state, condition, actions, transition_advancing_steps) in transitions {
                let action_count = parent_action_count
                    .saturating_add(u16::try_from(actions.len()).unwrap_or(u16::MAX));
                let duration = parent_duration.saturating_add(
                    actions
                        .iter()
                        .map(|action| u16::from(action.time_cost()))
                        .sum(),
                );
                let advancing_steps =
                    parent_advancing_steps.saturating_add(transition_advancing_steps);
                let candidate = Label {
                    state,
                    condition,
                    parent: Some(label_index),
                    actions,
                    advancing_steps,
                    action_count,
                    duration,
                    alive: true,
                };
                let key = Key::new(state, condition);
                let front = fronts.entry(key).or_default();
                if front.iter().any(|&existing| {
                    labels[existing].alive && labels[existing].dominates(&candidate, policy)
                }) {
                    continue;
                }
                for &existing in front.iter() {
                    if labels[existing].alive && candidate.dominates(&labels[existing], policy) {
                        labels[existing].alive = false;
                    }
                }
                let index = labels.len();
                labels.push(candidate);
                front.push(index);
                let estimated_actions = action_count.saturating_add(progress_step_lower_bound(
                    state,
                    &self.settings,
                    maximum_progress,
                ));
                queue.push((Reverse((estimated_actions, action_count, duration)), index));
            }
        }

        endpoints
            .into_iter()
            .filter(|&index| labels[index].alive)
            .map(|index| ProgressEndpoint {
                state: labels[index].state,
                condition: labels[index].condition,
                actions: reconstruct(&labels, index),
                advancing_steps: labels[index].advancing_steps,
                action_count: labels[index].action_count,
                duration: labels[index].duration,
            })
            .collect()
    }

    fn reached(&self, state: SimulationState, target: ProgressTarget) -> bool {
        match target {
            ProgressTarget::Complete => state.progress >= self.settings.max_progress(),
            ProgressTarget::OneShort => {
                state.progress == self.settings.max_progress().saturating_sub(1)
            }
        }
    }
}

fn progress_step_lower_bound(
    state: SimulationState,
    settings: &SolverSettings,
    maximum_progress: u16,
) -> u16 {
    settings
        .max_progress()
        .saturating_sub(state.progress)
        .div_ceil(maximum_progress.max(1))
}

fn maximum_progress_increase(settings: &SolverSettings) -> u16 {
    // Optimistic analytical ceiling: 500% action efficiency, Muscle Memory +
    // Veneration (250% effect modifier), then Malleable (150%). Using a ceiling
    // independent of currently reachable buffs keeps the A* step bound admissible.
    let base = u64::from(settings.simulator_settings.base_progress);
    let progress = base.saturating_mul(500).saturating_mul(25) / 1000;
    u16::try_from(progress.saturating_mul(3) / 2).unwrap_or(u16::MAX)
}

fn reconstruct(labels: &[Label], mut index: usize) -> Vec<Action> {
    let mut result = Vec::new();
    while let Some(parent) = labels[index].parent {
        result.extend(labels[index].actions.iter().rev());
        index = parent;
    }
    result.reverse();
    result
}

fn actions(target: ProgressTarget) -> &'static [Action] {
    const COMPLETE: &[Action] = &[
        Action::BasicSynthesis,
        Action::TricksOfTheTrade,
        Action::WasteNot,
        Action::Veneration,
        Action::WasteNot2,
        Action::MuscleMemory,
        Action::CarefulSynthesis,
        Action::Manipulation,
        Action::Groundwork,
        Action::DelicateSynthesis,
        Action::IntensiveSynthesis,
        Action::HeartAndSoul,
        Action::PrudentSynthesis,
        Action::QuickInnovation,
        Action::ImmaculateMend,
        Action::TrainedPerfection,
        Action::StellarSteadyHand,
        Action::RapidSynthesis,
        Action::MasterMend,
    ];
    const ONE_SHORT: &[Action] = &[
        Action::BasicSynthesis,
        Action::TricksOfTheTrade,
        Action::WasteNot,
        Action::Veneration,
        Action::WasteNot2,
        Action::MuscleMemory,
        Action::CarefulSynthesis,
        Action::Manipulation,
        Action::Groundwork,
        Action::DelicateSynthesis,
        Action::IntensiveSynthesis,
        Action::HeartAndSoul,
        Action::PrudentSynthesis,
        Action::QuickInnovation,
        Action::ImmaculateMend,
        Action::TrainedPerfection,
        Action::StellarSteadyHand,
        Action::RapidSynthesis,
        Action::MasterMend,
        Action::FinalAppraisal,
    ];
    match target {
        ProgressTarget::Complete => COMPLETE,
        ProgressTarget::OneShort => ONE_SHORT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raphael_sim::{ActionMask, Settings};

    fn settings() -> SolverSettings {
        SolverSettings {
            simulator_settings: Settings {
                max_cp: 300,
                max_durability: 40,
                max_progress: 500,
                max_quality: 1000,
                base_progress: 100,
                base_quality: 100,
                job_level: 100,
                allowed_actions: ActionMask::regular().add(Action::FinalAppraisal),
                adversarial: false,
                backload_progress: false,
                stellar_steady_hand_charges: 0,
            },
            allow_non_max_quality_solutions: true,
        }
    }

    #[test]
    fn complete_and_one_short_hit_exact_targets() {
        let settings = settings();
        let root = SimulationState::new(&settings.simulator_settings);
        let solver = ProgressFrontierSolver::new(settings, AtomicFlag::new());
        let complete = solver.solve(
            root,
            Condition::Normal,
            ProgressTarget::Complete,
            ProgressPolicy::Fastest,
        );
        assert_eq!(complete.len(), 1);
        assert!(complete[0].state.progress >= settings.max_progress());

        let one_short = solver.solve(
            root,
            Condition::Normal,
            ProgressTarget::OneShort,
            ProgressPolicy::Pareto,
        );
        assert!(!one_short.is_empty());
        assert!(
            one_short
                .iter()
                .all(|endpoint| endpoint.state.progress == settings.max_progress() - 1)
        );
    }

    #[test]
    fn complete_frontier_never_uses_final_appraisal() {
        let settings = settings();
        let root = SimulationState::new(&settings.simulator_settings);
        let solver = ProgressFrontierSolver::new(settings, AtomicFlag::new());
        let endpoints = solver.solve(
            root,
            Condition::Normal,
            ProgressTarget::Complete,
            ProgressPolicy::Pareto,
        );
        assert!(
            endpoints
                .iter()
                .all(|endpoint| { !endpoint.actions.contains(&Action::FinalAppraisal) })
        );
    }

    #[test]
    fn stellar_completion_is_reconstructed_and_stops_when_craft_finishes() {
        let mut settings = settings();
        settings.simulator_settings.max_progress = 1000;
        settings.simulator_settings.allowed_actions = ActionMask::none()
            .add(Action::StellarSteadyHand)
            .add(Action::RapidSynthesis)
            .add(Action::HastyTouch)
            .add(Action::DaringTouch);
        settings.simulator_settings.stellar_steady_hand_charges = 1;
        let root = SimulationState::new(&settings.simulator_settings);
        let endpoint = ProgressFrontierSolver::new(settings, AtomicFlag::new())
            .solve(
                root,
                Condition::Normal,
                ProgressTarget::Complete,
                ProgressPolicy::Fastest,
            )
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(
            endpoint.actions,
            [
                Action::StellarSteadyHand,
                Action::RapidSynthesis,
                Action::RapidSynthesis,
            ]
        );
        assert_eq!(endpoint.action_count, 3);
        assert_eq!(endpoint.duration, 8);
        assert!(endpoint.state.progress >= settings.max_progress());
    }

    #[test]
    fn one_short_uses_final_appraisal_clamp_without_zero_step_cycles() {
        let mut settings = settings();
        settings.simulator_settings.max_progress = 150;
        settings.simulator_settings.allowed_actions = ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::FinalAppraisal);
        let root = SimulationState::new(&settings.simulator_settings);
        let endpoints = ProgressFrontierSolver::new(settings, AtomicFlag::new()).solve(
            root,
            Condition::Excellent,
            ProgressTarget::OneShort,
            ProgressPolicy::Pareto,
        );
        assert!(!endpoints.is_empty());
        assert!(endpoints.iter().all(|endpoint| {
            endpoint.state.progress == 149
                && endpoint.actions.contains(&Action::FinalAppraisal)
                && endpoint.action_count == endpoint.actions.len() as u16
        }));
    }

    #[test]
    fn one_short_does_not_waste_final_appraisal_when_basic_synthesis_already_finishes() {
        let mut settings = settings();
        settings.simulator_settings.max_progress = 150;
        settings.simulator_settings.allowed_actions = ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::FinalAppraisal);
        let mut root = SimulationState::new(&settings.simulator_settings);
        root.progress = 50;

        let endpoints = ProgressFrontierSolver::new(settings, AtomicFlag::new()).solve(
            root,
            Condition::Normal,
            ProgressTarget::OneShort,
            ProgressPolicy::Pareto,
        );

        assert!(endpoints.is_empty());
    }

    #[test]
    fn pareto_frontier_preserves_cp_durability_action_tradeoffs() {
        let mut settings = settings();
        settings.simulator_settings.max_progress = 300;
        settings.simulator_settings.allowed_actions = ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::Groundwork);
        let root = SimulationState::new(&settings.simulator_settings);
        let endpoints = ProgressFrontierSolver::new(settings, AtomicFlag::new()).solve(
            root,
            Condition::Normal,
            ProgressTarget::Complete,
            ProgressPolicy::Pareto,
        );
        assert!(endpoints.iter().any(|endpoint| {
            endpoint.actions == [Action::Groundwork]
                && endpoint.state.cp < root.cp
                && endpoint.state.durability > 0
        }));
        assert!(endpoints.iter().any(|endpoint| {
            endpoint
                .actions
                .iter()
                .all(|action| *action == Action::BasicSynthesis)
                && endpoint.state.cp == root.cp
                && endpoint.action_count > 1
        }));
    }

    #[test]
    fn recipe_38202_completes_with_zero_one_or_two_delineations() {
        for delineations in 0..=2 {
            let settings = SolverSettings {
                simulator_settings: Settings {
                    max_cp: 573,
                    max_durability: 45,
                    max_progress: 6900,
                    max_quality: 22_100,
                    base_progress: 291,
                    base_quality: 269,
                    job_level: 100,
                    allowed_actions: ActionMask::regular()
                        .add(Action::FinalAppraisal)
                        .add(Action::HeartAndSoul)
                        .add(Action::QuickInnovation),
                    adversarial: false,
                    backload_progress: false,
                    stellar_steady_hand_charges: 0,
                },
                allow_non_max_quality_solutions: true,
            };
            let mut root = SimulationState::new(&settings.simulator_settings);
            root.effects.set_crafter_delineations(delineations);
            root.effects = root.effects.canonicalize_specialist_resources();
            let endpoint = ProgressFrontierSolver::new(settings, AtomicFlag::new())
                .solve(
                    root,
                    Condition::Normal,
                    ProgressTarget::Complete,
                    ProgressPolicy::Fastest,
                )
                .into_iter()
                .next()
                .unwrap_or_else(|| panic!("recipe 38202 failed with {delineations} delineations"));
            assert!(endpoint.state.progress >= settings.max_progress());
        }
    }

    #[test]
    fn pareto_endpoints_do_not_dominate_each_other_with_equal_exact_state() {
        let settings = settings();
        let root = SimulationState::new(&settings.simulator_settings);
        let endpoints = ProgressFrontierSolver::new(settings, AtomicFlag::new()).solve(
            root,
            Condition::Normal,
            ProgressTarget::OneShort,
            ProgressPolicy::Pareto,
        );
        for (index, lhs) in endpoints.iter().enumerate() {
            for rhs in endpoints.iter().skip(index + 1) {
                if Key::new(lhs.state, lhs.condition) != Key::new(rhs.state, rhs.condition) {
                    continue;
                }
                let lhs_dominates = lhs.state.cp >= rhs.state.cp
                    && lhs.state.durability >= rhs.state.durability
                    && lhs.state.quality >= rhs.state.quality
                    && lhs.action_count <= rhs.action_count
                    && lhs.duration <= rhs.duration;
                let rhs_dominates = rhs.state.cp >= lhs.state.cp
                    && rhs.state.durability >= lhs.state.durability
                    && rhs.state.quality >= lhs.state.quality
                    && rhs.action_count <= lhs.action_count
                    && rhs.duration <= lhs.duration;
                assert!(!lhs_dominates && !rhs_dominates);
            }
        }
    }
}
