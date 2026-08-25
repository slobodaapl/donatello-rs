use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex, OnceLock};

use raphael_sim::{Action, ActionError, Combo, Condition, Settings, SimulationState};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use strum::IntoEnumIterator;

use crate::utils::{ParetoFrontBuilder, ParetoValue};

pub const CONDITION_COUNT: usize = 11;
const OPTIMISTIC_INNER_QUIET_COUNT: usize = 11;
const OPTIMISTIC_STATE_COUNT: usize = OPTIMISTIC_INNER_QUIET_COUNT * 16;
type ProbabilityMasses = SmallVec<[f64; 4]>;
type OptimisticStepFrontiers = Box<[Box<[Box<[ParetoValue]>]>]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct OptimisticFrontierCacheKey {
    settings: Settings,
    required_quality: u16,
    max_steps: u8,
    supported_conditions: u16,
}

#[derive(Default)]
struct OptimisticFrontierCache {
    entry: Option<(OptimisticFrontierCacheKey, Arc<OptimisticStepFrontiers>)>,
}

impl OptimisticFrontierCache {
    fn get_or_build(
        &mut self,
        model: &StochasticModel,
        required_quality: u16,
    ) -> Arc<OptimisticStepFrontiers> {
        let key = OptimisticFrontierCacheKey {
            settings: model.settings,
            required_quality,
            max_steps: model.max_steps,
            supported_conditions: supported_condition_mask(model.condition_probabilities_bps),
        };
        if let Some((cached_key, frontiers)) = &self.entry
            && *cached_key == key
        {
            return Arc::clone(frontiers);
        }
        let optimistic_actions = Action::iter()
            .filter(|action| model.settings.allowed_actions.has(*action))
            .collect::<Vec<_>>();
        let supported_conditions = model
            .condition_probabilities_bps
            .iter()
            .enumerate()
            .filter(|(_, probability)| **probability != 0)
            .map(|(index, _)| condition_from_index(index))
            .collect::<Vec<_>>();
        let frontiers = Arc::new(optimistic_step_frontiers(
            model,
            required_quality,
            &optimistic_actions,
            &supported_conditions,
        ));
        self.entry = Some((key, Arc::clone(&frontiers)));
        frontiers
    }
}

static OPTIMISTIC_FRONTIER_CACHE: OnceLock<Mutex<OptimisticFrontierCache>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
struct StochasticPolicyCacheIdentity {
    model: StochasticModel,
    required_quality: u16,
    actions: Box<[Action]>,
}

#[derive(Default)]
pub struct StochasticPolicyCache {
    identity: Option<StochasticPolicyCacheIdentity>,
    action_outcomes: FxHashMap<(DecisionState, Action), Arc<[Weighted<PostDecisionState>]>>,
    condition_outcomes: FxHashMap<PostDecisionState, Arc<[Weighted<DecisionState>]>>,
    action_upper_bounds: FxHashMap<(DecisionState, Action), PolicyValue>,
    decision_upper_bounds: FxHashMap<DecisionState, PolicyValue>,
    post_decision_upper_bounds: FxHashMap<PostDecisionState, PolicyValue>,
}

impl StochasticPolicyCache {
    fn prepare(
        &mut self,
        model: StochasticModel,
        required_quality: u16,
        actions: &[Action],
        expansion_limit: usize,
    ) {
        let identity = StochasticPolicyCacheIdentity {
            model,
            required_quality,
            actions: actions.into(),
        };
        let retained_limit = expansion_limit.saturating_mul(8);
        if self.identity.as_ref() != Some(&identity) || self.entry_count() > retained_limit {
            *self = Self {
                identity: Some(identity),
                ..Self::default()
            };
        }
    }

    fn entry_count(&self) -> usize {
        self.action_outcomes.len()
            + self.condition_outcomes.len()
            + self.action_upper_bounds.len()
            + self.decision_upper_bounds.len()
            + self.post_decision_upper_bounds.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecisionState {
    pub simulation: SimulationState,
    pub condition: Condition,
    pub step: u8,
    pub decisions: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConditionTransition {
    Fixed(Condition),
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PostDecisionState {
    pub simulation: SimulationState,
    pub transition: ConditionTransition,
    pub step: u8,
    pub decisions: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PolicyFrontierState {
    Decision(DecisionState),
    PostDecision(PostDecisionState),
}

#[derive(Debug, Clone, Copy)]
struct PolicyFrontierEntry {
    priority: f64,
    sequence: u64,
    state: PolicyFrontierState,
}

impl PartialEq for PolicyFrontierEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority.to_bits() == other.priority.to_bits() && self.sequence == other.sequence
    }
}

impl Eq for PolicyFrontierEntry {}

impl PartialOrd for PolicyFrontierEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PolicyFrontierEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weighted<T> {
    pub value: T,
    pub probability: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StochasticModel {
    pub settings: Settings,
    pub condition_probabilities_bps: [u16; CONDITION_COUNT],
    pub max_steps: u8,
    pub max_decisions: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PolicyValue {
    pub full_quality_completion_probability: f64,
    pub completion_probability: f64,
    pub expected_quality: f64,
}

impl PolicyValue {
    fn scaled(self, probability: f64) -> Self {
        Self {
            full_quality_completion_probability: self.full_quality_completion_probability
                * probability,
            completion_probability: self.completion_probability * probability,
            expected_quality: self.expected_quality * probability,
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.full_quality_completion_probability += other.full_quality_completion_probability;
        self.completion_probability += other.completion_probability;
        self.expected_quality += other.expected_quality;
    }

    fn cmp_objective(self, other: Self) -> std::cmp::Ordering {
        self.full_quality_completion_probability
            .total_cmp(&other.full_quality_completion_probability)
            .then_with(|| {
                self.completion_probability
                    .total_cmp(&other.completion_probability)
            })
            .then_with(|| self.expected_quality.total_cmp(&other.expected_quality))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ProbabilityInterval {
    pub lower: f64,
    pub upper: f64,
}

impl ProbabilityInterval {
    fn exact(probability: f64) -> Self {
        Self {
            lower: probability,
            upper: probability,
        }
    }

    fn scaled(self, probability: f64) -> Self {
        Self {
            lower: self.lower * probability,
            upper: self.upper * probability,
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.lower += other.lower;
        self.upper += other.upper;
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StochasticSolveStats {
    pub decision_states: usize,
    pub post_decision_states: usize,
    pub policy_decision_states: usize,
    pub policy_post_decision_states: usize,
    pub pruned_actions: usize,
    pub pruned_policy_states: usize,
    pub deferred_policy_states: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticSolveOutcome {
    pub action: Action,
    pub value: PolicyValue,
    pub stats: StochasticSolveStats,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticPolicyImprovementOutcome {
    pub incumbent_action: Action,
    pub action: Action,
    pub incumbent_value: PolicyValue,
    pub value: PolicyValue,
    pub stats: StochasticSolveStats,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticFullQualityImprovementOutcome {
    pub incumbent_action: Action,
    pub action: Action,
    pub incumbent_probability: f64,
    pub probability: f64,
    pub stats: StochasticSolveStats,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticBoundedPolicyImprovementOutcome {
    pub incumbent_action: Action,
    pub action: Action,
    pub best_lower_action: Action,
    pub incumbent_interval: ProbabilityInterval,
    pub interval: ProbabilityInterval,
    pub best_lower_interval: ProbabilityInterval,
    pub improvement_proven: bool,
    pub optimality_proven: bool,
    pub stats: StochasticSolveStats,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BoundedDecisionValue {
    action: Option<Action>,
    interval: ProbabilityInterval,
}

#[derive(Debug)]
struct BoundedGraphAction {
    action: Action,
    successors: Vec<Weighted<usize>>,
}

#[derive(Debug)]
enum BoundedGraphEdges {
    Unexpanded,
    Terminal,
    Decision(Vec<BoundedGraphAction>),
    PostDecision(Vec<Weighted<usize>>),
}

#[derive(Debug)]
struct BoundedGraphNode {
    state: PolicyFrontierState,
    interval: ProbabilityInterval,
    edges: BoundedGraphEdges,
    parents: SmallVec<[usize; 4]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StochasticSolveError {
    TerminalRoot,
    NoLegalAction,
    ExpansionLimit,
}

pub struct StochasticPolicySolver {
    model: StochasticModel,
    required_quality: u16,
    actions: Box<[Action]>,
    expansion_limit: usize,
    decision_values: FxHashMap<DecisionState, PolicyValue>,
    post_decision_values: FxHashMap<PostDecisionState, PolicyValue>,
    policy_decision_values: FxHashMap<DecisionState, PolicyValue>,
    policy_post_decision_values: FxHashMap<PostDecisionState, PolicyValue>,
    policy_full_quality_decision_values: FxHashMap<DecisionState, f64>,
    policy_full_quality_post_decision_values: FxHashMap<PostDecisionState, f64>,
    bounded_decision_values: FxHashMap<DecisionState, BoundedDecisionValue>,
    bounded_post_decision_values: FxHashMap<PostDecisionState, ProbabilityInterval>,
    stats: StochasticSolveStats,
    optimistic_supported_step_frontiers: Arc<OptimisticStepFrontiers>,
    persistent_cache: RefCell<StochasticPolicyCache>,
}

impl StochasticPolicySolver {
    pub fn new(
        model: StochasticModel,
        required_quality: u16,
        actions: impl Into<Box<[Action]>>,
        expansion_limit: usize,
    ) -> Result<Self, String> {
        Self::new_with_cache(
            model,
            required_quality,
            actions,
            expansion_limit,
            StochasticPolicyCache::default(),
        )
    }

    pub fn new_with_cache(
        model: StochasticModel,
        required_quality: u16,
        actions: impl Into<Box<[Action]>>,
        expansion_limit: usize,
        mut persistent_cache: StochasticPolicyCache,
    ) -> Result<Self, String> {
        model.validate()?;
        if required_quality == 0 || required_quality > model.settings.max_quality {
            return Err(String::from(
                "required quality must be within 1..=max quality",
            ));
        }
        if expansion_limit == 0 {
            return Err(String::from("stochastic expansion limit must be positive"));
        }
        let actions = actions.into();
        persistent_cache.prepare(model, required_quality, &actions, expansion_limit);
        let optimistic_supported_step_frontiers = OPTIMISTIC_FRONTIER_CACHE
            .get_or_init(|| Mutex::new(OptimisticFrontierCache::default()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_or_build(&model, required_quality);
        Ok(Self {
            model,
            required_quality,
            actions,
            expansion_limit,
            decision_values: FxHashMap::default(),
            post_decision_values: FxHashMap::default(),
            policy_decision_values: FxHashMap::default(),
            policy_post_decision_values: FxHashMap::default(),
            policy_full_quality_decision_values: FxHashMap::default(),
            policy_full_quality_post_decision_values: FxHashMap::default(),
            bounded_decision_values: FxHashMap::default(),
            bounded_post_decision_values: FxHashMap::default(),
            stats: StochasticSolveStats::default(),
            optimistic_supported_step_frontiers,
            persistent_cache: RefCell::new(persistent_cache),
        })
    }

    pub fn into_cache(self) -> StochasticPolicyCache {
        self.persistent_cache.into_inner()
    }

    pub fn solve(
        &mut self,
        root: DecisionState,
    ) -> Result<StochasticSolveOutcome, StochasticSolveError> {
        self.solve_internal(root, None)
    }

    pub fn solve_with_policy(
        &mut self,
        root: DecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<StochasticSolveOutcome, StochasticSolveError> {
        self.solve_internal(root, Some(policy))
    }

    pub fn recursive_policy_action(&self, state: DecisionState) -> Option<Action> {
        self.bounded_decision_values
            .get(&state)
            .and_then(|value| value.action)
    }

    pub fn improve_policy(
        &mut self,
        root: DecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<StochasticPolicyImprovementOutcome, StochasticSolveError> {
        if self
            .terminal_value(root.simulation, root.step, root.decisions)
            .is_some()
        {
            return Err(StochasticSolveError::TerminalRoot);
        }
        self.clear_values();
        let incumbent_action = self
            .policy_action(root, policy)
            .ok_or(StochasticSolveError::NoLegalAction)?;
        let incumbent_value = self.policy_decision_value(root, policy)?;
        let mut best_action = incumbent_action;
        let mut best_value = incumbent_value;
        for (action, upper_bound) in self.ordered_legal_actions(root) {
            if action == incumbent_action {
                continue;
            }
            if !upper_bound_may_improve(upper_bound, best_value) {
                self.stats.pruned_actions += 1;
                continue;
            }
            let value = self.policy_action_value(root, action, policy)?;
            if value.cmp_objective(best_value).is_gt() {
                best_action = action;
                best_value = value;
            }
        }
        Ok(StochasticPolicyImprovementOutcome {
            incumbent_action,
            action: best_action,
            incumbent_value,
            value: best_value,
            stats: self.stats,
        })
    }

    pub fn improve_policy_full_quality(
        &mut self,
        root: DecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<StochasticFullQualityImprovementOutcome, StochasticSolveError> {
        if self
            .terminal_value(root.simulation, root.step, root.decisions)
            .is_some()
        {
            return Err(StochasticSolveError::TerminalRoot);
        }
        self.clear_values();
        let incumbent_action = self
            .policy_action(root, policy)
            .ok_or(StochasticSolveError::NoLegalAction)?;
        let incumbent_probability = self.policy_full_quality_decision_probability(root, policy)?;
        let mut best_action = incumbent_action;
        let mut best_probability = incumbent_probability;
        for (action, upper_bound) in self.ordered_legal_actions(root) {
            if action == incumbent_action {
                continue;
            }
            if upper_bound.full_quality_completion_probability + 1e-12 < best_probability {
                self.stats.pruned_actions += 1;
                continue;
            }
            let probability = self.policy_full_quality_action_probability(root, action, policy)?;
            if probability > best_probability + 1e-12 {
                best_action = action;
                best_probability = probability;
            }
        }
        Ok(StochasticFullQualityImprovementOutcome {
            incumbent_action,
            action: best_action,
            incumbent_probability,
            probability: best_probability,
            stats: self.stats,
        })
    }

    pub fn improve_policy_full_quality_bounded(
        &mut self,
        root: DecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<StochasticBoundedPolicyImprovementOutcome, StochasticSolveError> {
        if self
            .terminal_value(root.simulation, root.step, root.decisions)
            .is_some()
        {
            return Err(StochasticSolveError::TerminalRoot);
        }
        self.clear_values();
        let incumbent_action = self
            .policy_action(root, policy)
            .ok_or(StochasticSolveError::NoLegalAction)?;
        let mut root_actions = vec![incumbent_action];
        root_actions.extend(
            self.ordered_legal_actions(root)
                .into_iter()
                .map(|(action, _)| action)
                .filter(|action| *action != incumbent_action),
        );
        let intervals = root_actions
            .iter()
            .copied()
            .zip(self.policy_full_quality_action_intervals(root, &root_actions, policy)?)
            .collect::<Vec<_>>();
        let incumbent_interval = intervals[0].1;
        let (best_action, best_interval) = intervals
            .iter()
            .copied()
            .max_by(|left, right| {
                left.1.lower.total_cmp(&right.1.lower).then_with(|| {
                    left.0
                        .eq(&incumbent_action)
                        .cmp(&right.0.eq(&incumbent_action))
                })
            })
            .expect("incumbent interval must exist");
        let improvement_proven = best_action != incumbent_action
            && best_interval.lower > incumbent_interval.upper + 1e-12;
        let (action, interval) = if improvement_proven {
            (best_action, best_interval)
        } else {
            (incumbent_action, incumbent_interval)
        };
        let optimality_proven = intervals.iter().all(|(other_action, other_interval)| {
            *other_action == action || other_interval.upper <= interval.lower + 1e-12
        });
        Ok(StochasticBoundedPolicyImprovementOutcome {
            incumbent_action,
            action,
            best_lower_action: best_action,
            incumbent_interval,
            interval,
            best_lower_interval: best_interval,
            improvement_proven,
            optimality_proven,
            stats: self.stats,
        })
    }

    pub fn improve_policy_full_quality_recursive_bounded(
        &mut self,
        root: DecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<StochasticBoundedPolicyImprovementOutcome, StochasticSolveError> {
        if self
            .terminal_value(root.simulation, root.step, root.decisions)
            .is_some()
        {
            return Err(StochasticSolveError::TerminalRoot);
        }
        self.clear_values();
        let incumbent_action = self
            .policy_action(root, policy)
            .ok_or(StochasticSolveError::NoLegalAction)?;
        let incumbent_interval = self
            .policy_full_quality_action_intervals(root, &[incumbent_action], policy)?
            .into_iter()
            .next()
            .expect("incumbent interval must exist");
        let fixed_policy_stats = self.stats;

        self.clear_values();
        let mut intervals = vec![(incumbent_action, incumbent_interval)];
        for (action, upper_bound) in self.ordered_legal_actions(root) {
            if action == incumbent_action {
                continue;
            }
            if upper_bound.full_quality_completion_probability <= incumbent_interval.lower + 1e-12 {
                self.stats.pruned_actions += 1;
                continue;
            }
            intervals.push((
                action,
                self.bounded_recursive_action_interval(root, action, policy)?,
            ));
        }

        let (best_lower_action, best_lower_interval) = intervals
            .iter()
            .copied()
            .max_by(|left, right| {
                left.1.lower.total_cmp(&right.1.lower).then_with(|| {
                    left.0
                        .eq(&incumbent_action)
                        .cmp(&right.0.eq(&incumbent_action))
                })
            })
            .expect("incumbent interval must exist");
        let improvement_proven = best_lower_action != incumbent_action
            && best_lower_interval.lower > incumbent_interval.upper + 1e-12;
        let (action, interval) = if improvement_proven {
            (best_lower_action, best_lower_interval)
        } else {
            (incumbent_action, incumbent_interval)
        };
        let optimality_proven = intervals.iter().all(|(other_action, other_interval)| {
            *other_action == action || other_interval.upper <= interval.lower + 1e-12
        });
        let stats = add_solve_stats(fixed_policy_stats, self.stats);
        Ok(StochasticBoundedPolicyImprovementOutcome {
            incumbent_action,
            action,
            best_lower_action,
            incumbent_interval,
            interval,
            best_lower_interval,
            improvement_proven,
            optimality_proven,
            stats,
        })
    }

    pub fn improve_policy_full_quality_best_first_bounded(
        &mut self,
        root: DecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<StochasticBoundedPolicyImprovementOutcome, StochasticSolveError> {
        if self
            .terminal_value(root.simulation, root.step, root.decisions)
            .is_some()
        {
            return Err(StochasticSolveError::TerminalRoot);
        }
        self.clear_values();
        let incumbent_action = self
            .policy_action(root, policy)
            .ok_or(StochasticSolveError::NoLegalAction)?;
        let incumbent_interval = self
            .policy_full_quality_action_intervals(root, &[incumbent_action], policy)?
            .into_iter()
            .next()
            .expect("incumbent interval must exist");
        let fixed_policy_stats = self.stats;

        self.clear_values();
        let root_state = PolicyFrontierState::Decision(root);
        let mut nodes = Vec::new();
        let mut node_indices = FxHashMap::default();
        let root_index = self.bounded_graph_node(root_state, &mut nodes, &mut node_indices, policy);
        self.expand_bounded_graph_node(root_index, &mut nodes, &mut node_indices, policy)?;

        const EXPANSION_BATCH: usize = 4_096;
        loop {
            let frontier = self.bounded_graph_frontier(
                root_index,
                incumbent_action,
                incumbent_interval.lower,
                &nodes,
            );
            if frontier.is_empty() {
                break;
            }
            if self.total_states() >= self.expansion_limit {
                self.stats.deferred_policy_states += frontier.len();
                break;
            }
            let mut expanded = 0;
            for (node_index, _) in frontier.into_iter().take(EXPANSION_BATCH) {
                if self.total_states() >= self.expansion_limit {
                    break;
                }
                if self.expand_bounded_graph_node(
                    node_index,
                    &mut nodes,
                    &mut node_indices,
                    policy,
                )? {
                    self.propagate_bounded_graph_value(node_index, &mut nodes);
                    expanded += 1;
                }
            }
            if expanded == 0 {
                break;
            }
        }

        let mut intervals = self.bounded_graph_root_action_intervals(root_index, &nodes);
        if let Some((_, interval)) = intervals
            .iter_mut()
            .find(|(action, _)| *action == incumbent_action)
        {
            *interval = incumbent_interval;
        } else {
            intervals.insert(0, (incumbent_action, incumbent_interval));
        }
        let (best_lower_action, best_lower_interval) = intervals
            .iter()
            .copied()
            .max_by(|left, right| {
                left.1.lower.total_cmp(&right.1.lower).then_with(|| {
                    left.0
                        .eq(&incumbent_action)
                        .cmp(&right.0.eq(&incumbent_action))
                })
            })
            .expect("incumbent interval must exist");
        let improvement_proven = best_lower_action != incumbent_action
            && best_lower_interval.lower > incumbent_interval.upper + 1e-12;
        let (action, interval) = if improvement_proven {
            (best_lower_action, best_lower_interval)
        } else {
            (incumbent_action, incumbent_interval)
        };
        let optimality_proven = intervals.iter().all(|(other_action, other_interval)| {
            *other_action == action || other_interval.upper <= interval.lower + 1e-12
        });
        let stats = add_solve_stats(fixed_policy_stats, self.stats);
        Ok(StochasticBoundedPolicyImprovementOutcome {
            incumbent_action,
            action,
            best_lower_action,
            incumbent_interval,
            interval,
            best_lower_interval,
            improvement_proven,
            optimality_proven,
            stats,
        })
    }

    fn solve_internal(
        &mut self,
        root: DecisionState,
        policy: Option<&dyn Fn(DecisionState) -> Option<Action>>,
    ) -> Result<StochasticSolveOutcome, StochasticSolveError> {
        if self
            .terminal_value(root.simulation, root.step, root.decisions)
            .is_some()
        {
            return Err(StochasticSolveError::TerminalRoot);
        }
        self.clear_values();
        let mut best = if let Some(policy) = policy {
            self.policy_action(root, policy)
                .map(|action| {
                    self.policy_decision_value(root, policy)
                        .map(|value| (action, value))
                })
                .transpose()?
        } else {
            None
        };
        for (action, upper_bound) in self.ordered_legal_actions(root) {
            if best.is_some_and(|(_, best_value)| !upper_bound_may_improve(upper_bound, best_value))
            {
                self.stats.pruned_actions += 1;
                continue;
            }
            let value = match self.action_value(root, action, policy) {
                Ok(value) => value,
                Err(StochasticSolveError::NoLegalAction) => continue,
                Err(error) => return Err(error),
            };
            if best.is_none_or(|(_, best_value): (Action, PolicyValue)| {
                value.cmp_objective(best_value).is_gt()
            }) {
                best = Some((action, value));
            }
        }
        let (action, value) = best.ok_or(StochasticSolveError::NoLegalAction)?;
        Ok(StochasticSolveOutcome {
            action,
            value,
            stats: self.stats,
        })
    }

    fn decision_value(
        &mut self,
        state: DecisionState,
        policy: Option<&dyn Fn(DecisionState) -> Option<Action>>,
    ) -> Result<PolicyValue, StochasticSolveError> {
        if let Some(value) = self.terminal_value(state.simulation, state.step, state.decisions) {
            return Ok(value);
        }
        if let Some(value) = self.decision_values.get(&state) {
            return Ok(*value);
        }
        self.reserve_decision_state()?;
        let mut best = policy.and_then(|_| self.policy_decision_values.get(&state).copied());
        for (action, upper_bound) in self.ordered_legal_actions(state) {
            if best.is_some_and(|best_value| !upper_bound_may_improve(upper_bound, best_value)) {
                self.stats.pruned_actions += 1;
                continue;
            }
            let value = match self.action_value(state, action, policy) {
                Ok(value) => value,
                Err(StochasticSolveError::NoLegalAction) => continue,
                Err(error) => return Err(error),
            };
            if best.is_none_or(|best_value: PolicyValue| value.cmp_objective(best_value).is_gt()) {
                best = Some(value);
            }
        }
        let value = best.unwrap_or_else(|| self.failed_value(state.simulation));
        self.decision_values.insert(state, value);
        Ok(value)
    }

    fn ordered_legal_actions(&self, state: DecisionState) -> Vec<(Action, PolicyValue)> {
        let mut actions = self
            .actions
            .iter()
            .filter_map(|action| {
                self.action_upper_bound(state, *action)
                    .ok()
                    .map(|upper_bound| (*action, upper_bound))
            })
            .collect::<Vec<_>>();
        actions.sort_by(|left, right| right.1.cmp_objective(left.1));
        actions
    }

    fn action_upper_bound(
        &self,
        state: DecisionState,
        action: Action,
    ) -> Result<PolicyValue, ActionError> {
        if let Some(value) = self
            .persistent_cache
            .borrow()
            .action_upper_bounds
            .get(&(state, action))
            .copied()
        {
            return Ok(value);
        }
        let mut value = PolicyValue::default();
        for outcome in self.post_decision_outcomes_cached(state, action)?.iter() {
            value.add_assign(
                self.post_decision_upper_bound(outcome.value)
                    .scaled(outcome.probability),
            );
        }
        self.persistent_cache
            .borrow_mut()
            .action_upper_bounds
            .insert((state, action), value);
        Ok(value)
    }

    fn post_decision_upper_bound(&self, post: PostDecisionState) -> PolicyValue {
        if let Some(value) = self
            .persistent_cache
            .borrow()
            .post_decision_upper_bounds
            .get(&post)
            .copied()
        {
            return value;
        }
        let value = match post.transition {
            ConditionTransition::Random => self.state_upper_bound(
                post.simulation,
                post.step,
                post.decisions,
                &self.optimistic_supported_step_frontiers,
            ),
            ConditionTransition::Fixed(condition) => self.decision_upper_bound(DecisionState {
                simulation: post.simulation,
                condition,
                step: post.step,
                decisions: post.decisions,
            }),
        };
        self.persistent_cache
            .borrow_mut()
            .post_decision_upper_bounds
            .insert(post, value);
        value
    }

    fn decision_upper_bound(&self, state: DecisionState) -> PolicyValue {
        if let Some(value) = self
            .persistent_cache
            .borrow()
            .decision_upper_bounds
            .get(&state)
            .copied()
        {
            return value;
        }
        if let Some(value) = self.terminal_value(state.simulation, state.step, state.decisions) {
            return value;
        }
        let value = if self.model.condition_probabilities_bps[condition_index(state.condition)] != 0
        {
            self.state_upper_bound(
                state.simulation,
                state.step,
                state.decisions,
                &self.optimistic_supported_step_frontiers,
            )
        } else {
            let mut upper = PolicyValue::default();
            for action in
                Action::iter().filter(|action| self.model.settings.allowed_actions.has(*action))
            {
                let Ok(action_upper) = self.action_upper_bound(state, action) else {
                    continue;
                };
                upper.full_quality_completion_probability = upper
                    .full_quality_completion_probability
                    .max(action_upper.full_quality_completion_probability);
                upper.completion_probability = upper
                    .completion_probability
                    .max(action_upper.completion_probability);
                upper.expected_quality = upper.expected_quality.max(action_upper.expected_quality);
            }
            upper
        };
        self.persistent_cache
            .borrow_mut()
            .decision_upper_bounds
            .insert(state, value);
        value
    }

    fn post_decision_outcomes_cached(
        &self,
        state: DecisionState,
        action: Action,
    ) -> Result<Arc<[Weighted<PostDecisionState>]>, ActionError> {
        if let Some(outcomes) = self
            .persistent_cache
            .borrow()
            .action_outcomes
            .get(&(state, action))
            .cloned()
        {
            return Ok(outcomes);
        }
        let outcomes: Arc<[_]> = self.model.post_decision_outcomes(state, action)?.into();
        self.persistent_cache
            .borrow_mut()
            .action_outcomes
            .insert((state, action), Arc::clone(&outcomes));
        Ok(outcomes)
    }

    fn condition_outcomes_cached(&self, post: PostDecisionState) -> Arc<[Weighted<DecisionState>]> {
        if let Some(outcomes) = self
            .persistent_cache
            .borrow()
            .condition_outcomes
            .get(&post)
            .cloned()
        {
            return outcomes;
        }
        let outcomes: Arc<[_]> = self.model.condition_outcomes(post).into();
        self.persistent_cache
            .borrow_mut()
            .condition_outcomes
            .insert(post, Arc::clone(&outcomes));
        outcomes
    }

    fn state_upper_bound(
        &self,
        simulation: SimulationState,
        step: u8,
        decisions: u8,
        frontiers: &[Box<[Box<[ParetoValue]>]>],
    ) -> PolicyValue {
        if let Some(value) = self.terminal_value(simulation, step, decisions) {
            return value;
        }
        let advancing_steps = self
            .model
            .max_steps
            .saturating_sub(step)
            .min(self.model.max_decisions.saturating_sub(decisions));
        let remaining_progress = self
            .model
            .settings
            .max_progress
            .saturating_sub(simulation.progress);
        let remaining_quality = self.required_quality.saturating_sub(simulation.quality);
        let optimistic_state = optimistic_state_index(
            simulation.effects.inner_quiet(),
            simulation.effects.great_strides() != 0,
            simulation.effects.innovation() != 0,
            simulation.effects.muscle_memory() != 0,
            simulation.effects.veneration() != 0,
        );
        let frontier = &frontiers[usize::from(advancing_steps)][optimistic_state];
        let completion_possible = frontier
            .iter()
            .any(|value| value.progress >= remaining_progress);
        let full_quality_completion_possible = frontier.iter().any(|value| {
            value.progress >= remaining_progress && value.quality >= remaining_quality
        });
        let additional_quality = frontier
            .iter()
            .map(|value| value.quality)
            .max()
            .unwrap_or(0);
        let quality_upper_bound = simulation.quality.saturating_add(additional_quality);
        PolicyValue {
            full_quality_completion_probability: f64::from(full_quality_completion_possible),
            completion_probability: f64::from(completion_possible),
            expected_quality: f64::from(quality_upper_bound.min(self.required_quality))
                / f64::from(self.required_quality),
        }
    }

    fn action_value(
        &mut self,
        state: DecisionState,
        action: Action,
        policy: Option<&dyn Fn(DecisionState) -> Option<Action>>,
    ) -> Result<PolicyValue, StochasticSolveError> {
        let outcomes = self
            .post_decision_outcomes_cached(state, action)
            .map_err(|_| StochasticSolveError::NoLegalAction)?;
        let mut value = PolicyValue::default();
        for outcome in outcomes.iter() {
            let branch = self.post_decision_value(outcome.value, policy)?;
            value.add_assign(branch.scaled(outcome.probability));
        }
        Ok(value)
    }

    fn post_decision_value(
        &mut self,
        post: PostDecisionState,
        policy: Option<&dyn Fn(DecisionState) -> Option<Action>>,
    ) -> Result<PolicyValue, StochasticSolveError> {
        if let Some(value) = self.terminal_value(post.simulation, post.step, post.decisions) {
            return Ok(value);
        }
        if let Some(value) = self.post_decision_values.get(&post) {
            return Ok(*value);
        }
        self.reserve_post_decision_state()?;
        let mut value = PolicyValue::default();
        for outcome in self.condition_outcomes_cached(post).iter() {
            let branch = self.decision_value(outcome.value, policy)?;
            value.add_assign(branch.scaled(outcome.probability));
        }
        self.post_decision_values.insert(post, value);
        Ok(value)
    }

    fn bounded_recursive_decision_value(
        &mut self,
        state: DecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<BoundedDecisionValue, StochasticSolveError> {
        if let Some(value) = self.terminal_value(state.simulation, state.step, state.decisions) {
            return Ok(BoundedDecisionValue {
                action: None,
                interval: ProbabilityInterval::exact(value.full_quality_completion_probability),
            });
        }
        if let Some(value) = self.bounded_decision_values.get(&state) {
            return Ok(*value);
        }
        let state_upper = self
            .decision_upper_bound(state)
            .full_quality_completion_probability;
        if state_upper == 0.0 {
            self.stats.pruned_policy_states += 1;
            let value = BoundedDecisionValue {
                action: None,
                interval: ProbabilityInterval::exact(0.0),
            };
            self.bounded_decision_values.insert(state, value);
            return Ok(value);
        }
        if self.total_states() >= self.expansion_limit {
            self.stats.deferred_policy_states += 1;
            let lower = self.one_step_policy_decision_lower_bound(state, policy);
            return Ok(BoundedDecisionValue {
                action: self.policy_action(state, policy),
                interval: ProbabilityInterval {
                    lower,
                    upper: state_upper,
                },
            });
        }
        self.reserve_decision_state()?;

        let incumbent_action = self.policy_action(state, policy);
        let mut actions = Vec::with_capacity(self.actions.len() + 1);
        if let Some(action) = incumbent_action {
            actions.push((
                action,
                self.action_upper_bound(state, action)
                    .map_err(|_| StochasticSolveError::NoLegalAction)?,
            ));
        }
        for action in self.ordered_legal_actions(state) {
            if !actions.iter().any(|(existing, _)| *existing == action.0) {
                actions.push(action);
            }
        }

        let mut selected = None::<(Action, ProbabilityInterval)>;
        let mut optimal_upper = 0.0_f64;
        for (action, upper_bound) in actions {
            let action_upper = upper_bound.full_quality_completion_probability;
            if selected.is_some_and(|(_, interval)| action_upper <= interval.lower + 1e-12) {
                self.stats.pruned_actions += 1;
                continue;
            }
            let interval = self.bounded_recursive_action_interval(state, action, policy)?;
            optimal_upper = optimal_upper.max(interval.upper);
            match selected {
                None => selected = Some((action, interval)),
                Some((_, incumbent_interval))
                    if interval.lower > incumbent_interval.lower + 1e-12 =>
                {
                    selected = Some((action, interval));
                }
                _ => {}
            }
        }

        let value = if let Some((action, selected_interval)) = selected {
            BoundedDecisionValue {
                action: Some(action),
                interval: ProbabilityInterval {
                    lower: selected_interval.lower,
                    upper: optimal_upper.max(selected_interval.upper),
                },
            }
        } else {
            BoundedDecisionValue {
                action: None,
                interval: ProbabilityInterval::exact(0.0),
            }
        };
        self.bounded_decision_values.insert(state, value);
        Ok(value)
    }

    fn bounded_recursive_action_interval(
        &mut self,
        state: DecisionState,
        action: Action,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<ProbabilityInterval, StochasticSolveError> {
        let mut interval = ProbabilityInterval::exact(0.0);
        for outcome in self
            .post_decision_outcomes_cached(state, action)
            .map_err(|_| StochasticSolveError::NoLegalAction)?
            .iter()
        {
            interval.add_assign(
                self.bounded_recursive_post_decision_value(outcome.value, policy)?
                    .scaled(outcome.probability),
            );
        }
        Ok(interval)
    }

    fn bounded_recursive_post_decision_value(
        &mut self,
        post: PostDecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<ProbabilityInterval, StochasticSolveError> {
        if let Some(value) = self.terminal_value(post.simulation, post.step, post.decisions) {
            return Ok(ProbabilityInterval::exact(
                value.full_quality_completion_probability,
            ));
        }
        if let Some(interval) = self.bounded_post_decision_values.get(&post) {
            return Ok(*interval);
        }
        let post_upper = self
            .post_decision_upper_bound(post)
            .full_quality_completion_probability;
        if post_upper == 0.0 {
            self.stats.pruned_policy_states += 1;
            let interval = ProbabilityInterval::exact(0.0);
            self.bounded_post_decision_values.insert(post, interval);
            return Ok(interval);
        }
        if self.total_states() >= self.expansion_limit {
            self.stats.deferred_policy_states += 1;
            let upper = self
                .condition_outcomes_cached(post)
                .iter()
                .map(|outcome| {
                    outcome.probability
                        * self
                            .decision_upper_bound(outcome.value)
                            .full_quality_completion_probability
                })
                .sum();
            let lower = self.one_step_policy_post_decision_lower_bound(post, policy);
            return Ok(ProbabilityInterval { lower, upper });
        }
        self.reserve_post_decision_state()?;

        let mut interval = ProbabilityInterval::exact(0.0);
        for outcome in self.condition_outcomes_cached(post).iter() {
            interval.add_assign(
                self.bounded_recursive_decision_value(outcome.value, policy)?
                    .interval
                    .scaled(outcome.probability),
            );
        }
        self.bounded_post_decision_values.insert(post, interval);
        Ok(interval)
    }

    fn one_step_policy_decision_lower_bound(
        &self,
        state: DecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> f64 {
        let Some(action) = self.policy_action(state, policy) else {
            return 0.0;
        };
        let Ok(outcomes) = self.post_decision_outcomes_cached(state, action) else {
            return 0.0;
        };
        let mut lower = 0.0;
        for outcome in outcomes.iter() {
            let success = self
                .terminal_value(
                    outcome.value.simulation,
                    outcome.value.step,
                    outcome.value.decisions,
                )
                .is_some_and(|value| value.full_quality_completion_probability == 1.0);
            lower += outcome.probability * f64::from(success);
        }
        lower
    }

    fn one_step_policy_post_decision_lower_bound(
        &self,
        post: PostDecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> f64 {
        let mut lower = 0.0;
        for outcome in self.condition_outcomes_cached(post).iter() {
            lower += outcome.probability
                * self.one_step_policy_decision_lower_bound(outcome.value, policy);
        }
        lower
    }

    fn bounded_graph_node(
        &self,
        state: PolicyFrontierState,
        nodes: &mut Vec<BoundedGraphNode>,
        node_indices: &mut FxHashMap<PolicyFrontierState, usize>,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> usize {
        if let Some(index) = node_indices.get(&state) {
            return *index;
        }
        let (simulation, step, decisions, optimistic) = match state {
            PolicyFrontierState::Decision(state) => (
                state.simulation,
                state.step,
                state.decisions,
                self.decision_upper_bound(state),
            ),
            PolicyFrontierState::PostDecision(post) => (
                post.simulation,
                post.step,
                post.decisions,
                self.post_decision_upper_bound(post),
            ),
        };
        let terminal = self.terminal_value(simulation, step, decisions);
        let interval = match terminal {
            Some(value) => ProbabilityInterval::exact(value.full_quality_completion_probability),
            None => ProbabilityInterval {
                lower: match state {
                    PolicyFrontierState::Decision(state) => {
                        self.one_step_policy_decision_lower_bound(state, policy)
                    }
                    PolicyFrontierState::PostDecision(post) => {
                        self.one_step_policy_post_decision_lower_bound(post, policy)
                    }
                },
                upper: optimistic.full_quality_completion_probability,
            },
        };
        let index = nodes.len();
        nodes.push(BoundedGraphNode {
            state,
            interval,
            edges: if terminal.is_some() {
                BoundedGraphEdges::Terminal
            } else {
                BoundedGraphEdges::Unexpanded
            },
            parents: SmallVec::new(),
        });
        node_indices.insert(state, index);
        index
    }

    fn expand_bounded_graph_node(
        &mut self,
        node_index: usize,
        nodes: &mut Vec<BoundedGraphNode>,
        node_indices: &mut FxHashMap<PolicyFrontierState, usize>,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<bool, StochasticSolveError> {
        if !matches!(nodes[node_index].edges, BoundedGraphEdges::Unexpanded) {
            return Ok(false);
        }
        let state = nodes[node_index].state;
        let edges = match state {
            PolicyFrontierState::Decision(state) => {
                self.reserve_decision_state()?;
                let incumbent_action = self.policy_action(state, policy);
                let mut actions = Vec::with_capacity(self.actions.len() + 1);
                if let Some(action) = incumbent_action {
                    actions.push(action);
                }
                for (action, _) in self.ordered_legal_actions(state) {
                    if !actions.contains(&action) {
                        actions.push(action);
                    }
                }
                let mut graph_actions = Vec::with_capacity(actions.len());
                for action in actions {
                    let mut successors = Vec::new();
                    for outcome in self
                        .post_decision_outcomes_cached(state, action)
                        .map_err(|_| StochasticSolveError::NoLegalAction)?
                        .iter()
                    {
                        let child = self.bounded_graph_node(
                            PolicyFrontierState::PostDecision(outcome.value),
                            nodes,
                            node_indices,
                            policy,
                        );
                        if !nodes[child].parents.contains(&node_index) {
                            nodes[child].parents.push(node_index);
                        }
                        successors.push(Weighted {
                            value: child,
                            probability: outcome.probability,
                        });
                    }
                    graph_actions.push(BoundedGraphAction { action, successors });
                }
                BoundedGraphEdges::Decision(graph_actions)
            }
            PolicyFrontierState::PostDecision(post) => {
                self.reserve_post_decision_state()?;
                let mut successors = Vec::new();
                for outcome in self.condition_outcomes_cached(post).iter() {
                    let child = self.bounded_graph_node(
                        PolicyFrontierState::Decision(outcome.value),
                        nodes,
                        node_indices,
                        policy,
                    );
                    if !nodes[child].parents.contains(&node_index) {
                        nodes[child].parents.push(node_index);
                    }
                    successors.push(Weighted {
                        value: child,
                        probability: outcome.probability,
                    });
                }
                BoundedGraphEdges::PostDecision(successors)
            }
        };
        nodes[node_index].edges = edges;
        recompute_bounded_graph_node(node_index, nodes);
        Ok(true)
    }

    fn propagate_bounded_graph_value(&self, node_index: usize, nodes: &mut [BoundedGraphNode]) {
        let mut pending = nodes[node_index].parents.to_vec();
        while let Some(parent) = pending.pop() {
            if recompute_bounded_graph_node(parent, nodes) {
                pending.extend(nodes[parent].parents.iter().copied());
            }
        }
    }

    fn bounded_graph_frontier(
        &self,
        root_index: usize,
        incumbent_action: Action,
        incumbent_lower: f64,
        nodes: &[BoundedGraphNode],
    ) -> Vec<(usize, f64)> {
        let BoundedGraphEdges::Decision(root_actions) = &nodes[root_index].edges else {
            return Vec::new();
        };
        let mut masses = FxHashMap::<usize, f64>::default();
        let mut queue = BinaryHeap::<Reverse<(u16, usize)>>::new();
        for action in root_actions.iter().filter(|action| {
            action.action != incumbent_action
                && bounded_graph_action_interval(action, nodes).upper > incumbent_lower + 1e-12
        }) {
            for successor in &action.successors {
                enqueue_bounded_graph_mass(
                    successor.value,
                    successor.probability,
                    nodes,
                    &mut masses,
                    &mut queue,
                );
            }
        }

        let mut frontier = FxHashMap::<usize, f64>::default();
        while let Some(Reverse((_, node_index))) = queue.pop() {
            let Some(mass) = masses.remove(&node_index) else {
                continue;
            };
            match &nodes[node_index].edges {
                BoundedGraphEdges::Unexpanded => {
                    *frontier.entry(node_index).or_default() += mass;
                }
                BoundedGraphEdges::Terminal => {}
                BoundedGraphEdges::PostDecision(successors) => {
                    for successor in successors {
                        enqueue_bounded_graph_mass(
                            successor.value,
                            mass * successor.probability,
                            nodes,
                            &mut masses,
                            &mut queue,
                        );
                    }
                }
                BoundedGraphEdges::Decision(actions) => {
                    let Some(first_action) = actions.first() else {
                        continue;
                    };
                    let mut upper_action = first_action;
                    let mut upper_interval = bounded_graph_action_interval(upper_action, nodes);
                    let mut lower_action = first_action;
                    let mut lower_interval = upper_interval;
                    for candidate in &actions[1..] {
                        let candidate_interval = bounded_graph_action_interval(candidate, nodes);
                        if candidate_interval
                            .upper
                            .total_cmp(&upper_interval.upper)
                            .then_with(|| candidate_interval.lower.total_cmp(&upper_interval.lower))
                            .is_gt()
                        {
                            upper_action = candidate;
                            upper_interval = candidate_interval;
                        }
                        if candidate_interval
                            .lower
                            .total_cmp(&lower_interval.lower)
                            .then_with(|| candidate_interval.upper.total_cmp(&lower_interval.upper))
                            .is_gt()
                        {
                            lower_action = candidate;
                            lower_interval = candidate_interval;
                        }
                    }
                    for successor in &upper_action.successors {
                        enqueue_bounded_graph_mass(
                            successor.value,
                            mass * successor.probability,
                            nodes,
                            &mut masses,
                            &mut queue,
                        );
                    }
                    if lower_action.action != upper_action.action {
                        for successor in &lower_action.successors {
                            enqueue_bounded_graph_mass(
                                successor.value,
                                mass * successor.probability,
                                nodes,
                                &mut masses,
                                &mut queue,
                            );
                        }
                    }
                }
            }
        }

        let mut frontier = frontier
            .into_iter()
            .filter_map(|(node_index, mass)| {
                let interval = nodes[node_index].interval;
                let priority = mass * (interval.upper - interval.lower);
                (priority > 0.0).then_some((node_index, priority))
            })
            .collect::<Vec<_>>();
        frontier.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        frontier
    }

    fn bounded_graph_root_action_intervals(
        &self,
        root_index: usize,
        nodes: &[BoundedGraphNode],
    ) -> Vec<(Action, ProbabilityInterval)> {
        let BoundedGraphEdges::Decision(actions) = &nodes[root_index].edges else {
            return Vec::new();
        };
        actions
            .iter()
            .map(|action| (action.action, bounded_graph_action_interval(action, nodes)))
            .collect()
    }

    fn policy_action(
        &self,
        state: DecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Option<Action> {
        policy(state).filter(|action| self.post_decision_outcomes_cached(state, *action).is_ok())
    }

    fn policy_decision_value(
        &mut self,
        state: DecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<PolicyValue, StochasticSolveError> {
        if let Some(value) = self.terminal_value(state.simulation, state.step, state.decisions) {
            return Ok(value);
        }
        if let Some(value) = self.policy_decision_values.get(&state) {
            return Ok(*value);
        }
        self.reserve_policy_decision_state()?;
        let value = if let Some(action) = self.policy_action(state, policy) {
            self.policy_action_value(state, action, policy)?
        } else {
            self.failed_value(state.simulation)
        };
        self.policy_decision_values.insert(state, value);
        Ok(value)
    }

    fn policy_action_value(
        &mut self,
        state: DecisionState,
        action: Action,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<PolicyValue, StochasticSolveError> {
        let mut value = PolicyValue::default();
        for outcome in self
            .post_decision_outcomes_cached(state, action)
            .map_err(|_| StochasticSolveError::NoLegalAction)?
            .iter()
        {
            let branch = self.policy_post_decision_value(outcome.value, policy)?;
            value.add_assign(branch.scaled(outcome.probability));
        }
        Ok(value)
    }

    fn policy_post_decision_value(
        &mut self,
        post: PostDecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<PolicyValue, StochasticSolveError> {
        if let Some(value) = self.terminal_value(post.simulation, post.step, post.decisions) {
            return Ok(value);
        }
        if let Some(value) = self.policy_post_decision_values.get(&post) {
            return Ok(*value);
        }
        self.reserve_policy_post_decision_state()?;
        let mut value = PolicyValue::default();
        for outcome in self.condition_outcomes_cached(post).iter() {
            let branch = self.policy_decision_value(outcome.value, policy)?;
            value.add_assign(branch.scaled(outcome.probability));
        }
        self.policy_post_decision_values.insert(post, value);
        Ok(value)
    }

    fn policy_full_quality_decision_probability(
        &mut self,
        state: DecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<f64, StochasticSolveError> {
        if let Some(value) = self.terminal_value(state.simulation, state.step, state.decisions) {
            return Ok(value.full_quality_completion_probability);
        }
        if let Some(probability) = self.policy_full_quality_decision_values.get(&state) {
            return Ok(*probability);
        }
        if self
            .decision_upper_bound(state)
            .full_quality_completion_probability
            == 0.0
        {
            self.stats.pruned_policy_states += 1;
            self.policy_full_quality_decision_values.insert(state, 0.0);
            return Ok(0.0);
        }
        let probability = if let Some(action) = self.policy_action(state, policy) {
            if self
                .action_upper_bound(state, action)
                .map_err(|_| StochasticSolveError::NoLegalAction)?
                .full_quality_completion_probability
                == 0.0
            {
                self.stats.pruned_policy_states += 1;
                0.0
            } else {
                self.reserve_policy_decision_state()?;
                self.policy_full_quality_action_probability(state, action, policy)?
            }
        } else {
            0.0
        };
        self.policy_full_quality_decision_values
            .insert(state, probability);
        Ok(probability)
    }

    fn policy_full_quality_action_probability(
        &mut self,
        state: DecisionState,
        action: Action,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<f64, StochasticSolveError> {
        let mut probability = 0.0;
        for outcome in self
            .post_decision_outcomes_cached(state, action)
            .map_err(|_| StochasticSolveError::NoLegalAction)?
            .iter()
        {
            probability += outcome.probability
                * self.policy_full_quality_post_decision_probability(outcome.value, policy)?;
        }
        Ok(probability)
    }

    fn policy_full_quality_post_decision_probability(
        &mut self,
        post: PostDecisionState,
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<f64, StochasticSolveError> {
        if let Some(value) = self.terminal_value(post.simulation, post.step, post.decisions) {
            return Ok(value.full_quality_completion_probability);
        }
        if let Some(probability) = self.policy_full_quality_post_decision_values.get(&post) {
            return Ok(*probability);
        }
        if self
            .post_decision_upper_bound(post)
            .full_quality_completion_probability
            == 0.0
        {
            self.stats.pruned_policy_states += 1;
            self.policy_full_quality_post_decision_values
                .insert(post, 0.0);
            return Ok(0.0);
        }
        self.reserve_policy_post_decision_state()?;
        let mut probability = 0.0;
        for outcome in self.condition_outcomes_cached(post).iter() {
            probability += outcome.probability
                * self.policy_full_quality_decision_probability(outcome.value, policy)?;
        }
        self.policy_full_quality_post_decision_values
            .insert(post, probability);
        Ok(probability)
    }

    fn policy_full_quality_action_intervals(
        &mut self,
        root: DecisionState,
        root_actions: &[Action],
        policy: &dyn Fn(DecisionState) -> Option<Action>,
    ) -> Result<Vec<ProbabilityInterval>, StochasticSolveError> {
        let action_count = root_actions.len();
        let mut successes = vec![0.0; action_count];
        let mut frontier = FxHashMap::<PolicyFrontierState, ProbabilityMasses>::default();
        let mut queue = BinaryHeap::new();
        let mut sequence = 0;
        for (action_index, action) in root_actions.iter().copied().enumerate() {
            for outcome in self
                .post_decision_outcomes_cached(root, action)
                .map_err(|_| StochasticSolveError::NoLegalAction)?
                .iter()
            {
                let mut masses = zero_probability_masses(action_count);
                masses[action_index] = outcome.probability;
                self.queue_policy_masses(
                    &mut frontier,
                    &mut queue,
                    &mut sequence,
                    PolicyFrontierState::PostDecision(outcome.value),
                    masses,
                    &mut successes,
                );
            }
        }

        while let Some(entry) = queue.pop() {
            let Some(priority) = frontier
                .get(&entry.state)
                .map(|masses| total_probability_mass(masses))
            else {
                continue;
            };
            if priority.total_cmp(&entry.priority).is_ne() {
                continue;
            }
            if self.total_states() >= self.expansion_limit {
                self.stats.deferred_policy_states += frontier.len();
                let unresolved = sum_probability_masses(action_count, frontier.values());
                return Ok(probability_intervals(successes, unresolved));
            }
            let masses = frontier
                .remove(&entry.state)
                .expect("queued policy state must retain probability mass");
            match entry.state {
                PolicyFrontierState::PostDecision(post) => {
                    self.reserve_policy_post_decision_state()?;
                    for outcome in self.condition_outcomes_cached(post).iter() {
                        self.queue_policy_masses(
                            &mut frontier,
                            &mut queue,
                            &mut sequence,
                            PolicyFrontierState::Decision(outcome.value),
                            scaled_probability_masses(&masses, outcome.probability),
                            &mut successes,
                        );
                    }
                }
                PolicyFrontierState::Decision(state) => {
                    self.reserve_policy_decision_state()?;
                    let Some(action) = self.policy_action(state, policy) else {
                        continue;
                    };
                    for outcome in self
                        .post_decision_outcomes_cached(state, action)
                        .map_err(|_| StochasticSolveError::NoLegalAction)?
                        .iter()
                    {
                        self.queue_policy_masses(
                            &mut frontier,
                            &mut queue,
                            &mut sequence,
                            PolicyFrontierState::PostDecision(outcome.value),
                            scaled_probability_masses(&masses, outcome.probability),
                            &mut successes,
                        );
                    }
                }
            }
        }

        Ok(successes
            .into_iter()
            .map(ProbabilityInterval::exact)
            .collect())
    }

    fn queue_policy_masses(
        &mut self,
        frontier: &mut FxHashMap<PolicyFrontierState, ProbabilityMasses>,
        queue: &mut BinaryHeap<PolicyFrontierEntry>,
        sequence: &mut u64,
        state: PolicyFrontierState,
        masses: ProbabilityMasses,
        successes: &mut [f64],
    ) {
        let (simulation, step, decisions, upper_bound) = match state {
            PolicyFrontierState::Decision(state) => (
                state.simulation,
                state.step,
                state.decisions,
                self.decision_upper_bound(state),
            ),
            PolicyFrontierState::PostDecision(post) => (
                post.simulation,
                post.step,
                post.decisions,
                self.post_decision_upper_bound(post),
            ),
        };
        if let Some(value) = self.terminal_value(simulation, step, decisions) {
            if value.full_quality_completion_probability == 1.0 {
                add_probability_slice(successes, &masses);
            }
        } else if upper_bound.full_quality_completion_probability == 0.0 {
            self.stats.pruned_policy_states += 1;
        } else {
            add_probability_masses(frontier, state, masses);
            *sequence = sequence.wrapping_add(1);
            queue.push(PolicyFrontierEntry {
                priority: total_probability_mass(
                    frontier
                        .get(&state)
                        .expect("queued policy state must retain probability mass"),
                ),
                sequence: *sequence,
                state,
            });
        }
    }

    fn terminal_value(
        &self,
        simulation: SimulationState,
        step: u8,
        decisions: u8,
    ) -> Option<PolicyValue> {
        let complete = simulation.progress >= self.model.settings.max_progress;
        let failed = simulation.durability == 0
            || step >= self.model.max_steps
            || decisions >= self.model.max_decisions;
        if !complete && !failed {
            return None;
        }
        Some(PolicyValue {
            full_quality_completion_probability: f64::from(
                complete && simulation.quality >= self.required_quality,
            ),
            completion_probability: f64::from(complete),
            expected_quality: f64::from(simulation.quality.min(self.required_quality))
                / f64::from(self.required_quality),
        })
    }

    fn failed_value(&self, simulation: SimulationState) -> PolicyValue {
        PolicyValue {
            expected_quality: f64::from(simulation.quality.min(self.required_quality))
                / f64::from(self.required_quality),
            ..PolicyValue::default()
        }
    }

    fn reserve_decision_state(&mut self) -> Result<(), StochasticSolveError> {
        if self.total_states() >= self.expansion_limit {
            return Err(StochasticSolveError::ExpansionLimit);
        }
        self.stats.decision_states += 1;
        Ok(())
    }

    fn reserve_post_decision_state(&mut self) -> Result<(), StochasticSolveError> {
        if self.total_states() >= self.expansion_limit {
            return Err(StochasticSolveError::ExpansionLimit);
        }
        self.stats.post_decision_states += 1;
        Ok(())
    }

    fn reserve_policy_decision_state(&mut self) -> Result<(), StochasticSolveError> {
        if self.total_states() >= self.expansion_limit {
            return Err(StochasticSolveError::ExpansionLimit);
        }
        self.stats.policy_decision_states += 1;
        Ok(())
    }

    fn reserve_policy_post_decision_state(&mut self) -> Result<(), StochasticSolveError> {
        if self.total_states() >= self.expansion_limit {
            return Err(StochasticSolveError::ExpansionLimit);
        }
        self.stats.policy_post_decision_states += 1;
        Ok(())
    }

    fn total_states(&self) -> usize {
        self.stats.decision_states
            + self.stats.post_decision_states
            + self.stats.policy_decision_states
            + self.stats.policy_post_decision_states
    }

    fn clear_values(&mut self) {
        self.decision_values.clear();
        self.post_decision_values.clear();
        self.policy_decision_values.clear();
        self.policy_post_decision_values.clear();
        self.policy_full_quality_decision_values.clear();
        self.policy_full_quality_post_decision_values.clear();
        self.bounded_decision_values.clear();
        self.bounded_post_decision_values.clear();
        self.stats = StochasticSolveStats::default();
    }
}

fn add_solve_stats(
    left: StochasticSolveStats,
    right: StochasticSolveStats,
) -> StochasticSolveStats {
    StochasticSolveStats {
        decision_states: left.decision_states + right.decision_states,
        post_decision_states: left.post_decision_states + right.post_decision_states,
        policy_decision_states: left.policy_decision_states + right.policy_decision_states,
        policy_post_decision_states: left.policy_post_decision_states
            + right.policy_post_decision_states,
        pruned_actions: left.pruned_actions + right.pruned_actions,
        pruned_policy_states: left.pruned_policy_states + right.pruned_policy_states,
        deferred_policy_states: left.deferred_policy_states + right.deferred_policy_states,
    }
}

fn bounded_graph_action_interval(
    action: &BoundedGraphAction,
    nodes: &[BoundedGraphNode],
) -> ProbabilityInterval {
    let mut interval = ProbabilityInterval::exact(0.0);
    for successor in &action.successors {
        interval.add_assign(
            nodes[successor.value]
                .interval
                .scaled(successor.probability),
        );
    }
    interval
}

fn recompute_bounded_graph_node(node_index: usize, nodes: &mut [BoundedGraphNode]) -> bool {
    let interval = match &nodes[node_index].edges {
        BoundedGraphEdges::Unexpanded | BoundedGraphEdges::Terminal => {
            return false;
        }
        BoundedGraphEdges::PostDecision(successors) => {
            let mut interval = ProbabilityInterval::exact(0.0);
            for successor in successors {
                interval.add_assign(
                    nodes[successor.value]
                        .interval
                        .scaled(successor.probability),
                );
            }
            interval
        }
        BoundedGraphEdges::Decision(actions) => {
            let mut lower = 0.0_f64;
            let mut upper = 0.0_f64;
            for action in actions {
                let interval = bounded_graph_action_interval(action, nodes);
                lower = lower.max(interval.lower);
                upper = upper.max(interval.upper);
            }
            ProbabilityInterval { lower, upper }
        }
    };
    if nodes[node_index].interval == interval {
        false
    } else {
        nodes[node_index].interval = interval;
        true
    }
}

fn enqueue_bounded_graph_mass(
    node_index: usize,
    mass: f64,
    nodes: &[BoundedGraphNode],
    masses: &mut FxHashMap<usize, f64>,
    queue: &mut BinaryHeap<Reverse<(u16, usize)>>,
) {
    if mass == 0.0 {
        return;
    }
    if let Some(existing) = masses.get_mut(&node_index) {
        *existing += mass;
        return;
    }
    masses.insert(node_index, mass);
    queue.push(Reverse((
        bounded_graph_rank(nodes[node_index].state),
        node_index,
    )));
}

fn bounded_graph_rank(state: PolicyFrontierState) -> u16 {
    match state {
        PolicyFrontierState::PostDecision(post) => u16::from(post.decisions) * 2,
        PolicyFrontierState::Decision(state) => u16::from(state.decisions) * 2 + 1,
    }
}

fn zero_probability_masses(action_count: usize) -> ProbabilityMasses {
    let mut masses = ProbabilityMasses::new();
    masses.resize(action_count, 0.0);
    masses
}

fn scaled_probability_masses(masses: &[f64], probability: f64) -> ProbabilityMasses {
    masses.iter().map(|mass| mass * probability).collect()
}

fn add_probability_slice(target: &mut [f64], masses: &[f64]) {
    for (target, mass) in target.iter_mut().zip(masses) {
        *target += mass;
    }
}

fn add_probability_masses<K: Eq + std::hash::Hash>(
    frontier: &mut FxHashMap<K, ProbabilityMasses>,
    state: K,
    masses: ProbabilityMasses,
) {
    let target = frontier
        .entry(state)
        .or_insert_with(|| zero_probability_masses(masses.len()));
    add_probability_slice(target, &masses);
}

fn total_probability_mass(masses: &[f64]) -> f64 {
    masses.iter().sum()
}

fn sum_probability_masses<'a>(
    action_count: usize,
    masses: impl Iterator<Item = &'a ProbabilityMasses>,
) -> Vec<f64> {
    let mut total = vec![0.0; action_count];
    for masses in masses {
        add_probability_slice(&mut total, masses);
    }
    total
}

fn probability_intervals(successes: Vec<f64>, unresolved: Vec<f64>) -> Vec<ProbabilityInterval> {
    successes
        .into_iter()
        .zip(unresolved)
        .map(|(lower, unresolved)| ProbabilityInterval {
            lower: lower.min(1.0),
            upper: (lower + unresolved).min(1.0),
        })
        .collect()
}

fn upper_bound_may_improve(upper: PolicyValue, lower: PolicyValue) -> bool {
    const EPSILON: f64 = 1e-12;
    if upper.full_quality_completion_probability + EPSILON
        < lower.full_quality_completion_probability
    {
        return false;
    }
    if upper.full_quality_completion_probability
        > lower.full_quality_completion_probability + EPSILON
    {
        return true;
    }
    if upper.completion_probability + EPSILON < lower.completion_probability {
        return false;
    }
    if upper.completion_probability > lower.completion_probability + EPSILON {
        return true;
    }
    upper.expected_quality + EPSILON >= lower.expected_quality
}

fn optimistic_state_index(
    inner_quiet: u8,
    great_strides: bool,
    innovation: bool,
    muscle_memory: bool,
    veneration: bool,
) -> usize {
    usize::from(inner_quiet.min(10)) * 16
        + usize::from(great_strides)
        + 2 * usize::from(innovation)
        + 4 * usize::from(muscle_memory)
        + 8 * usize::from(veneration)
}

fn optimistic_step_frontiers(
    model: &StochasticModel,
    required_quality: u16,
    actions: &[Action],
    conditions: &[Condition],
) -> OptimisticStepFrontiers {
    const COMBOS: [Combo; 4] = [
        Combo::None,
        Combo::SynthesisBegin,
        Combo::BasicTouch,
        Combo::StandardTouch,
    ];
    let cutoff = ParetoValue::new(model.settings.max_progress, required_quality);
    let mut transitions = (0..OPTIMISTIC_STATE_COUNT)
        .map(|_| Vec::<(usize, ParetoValue)>::new())
        .collect::<Vec<_>>();
    let mut nonadvancing_gain = false;
    for inner_quiet in 0..OPTIMISTIC_INNER_QUIET_COUNT {
        for great_strides in [false, true] {
            for innovation in [false, true] {
                for muscle_memory in [false, true] {
                    for veneration in [false, true] {
                        let optimistic_state = optimistic_state_index(
                            inner_quiet as u8,
                            great_strides,
                            innovation,
                            muscle_memory,
                            veneration,
                        );
                        transitions[optimistic_state]
                            .push((optimistic_state, ParetoValue::default()));
                        for combo in COMBOS {
                            let mut god_state = SimulationState::new(&model.settings);
                            god_state.effects = god_state
                                .effects
                                .with_inner_quiet(inner_quiet as u8)
                                .with_innovation(if innovation { 7 } else { 0 })
                                .with_veneration(if veneration { 7 } else { 0 })
                                .with_great_strides(if great_strides { 7 } else { 0 })
                                .with_muscle_memory(if muscle_memory { 7 } else { 0 })
                                .with_trained_perfection_active(true)
                                .with_heart_and_soul_active(true)
                                .with_stellar_steady_hand_charges(7)
                                .with_stellar_steady_hand(3)
                                .with_expedience(true)
                                .with_splendor_cosmic(true)
                                .with_combo(combo);
                            for action in actions {
                                for condition in conditions.iter().copied() {
                                    let Ok(next) = god_state.use_action_with_outcome(
                                        *action,
                                        condition,
                                        &model.settings,
                                        true,
                                    ) else {
                                        continue;
                                    };
                                    let gain = ParetoValue::new(next.progress, next.quality);
                                    nonadvancing_gain |= !action.increases_step_count()
                                        && (gain.progress != 0 || gain.quality != 0);
                                    if action.increases_step_count() {
                                        transitions[optimistic_state].push((
                                            optimistic_state_index(
                                                next.effects.inner_quiet(),
                                                next.effects.great_strides() != 0,
                                                next.effects.innovation() != 0,
                                                next.effects.muscle_memory() != 0,
                                                next.effects.veneration() != 0,
                                            ),
                                            gain,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if nonadvancing_gain {
        return (0..=model.max_steps)
            .map(|_| {
                (0..OPTIMISTIC_STATE_COUNT)
                    .map(|_| Box::from([cutoff]))
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
    }
    let mut frontiers: Vec<Box<[Box<[ParetoValue]>]>> =
        Vec::with_capacity(usize::from(model.max_steps) + 1);
    frontiers.push(
        (0..OPTIMISTIC_STATE_COUNT)
            .map(|_| Box::from([ParetoValue::default()]))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    for _ in 0..model.max_steps {
        let previous = frontiers.last().unwrap();
        let mut next = Vec::with_capacity(OPTIMISTIC_STATE_COUNT);
        for state_transitions in &transitions {
            let mut builder = ParetoFrontBuilder::new();
            builder.initialize_with_cutoff(cutoff);
            for (next_inner_quiet, gain) in state_transitions.iter().copied() {
                for continuation in previous[next_inner_quiet].iter().copied() {
                    builder.push(gain.saturating_add(continuation));
                }
            }
            next.push(builder.result_as_slice().into());
        }
        frontiers.push(next.into_boxed_slice());
    }
    frontiers.into_boxed_slice()
}

impl StochasticModel {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_steps == 0 || self.max_decisions < self.max_steps {
            return Err(String::from("invalid stochastic decision horizon"));
        }
        let probability_sum = self
            .condition_probabilities_bps
            .iter()
            .map(|probability| u32::from(*probability))
            .sum::<u32>();
        if probability_sum != 10_000 {
            return Err(format!(
                "condition probabilities total {probability_sum} basis points; expected 10000"
            ));
        }
        Ok(())
    }

    pub fn post_decision_outcomes(
        &self,
        state: DecisionState,
        action: Action,
    ) -> Result<Vec<Weighted<PostDecisionState>>, ActionError> {
        let success_rate = action.success_rate(&state.simulation, state.condition);
        let transition = self.condition_transition(state.condition, action);
        let mut outcomes = Vec::with_capacity(2);
        if success_rate > 0 {
            outcomes.push(Weighted {
                value: self.apply_action_outcome(state, action, true, transition)?,
                probability: f64::from(success_rate) / 100.0,
            });
        }
        if success_rate < 100 {
            outcomes.push(Weighted {
                value: self.apply_action_outcome(state, action, false, transition)?,
                probability: 1.0 - f64::from(success_rate) / 100.0,
            });
        }
        Ok(outcomes)
    }

    pub fn apply_outcome(
        &self,
        state: DecisionState,
        action: Action,
        succeeded: bool,
        condition_draw: f64,
    ) -> Result<DecisionState, ActionError> {
        let transition = self.condition_transition(state.condition, action);
        let post = self.apply_action_outcome(state, action, succeeded, transition)?;
        let condition = match transition {
            ConditionTransition::Fixed(condition) => condition,
            ConditionTransition::Random => self.condition_from_draw(condition_draw),
        };
        Ok(DecisionState {
            simulation: post.simulation,
            condition,
            step: post.step,
            decisions: post.decisions,
        })
    }

    pub fn condition_outcomes(&self, post: PostDecisionState) -> Vec<Weighted<DecisionState>> {
        match post.transition {
            ConditionTransition::Fixed(condition) => vec![Weighted {
                value: DecisionState {
                    simulation: post.simulation,
                    condition,
                    step: post.step,
                    decisions: post.decisions,
                },
                probability: 1.0,
            }],
            ConditionTransition::Random => self
                .condition_probabilities_bps
                .iter()
                .enumerate()
                .filter(|(_, probability)| **probability != 0)
                .map(|(index, probability)| Weighted {
                    value: DecisionState {
                        simulation: post.simulation,
                        condition: condition_from_index(index),
                        step: post.step,
                        decisions: post.decisions,
                    },
                    probability: f64::from(*probability) / 10_000.0,
                })
                .collect(),
        }
    }

    fn apply_action_outcome(
        &self,
        state: DecisionState,
        action: Action,
        succeeded: bool,
        transition: ConditionTransition,
    ) -> Result<PostDecisionState, ActionError> {
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
        let spent_delineation = succeeded && specialist_action;
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
        Ok(PostDecisionState {
            simulation,
            transition,
            step: state
                .step
                .saturating_add(u8::from(action.increases_step_count())),
            decisions: state.decisions.saturating_add(1),
        })
    }

    fn condition_transition(&self, current: Condition, action: Action) -> ConditionTransition {
        if !action.advances_condition() {
            return ConditionTransition::Fixed(current);
        }
        match current {
            Condition::Excellent => ConditionTransition::Fixed(Condition::Poor),
            Condition::Poor => ConditionTransition::Fixed(Condition::Normal),
            Condition::GoodOmen => ConditionTransition::Fixed(Condition::Good),
            Condition::Robust => ConditionTransition::Fixed(Condition::Sturdy),
            _ => ConditionTransition::Random,
        }
    }

    fn condition_from_draw(&self, draw: f64) -> Condition {
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

fn condition_index(condition: Condition) -> usize {
    match condition {
        Condition::Normal => 0,
        Condition::Good => 1,
        Condition::Excellent => 2,
        Condition::Poor => 3,
        Condition::Centered => 4,
        Condition::Sturdy => 5,
        Condition::Pliant => 6,
        Condition::Malleable => 7,
        Condition::Primed => 8,
        Condition::GoodOmen => 9,
        Condition::Robust => 10,
    }
}

fn supported_condition_mask(probabilities_bps: [u16; CONDITION_COUNT]) -> u16 {
    probabilities_bps
        .into_iter()
        .enumerate()
        .fold(0, |mask, (index, probability)| {
            mask | (u16::from(probability != 0) << index)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use raphael_sim::{ActionMask, Effects};

    fn model() -> StochasticModel {
        StochasticModel {
            settings: Settings {
                max_cp: 771,
                max_durability: 60,
                max_progress: 11_250,
                max_quality: 31_520,
                base_progress: 322,
                base_quality: 343,
                job_level: 100,
                allowed_actions: ActionMask::all().remove(Action::TrainedEye),
                adversarial: false,
                backload_progress: false,
                stellar_steady_hand_charges: 0,
            },
            condition_probabilities_bps: [
                2_000, 1_000, 0, 0, 1_500, 1_000, 1_500, 1_000, 1_000, 0, 1_000,
            ],
            max_steps: 55,
            max_decisions: 64,
        }
    }

    fn state(condition: Condition) -> DecisionState {
        DecisionState {
            simulation: SimulationState {
                cp: 771,
                durability: 60,
                progress: 0,
                quality: 0,
                unreliable_quality: 0,
                effects: Effects::new(),
            },
            condition,
            step: 1,
            decisions: 1,
        }
    }

    #[test]
    fn optimistic_frontier_cache_reuses_only_matching_bound_models() {
        let mut cache = OptimisticFrontierCache::default();
        let mut first_model = model();
        first_model.max_steps = 2;
        let first = cache.get_or_build(&first_model, 100);

        let mut different_weights = first_model;
        different_weights.condition_probabilities_bps[0] -= 100;
        different_weights.condition_probabilities_bps[1] += 100;
        let same_bound = cache.get_or_build(&different_weights, 100);
        assert!(Arc::ptr_eq(&first, &same_bound));

        let mut different_support = different_weights;
        different_support.condition_probabilities_bps[0] +=
            different_support.condition_probabilities_bps[1];
        different_support.condition_probabilities_bps[1] = 0;
        let support_bound = cache.get_or_build(&different_support, 100);
        assert!(!Arc::ptr_eq(&same_bound, &support_bound));

        let quality_bound = cache.get_or_build(&different_support, 101);
        assert!(!Arc::ptr_eq(&support_bound, &quality_bound));

        let mut different_settings = different_support;
        different_settings.settings.base_quality += 1;
        let settings_bound = cache.get_or_build(&different_settings, 101);
        assert!(!Arc::ptr_eq(&quality_bound, &settings_bound));

        different_settings.max_steps += 1;
        let step_bound = cache.get_or_build(&different_settings, 101);
        assert!(!Arc::ptr_eq(&settings_bound, &step_bound));
    }

    #[test]
    fn policy_independent_cache_survives_only_matching_solver_identity() {
        let mut model = model();
        model.settings.max_progress = 100;
        model.settings.max_quality = 100;
        model.settings.base_progress = 100;
        model.settings.allowed_actions = ActionMask::none().add(Action::BasicSynthesis);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 2;
        model.max_decisions = 2;
        let root = DecisionState {
            simulation: SimulationState::new(&model.settings),
            condition: Condition::Normal,
            step: 0,
            decisions: 0,
        };
        let solver =
            StochasticPolicySolver::new(model, 100, [Action::BasicSynthesis], 100).unwrap();
        let post = solver
            .post_decision_outcomes_cached(root, Action::BasicSynthesis)
            .unwrap()[0]
            .value;
        solver.condition_outcomes_cached(post);
        solver
            .action_upper_bound(root, Action::BasicSynthesis)
            .unwrap();
        let retained_entries = solver.persistent_cache.borrow().entry_count();
        assert!(retained_entries >= 3);

        let solver = StochasticPolicySolver::new_with_cache(
            model,
            100,
            [Action::BasicSynthesis],
            100,
            solver.into_cache(),
        )
        .unwrap();
        assert_eq!(
            solver.persistent_cache.borrow().entry_count(),
            retained_entries
        );

        let mut changed_model = model;
        changed_model.max_steps = 3;
        changed_model.max_decisions = 3;
        let solver = StochasticPolicySolver::new_with_cache(
            changed_model,
            100,
            [Action::BasicSynthesis],
            100,
            solver.into_cache(),
        )
        .unwrap();
        assert_eq!(solver.persistent_cache.borrow().entry_count(), 0);
    }

    #[test]
    fn random_condition_expansion_uses_the_stored_aqueduct_vector() {
        let model = model();
        let post = model
            .post_decision_outcomes(state(Condition::Normal), Action::Observe)
            .unwrap()[0]
            .value;
        assert_eq!(post.transition, ConditionTransition::Random);
        let outcomes = model.condition_outcomes(post);
        assert_eq!(outcomes.len(), 8);
        assert!(
            (outcomes
                .iter()
                .map(|outcome| outcome.probability)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );
        assert!(outcomes.iter().any(|outcome| {
            outcome.value.condition == Condition::Malleable
                && (outcome.probability - 0.1).abs() < 1e-12
        }));
    }

    #[test]
    fn robust_transition_is_factored_as_one_forced_post_decision_edge() {
        let model = model();
        let post = model
            .post_decision_outcomes(state(Condition::Robust), Action::Observe)
            .unwrap()[0]
            .value;
        assert_eq!(
            post.transition,
            ConditionTransition::Fixed(Condition::Sturdy)
        );
        let outcomes = model.condition_outcomes(post);
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].value.condition, Condition::Sturdy);
        assert_eq!(outcomes[0].probability, 1.0);
    }

    #[test]
    fn unreliable_action_branches_before_condition_expansion() {
        let model = model();
        let post = model
            .post_decision_outcomes(state(Condition::Normal), Action::RapidSynthesis)
            .unwrap();
        assert_eq!(post.len(), 2);
        assert!((post.iter().map(|outcome| outcome.probability).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            post.iter()
                .all(|outcome| outcome.value.transition == ConditionTransition::Random)
        );
        assert_ne!(
            post[0].value.simulation.progress,
            post[1].value.simulation.progress
        );
    }

    #[test]
    fn specialist_zero_step_transition_preserves_condition_and_consumes_one_delineation() {
        let model = model();
        let mut root = state(Condition::Centered);
        root.simulation.effects = root
            .simulation
            .effects
            .with_careful_observation_charges(3)
            .with_crafter_delineations(5);
        let post = model
            .post_decision_outcomes(root, Action::CarefulObservation)
            .unwrap()[0]
            .value;
        assert_eq!(post.step, root.step);
        assert_eq!(post.decisions, root.decisions + 1);
        assert_eq!(post.simulation.effects.careful_observation_charges(), 2);
        assert_eq!(post.simulation.effects.crafter_delineations(), 4);
        assert_eq!(post.transition, ConditionTransition::Random);
    }

    #[test]
    fn exact_bellman_search_delays_completion_until_required_quality_is_reached() {
        let mut model = model();
        model.settings.max_progress = 100;
        model.settings.max_quality = 100;
        model.settings.base_progress = 100;
        model.settings.base_quality = 100;
        model.settings.allowed_actions = ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::BasicTouch);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 2;
        model.max_decisions = 2;
        let mut solver = StochasticPolicySolver::new(
            model,
            100,
            [Action::BasicSynthesis, Action::BasicTouch],
            10_000,
        )
        .unwrap();

        let mut root = state(Condition::Normal);
        root.simulation = SimulationState::new(&model.settings);
        root.step = 0;
        root.decisions = 0;
        let after_touch = model
            .apply_outcome(root, Action::BasicTouch, true, 0.0)
            .unwrap();
        assert_eq!(after_touch.simulation.quality, 100);
        let outcome = solver.solve(root).unwrap();
        assert_eq!(outcome.action, Action::BasicTouch);
        assert_eq!(outcome.value.full_quality_completion_probability, 1.0);
        assert_eq!(outcome.value.completion_probability, 1.0);

        let outcome = solver
            .solve_with_policy(root, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(outcome.action, Action::BasicTouch);
        assert_eq!(outcome.value.full_quality_completion_probability, 1.0);
        assert!(outcome.stats.policy_decision_states > 0);

        let outcome = solver
            .improve_policy_full_quality(root, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(outcome.incumbent_action, Action::BasicSynthesis);
        assert_eq!(outcome.incumbent_probability, 0.0);
        assert_eq!(outcome.action, Action::BasicTouch);
        assert_eq!(outcome.probability, 1.0);

        let outcome = solver
            .improve_policy_full_quality_bounded(root, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(outcome.incumbent_action, Action::BasicSynthesis);
        assert_eq!(outcome.action, Action::BasicTouch);
        assert_eq!(outcome.interval, ProbabilityInterval::exact(1.0));
        assert!(outcome.improvement_proven);
        assert!(outcome.optimality_proven);

        let outcome = solver
            .improve_policy_full_quality_recursive_bounded(root, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(outcome.incumbent_action, Action::BasicSynthesis);
        assert_eq!(outcome.action, Action::BasicTouch);
        assert_eq!(outcome.interval, ProbabilityInterval::exact(1.0));
        assert!(outcome.improvement_proven);
        assert!(outcome.optimality_proven);
        assert_eq!(
            solver.recursive_policy_action(after_touch),
            Some(Action::BasicSynthesis)
        );

        let outcome = solver
            .improve_policy_full_quality_best_first_bounded(root, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(outcome.incumbent_action, Action::BasicSynthesis);
        assert_eq!(outcome.action, Action::BasicTouch);
        assert_eq!(outcome.interval, ProbabilityInterval::exact(1.0));
        assert!(outcome.improvement_proven);
        assert!(outcome.optimality_proven);
    }

    #[test]
    fn recursive_bellman_interval_uses_the_best_executable_lower_bound() {
        let mut model = model();
        model.settings.allowed_actions = ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::BasicTouch);
        model.settings.max_progress = 200;
        model.settings.max_quality = 200;
        model.settings.base_progress = 100;
        model.settings.base_quality = 100;
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut root = state(Condition::Normal);
        root.simulation = SimulationState::new(&model.settings);
        let synthesis_post = model
            .post_decision_outcomes(root, Action::BasicSynthesis)
            .unwrap()[0]
            .value;
        let touch_post = model
            .post_decision_outcomes(root, Action::BasicTouch)
            .unwrap()[0]
            .value;
        let mut solver = StochasticPolicySolver::new(
            model,
            200,
            [Action::BasicSynthesis, Action::BasicTouch],
            10_000,
        )
        .unwrap();
        solver.bounded_post_decision_values.insert(
            synthesis_post,
            ProbabilityInterval {
                lower: 0.10,
                upper: 0.90,
            },
        );
        solver.bounded_post_decision_values.insert(
            touch_post,
            ProbabilityInterval {
                lower: 0.70,
                upper: 1.0,
            },
        );

        let value = solver
            .bounded_recursive_decision_value(root, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(value.action, Some(Action::BasicTouch));
        assert_eq!(
            value.interval,
            ProbabilityInterval {
                lower: 0.70,
                upper: 1.0,
            }
        );
    }

    #[test]
    fn recursive_cutoff_credits_terminal_success_from_the_concrete_policy() {
        let mut model = model();
        model.settings.max_progress = 200;
        model.settings.max_quality = 100;
        model.settings.base_progress = 100;
        model.settings.allowed_actions = ActionMask::none().add(Action::BasicSynthesis);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 3;
        model.max_decisions = 3;
        let mut root = state(Condition::Normal);
        root.simulation = SimulationState::new(&model.settings);
        root.simulation.quality = 100;
        root.step = 0;
        root.decisions = 0;
        let post = model
            .post_decision_outcomes(root, Action::BasicSynthesis)
            .unwrap()[0]
            .value;
        let mut solver =
            StochasticPolicySolver::new(model, 100, [Action::BasicSynthesis], 1).unwrap();

        let interval = solver
            .bounded_recursive_post_decision_value(post, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(interval, ProbabilityInterval::exact(1.0));
        assert!(solver.stats.deferred_policy_states > 0);
    }

    #[test]
    fn exact_bellman_search_integrates_unreliable_action_success() {
        let mut model = model();
        model.settings.max_progress = 100;
        model.settings.max_quality = 100;
        model.settings.base_progress = 100;
        model.settings.allowed_actions = ActionMask::none().add(Action::RapidSynthesis);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 1;
        model.max_decisions = 1;
        let mut root = state(Condition::Normal);
        root.simulation = SimulationState::new(&model.settings);
        root.step = 0;
        root.decisions = 0;
        root.simulation.quality = 100;
        let mut solver =
            StochasticPolicySolver::new(model, 100, [Action::RapidSynthesis], 10_000).unwrap();

        let outcome = solver.solve(root).unwrap();
        assert_eq!(outcome.action, Action::RapidSynthesis);
        assert_eq!(outcome.value.full_quality_completion_probability, 0.5);
        assert_eq!(outcome.value.completion_probability, 0.5);
    }

    #[test]
    fn exact_policy_improvement_replaces_a_worse_incumbent_root_action() {
        let mut model = model();
        model.settings.max_progress = 100;
        model.settings.max_quality = 100;
        model.settings.base_progress = 100;
        model.settings.base_quality = 100;
        model.settings.allowed_actions = ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::BasicTouch);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 2;
        model.max_decisions = 2;
        let mut root = state(Condition::Normal);
        root.simulation = SimulationState::new(&model.settings);
        root.step = 0;
        root.decisions = 0;
        let mut solver = StochasticPolicySolver::new(
            model,
            100,
            [Action::BasicSynthesis, Action::BasicTouch],
            10_000,
        )
        .unwrap();

        let outcome = solver
            .improve_policy(root, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(outcome.incumbent_action, Action::BasicSynthesis);
        assert_eq!(
            outcome.incumbent_value.full_quality_completion_probability,
            0.0
        );
        assert_eq!(outcome.action, Action::BasicTouch);
        assert_eq!(outcome.value.full_quality_completion_probability, 1.0);
        assert!(outcome.stats.policy_decision_states > 0);
    }

    #[test]
    fn policy_improvement_bound_includes_allowed_continuation_actions() {
        let mut model = model();
        model.settings.max_progress = 100;
        model.settings.max_quality = 100;
        model.settings.base_progress = 100;
        model.settings.base_quality = 100;
        model.settings.allowed_actions = ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::BasicTouch);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 2;
        model.max_decisions = 2;
        let mut root = state(Condition::Normal);
        root.simulation = SimulationState::new(&model.settings);
        root.step = 0;
        root.decisions = 0;
        let mut solver =
            StochasticPolicySolver::new(model, 100, [Action::BasicTouch], 10_000).unwrap();

        let outcome = solver
            .improve_policy(root, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(outcome.incumbent_action, Action::BasicSynthesis);
        assert_eq!(outcome.action, Action::BasicTouch);
        assert_eq!(outcome.value.full_quality_completion_probability, 1.0);
    }

    #[test]
    fn full_quality_policy_evaluation_stops_at_an_impossible_optimistic_bound() {
        let mut model = model();
        model.settings.max_progress = 100;
        model.settings.max_quality = 100;
        model.settings.allowed_actions = ActionMask::none().add(Action::Observe);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 2;
        model.max_decisions = 2;
        let mut root = state(Condition::Normal);
        root.simulation = SimulationState::new(&model.settings);
        root.step = 0;
        root.decisions = 0;
        let mut solver =
            StochasticPolicySolver::new(model, 100, [Action::Observe], 10_000).unwrap();

        let outcome = solver
            .improve_policy_full_quality(root, &|_| Some(Action::Observe))
            .unwrap();
        assert_eq!(outcome.probability, 0.0);
        assert_eq!(outcome.stats.policy_decision_states, 0);
        assert!(outcome.stats.pruned_policy_states > 0);
    }

    #[test]
    fn concrete_policy_lower_bound_enables_proof_at_the_expansion_cap() {
        let mut model = model();
        model.settings.max_progress = 100;
        model.settings.max_quality = 100;
        model.settings.base_progress = 100;
        model.settings.base_quality = 100;
        model.settings.allowed_actions = ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::BasicTouch);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 2;
        model.max_decisions = 2;
        let mut root = state(Condition::Normal);
        root.simulation = SimulationState::new(&model.settings);
        root.step = 0;
        root.decisions = 0;
        let mut solver = StochasticPolicySolver::new(model, 100, [Action::BasicTouch], 1).unwrap();

        let outcome = solver
            .improve_policy_full_quality_bounded(root, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(outcome.action, Action::BasicSynthesis);
        assert!(!outcome.improvement_proven);
        assert!(!outcome.optimality_proven);
        assert!(outcome.stats.deferred_policy_states > 0);

        let outcome = solver
            .improve_policy_full_quality_best_first_bounded(root, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(outcome.action, Action::BasicTouch);
        assert_eq!(outcome.interval, ProbabilityInterval::exact(1.0));
        assert!(outcome.improvement_proven);
        assert!(outcome.optimality_proven);

        let outcome = solver
            .improve_policy_full_quality_recursive_bounded(root, &|_| Some(Action::BasicSynthesis))
            .unwrap();
        assert_eq!(outcome.action, Action::BasicTouch);
        assert_eq!(outcome.interval, ProbabilityInterval::exact(1.0));
        assert!(outcome.improvement_proven);
        assert!(outcome.optimality_proven);
        assert!(outcome.stats.deferred_policy_states > 0);
    }

    #[test]
    fn stochastic_bound_prunes_an_action_that_cannot_match_the_exact_incumbent() {
        let mut model = model();
        model.settings.max_progress = 100;
        model.settings.max_quality = 100;
        model.settings.base_progress = 100;
        model.settings.allowed_actions = ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::Observe);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 1;
        model.max_decisions = 1;
        let mut root = state(Condition::Normal);
        root.simulation = SimulationState::new(&model.settings);
        root.simulation.quality = 100;
        root.step = 0;
        root.decisions = 0;
        let mut solver = StochasticPolicySolver::new(
            model,
            100,
            [Action::BasicSynthesis, Action::Observe],
            10_000,
        )
        .unwrap();

        let outcome = solver.solve(root).unwrap();
        assert_eq!(outcome.action, Action::BasicSynthesis);
        assert_eq!(outcome.value.full_quality_completion_probability, 1.0);
        assert!(outcome.stats.pruned_actions >= 1);
    }

    #[test]
    fn optimistic_pareto_bound_does_not_merge_touch_and_synthesis_into_one_step() {
        let mut model = model();
        model.settings.max_progress = 100;
        model.settings.max_quality = 100;
        model.settings.base_progress = 100;
        model.settings.base_quality = 100;
        model.settings.allowed_actions = ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::BasicTouch);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 1;
        model.max_decisions = 1;
        let post = PostDecisionState {
            simulation: SimulationState::new(&model.settings),
            transition: ConditionTransition::Random,
            step: 0,
            decisions: 0,
        };
        let solver = StochasticPolicySolver::new(
            model,
            100,
            [Action::BasicSynthesis, Action::BasicTouch],
            10_000,
        )
        .unwrap();
        assert_eq!(
            solver
                .post_decision_upper_bound(post)
                .full_quality_completion_probability,
            0.0
        );

        model.max_steps = 2;
        model.max_decisions = 2;
        let solver = StochasticPolicySolver::new(
            model,
            100,
            [Action::BasicSynthesis, Action::BasicTouch],
            10_000,
        )
        .unwrap();
        assert_eq!(
            solver
                .post_decision_upper_bound(post)
                .full_quality_completion_probability,
            1.0
        );
    }

    #[test]
    fn optimistic_quality_bound_tracks_inner_quiet_and_byregot_consumption() {
        let mut model = model();
        model.settings.allowed_actions = ActionMask::none().add(Action::ByregotsBlessing);
        model.max_steps = 2;
        model.max_decisions = 2;
        let frontiers = optimistic_step_frontiers(
            &model,
            model.settings.max_quality,
            &[Action::ByregotsBlessing],
            &(0..CONDITION_COUNT)
                .map(condition_from_index)
                .collect::<Vec<_>>(),
        );
        let maximum_quality = |steps: usize,
                               inner_quiet: u8,
                               great_strides: bool,
                               innovation: bool,
                               muscle_memory: bool,
                               veneration: bool| {
            frontiers[steps][optimistic_state_index(
                inner_quiet,
                great_strides,
                innovation,
                muscle_memory,
                veneration,
            )]
            .iter()
            .map(|value| value.quality)
            .max()
            .unwrap_or(0)
        };

        assert_eq!(maximum_quality(1, 0, false, false, false, false), 0);
        assert!(maximum_quality(1, 10, true, true, false, false) > 0);
        assert_eq!(
            maximum_quality(2, 10, true, true, false, false),
            maximum_quality(1, 10, true, true, false, false)
        );
    }

    #[test]
    fn random_condition_bound_excludes_zero_probability_excellent() {
        let mut model = model();
        model.settings.max_quality = 1_000;
        model.settings.base_quality = 100;
        model.settings.allowed_actions = ActionMask::none().add(Action::BasicTouch);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 1;
        model.max_decisions = 1;
        let solver = StochasticPolicySolver::new(model, 1_000, [Action::BasicTouch], 10).unwrap();
        let mut normal = state(Condition::Normal);
        normal.simulation = SimulationState::new(&model.settings);
        normal.simulation.effects = normal.simulation.effects.with_inner_quiet(10);
        normal.step = 0;
        normal.decisions = 0;
        let excellent = DecisionState {
            condition: Condition::Excellent,
            ..normal
        };
        let random = PostDecisionState {
            simulation: normal.simulation,
            transition: ConditionTransition::Random,
            step: 0,
            decisions: 0,
        };

        let normal_bound = solver.decision_upper_bound(normal);
        let excellent_bound = solver.decision_upper_bound(excellent);
        let random_bound = solver.post_decision_upper_bound(random);
        assert_eq!(random_bound.expected_quality, normal_bound.expected_quality);
        assert!(excellent_bound.expected_quality > normal_bound.expected_quality);
    }

    #[test]
    fn exact_bellman_search_reports_expansion_limit_without_partial_policy_claim() {
        let mut model = model();
        model.settings.allowed_actions = ActionMask::none().add(Action::Observe);
        model.condition_probabilities_bps = [10_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        model.max_steps = 2;
        model.max_decisions = 2;
        let mut solver = StochasticPolicySolver::new(model, 100, [Action::Observe], 1).unwrap();
        let mut root = state(Condition::Normal);
        root.step = 0;
        root.decisions = 0;

        assert_eq!(
            solver.solve(root),
            Err(StochasticSolveError::ExpansionLimit)
        );
    }
}
