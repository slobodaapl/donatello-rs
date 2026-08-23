use std::collections::{BTreeSet, hash_map::Entry};

use raphael_sim::{Action, Condition, SimulationState};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::{
    SolverException, SolverSettings,
    actions::{ActionCombo, use_action_combo, use_action_combo_with_condition},
};

use super::{FirstActionPreference, pareto_front::ParetoFront};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SearchScore {
    pub quality_upper_bound: u16,
    pub steps_lower_bound: u8,
    pub duration_lower_bound: u8,
    pub current_steps: u8,
    pub current_duration: u8,
}

impl SearchScore {
    pub const MIN: Self = Self {
        quality_upper_bound: 0,
        steps_lower_bound: u8::MAX,
        duration_lower_bound: u8::MAX,
        current_steps: u8::MAX,
        current_duration: u8::MAX,
    };

    pub const MAX: Self = Self {
        quality_upper_bound: u16::MAX,
        steps_lower_bound: 0,
        duration_lower_bound: 0,
        current_steps: 0,
        current_duration: 0,
    };
}

impl std::cmp::PartialOrd for SearchScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(std::cmp::Ord::cmp(self, other))
    }
}

impl std::cmp::Ord for SearchScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.quality_upper_bound
            .cmp(&other.quality_upper_bound)
            .then(other.steps_lower_bound.cmp(&self.steps_lower_bound))
            .then(other.duration_lower_bound.cmp(&self.duration_lower_bound))
            .then(other.current_steps.cmp(&self.current_steps))
            .then(other.current_duration.cmp(&self.current_duration))
    }
}

#[cfg(target_pointer_width = "32")]
#[bitfield_struct::bitfield(u32)]
struct SearchNode {
    #[bits(26)]
    parent_idx: usize,
    #[bits(6)]
    action: ActionCombo,
}

#[cfg(target_pointer_width = "64")]
#[bitfield_struct::bitfield(u64)]
struct SearchNode {
    #[bits(58)]
    parent_idx: usize,
    #[bits(6)]
    action: ActionCombo,
}

#[derive(Debug)]
pub struct Batch {
    pub score: SearchScore,
    pub nodes: Vec<(SimulationState, Condition, usize)>,
}

#[derive(Debug, Clone)]
pub struct QueueCandidate {
    pub score: SearchScore,
    pub actions: SmallVec<[ActionCombo; 4]>,
    pub parent_idx: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SearchQueueStats {
    pub inserted_nodes: usize,
    pub processed_nodes: usize,
}

pub struct SearchQueue {
    settings: SolverSettings,
    pareto_fronts: FxHashMap<Condition, ParetoFront>,
    preferred_pareto_fronts: Option<FxHashMap<(Condition, FirstActionPreference), ParetoFront>>,
    batch_ordering: BTreeSet<SearchScore>,
    batches: FxHashMap<SearchScore, Vec<SearchNode>>,
    visited_nodes: Vec<SearchNode>,
    /// Cached reconstructed states aligned with `visited_nodes`. Survivors store
    /// the live `(state, condition)`; prefix-only nodes are `None`.
    visited_states: Vec<Option<(SimulationState, Condition)>>,
    visited_first_action_preferences: Option<Vec<FirstActionPreference>>,
    visited_has_action: Option<Vec<bool>>,
    num_inserted_nodes: usize,
    initial_state: SimulationState,
    initial_condition: Condition,
}

impl SearchQueue {
    pub fn new(
        settings: SolverSettings,
        initial_state: SimulationState,
        initial_condition: Condition,
        prefer_quality_first: bool,
    ) -> Self {
        Self::with_anchor(
            settings,
            initial_state,
            initial_condition,
            prefer_quality_first,
            true,
        )
    }

    pub(crate) fn seeded(
        settings: SolverSettings,
        initial_state: SimulationState,
        initial_condition: Condition,
        prefer_quality_first: bool,
    ) -> Self {
        Self::with_anchor(
            settings,
            initial_state,
            initial_condition,
            prefer_quality_first,
            false,
        )
    }

    fn with_anchor(
        settings: SolverSettings,
        initial_state: SimulationState,
        initial_condition: Condition,
        prefer_quality_first: bool,
        include_anchor: bool,
    ) -> Self {
        let mut search_queue = Self {
            settings,
            pareto_fronts: FxHashMap::default(),
            preferred_pareto_fronts: prefer_quality_first.then(FxHashMap::default),
            batch_ordering: BTreeSet::default(),
            batches: FxHashMap::default(),
            visited_nodes: vec![
                SearchNode::new()
                    .with_parent_idx(0)
                    .with_action(ActionCombo::None),
            ],
            visited_states: vec![Some((initial_state, initial_condition))],
            visited_first_action_preferences: prefer_quality_first
                .then(|| vec![FirstActionPreference::Other]),
            visited_has_action: prefer_quality_first.then(|| vec![false]),
            num_inserted_nodes: 0,
            initial_state,
            initial_condition,
        };
        if include_anchor {
            let _ = search_queue.push(SearchScore::MAX, ActionCombo::None, 0);
        }
        search_queue
    }

    /// Add a reconstructed prefix as another root lane.
    pub fn seed_prefix(
        &mut self,
        actions: &[Action],
        score: SearchScore,
    ) -> Result<(), SolverException> {
        let mut parent_idx = 0;
        for &action in actions {
            let first_action_preference =
                self.first_action_preference(parent_idx, ActionCombo::Single(action));
            let node = SearchNode::new()
                .with_parent_idx_checked(parent_idx)
                .map_err(|_| SolverException::SearchQueueCapacityExceeded)?
                .with_action(ActionCombo::Single(action));
            self.push_visited(node, None, first_action_preference);
            parent_idx = self.visited_nodes.len() - 1;
        }
        self.push(score, ActionCombo::None, parent_idx)
    }

    pub fn push(
        &mut self,
        score: SearchScore,
        action: ActionCombo,
        parent_idx: usize,
    ) -> Result<(), SolverException> {
        self.push_transition(score, &[action], parent_idx)
    }

    pub fn push_transition(
        &mut self,
        score: SearchScore,
        actions: &[ActionCombo],
        parent_idx: usize,
    ) -> Result<(), SolverException> {
        let node = self.transition_node(actions, parent_idx)?;
        match self.batches.entry(score) {
            Entry::Occupied(occupied_entry) => {
                occupied_entry.into_mut().push(node);
            }
            Entry::Vacant(vacant_entry) => {
                self.batch_ordering.insert(score);
                vacant_entry.insert(vec![node]);
            }
        }
        self.num_inserted_nodes += 1;
        Ok(())
    }

    pub fn push_batch(
        &mut self,
        mut candidates: Vec<QueueCandidate>,
    ) -> Result<(), SolverException> {
        candidates.par_sort_unstable_by_key(|candidate| candidate.score);
        let mut candidates = candidates.into_iter().peekable();
        while let Some(candidate) = candidates.next() {
            let score = candidate.score;
            let mut nodes = vec![self.transition_node(&candidate.actions, candidate.parent_idx)?];
            while candidates
                .peek()
                .is_some_and(|candidate| candidate.score == score)
            {
                let candidate = candidates.next().unwrap();
                nodes.push(self.transition_node(&candidate.actions, candidate.parent_idx)?);
            }
            let node_count = nodes.len();
            match self.batches.entry(score) {
                Entry::Occupied(entry) => entry.into_mut().extend(nodes),
                Entry::Vacant(entry) => {
                    self.batch_ordering.insert(score);
                    entry.insert(nodes);
                }
            }
            self.num_inserted_nodes += node_count;
        }
        Ok(())
    }

    fn transition_node(
        &mut self,
        actions: &[ActionCombo],
        mut parent_idx: usize,
    ) -> Result<SearchNode, SolverException> {
        let (&last, prefix) = actions
            .split_last()
            .expect("search transitions must contain at least one action");
        for &action in prefix {
            let first_action_preference = self.first_action_preference(parent_idx, action);
            let node = SearchNode::new()
                .with_parent_idx_checked(parent_idx)
                .map_err(|_| SolverException::SearchQueueCapacityExceeded)?
                .with_action(action);
            self.push_visited(node, None, first_action_preference);
            parent_idx = self.visited_nodes.len() - 1;
        }
        SearchNode::new()
            .with_parent_idx_checked(parent_idx)
            .map_err(|_| SolverException::SearchQueueCapacityExceeded)
            .map(|node| node.with_action(last))
    }

    pub fn drop_nodes_below_score(&mut self, min_score: SearchScore) {
        let mut dropped = 0;
        while let Some(&score) = self.batch_ordering.first()
            && score < min_score
        {
            self.batch_ordering.pop_first();
            dropped += self.batches.remove(&score).map_or(0, |batch| batch.len());
        }
        if dropped != 0 {
            log::trace!("{dropped} nodes dropped ({min_score:?})");
        }
    }

    pub fn pop_batch(&mut self) -> Option<Batch> {
        if let Some(score) = self.batch_ordering.pop_last()
            && let Some(batch) = self.batches.remove(&score)
        {
            // Replay only from the nearest cached ancestor. Survivors store their
            // reconstructed state, so typical pops apply a single action.
            let batch: Vec<(SearchNode, SimulationState, Condition)> = batch
                .into_par_iter()
                .map(|search_node| {
                    let (state, condition) = self.reconstruct_node(search_node);
                    (search_node, state, condition)
                })
                .collect();
            // Prefix phase is part of state identity. Once Normal is reached, all paths share the
            // same Pareto front and collapse into Raphael's ordinary state space.
            let mut non_dominated_nodes = Vec::new();
            if self.preferred_pareto_fronts.is_some() {
                let mut by_condition_and_preference =
                    FxHashMap::<(Condition, FirstActionPreference), Vec<_>>::default();
                for node in batch {
                    let preference =
                        self.first_action_preference(node.0.parent_idx(), node.0.action());
                    by_condition_and_preference
                        .entry((node.2, preference))
                        .or_default()
                        .push(node);
                }
                let Some(preferred_pareto_fronts) = self.preferred_pareto_fronts.as_mut() else {
                    unreachable!("preferred Pareto fronts were checked above");
                };
                for (key, nodes) in by_condition_and_preference {
                    non_dominated_nodes.extend(
                        preferred_pareto_fronts
                            .entry(key)
                            .or_default()
                            .insert_batch(
                                nodes,
                                |expanded_node| &expanded_node.1,
                                score.current_steps,
                                score.current_duration,
                            ),
                    );
                }
            } else {
                let mut by_condition = FxHashMap::<Condition, Vec<_>>::default();
                for node in batch {
                    by_condition.entry(node.2).or_default().push(node);
                }
                for (condition, nodes) in by_condition {
                    non_dominated_nodes.extend(
                        self.pareto_fronts
                            .entry(condition)
                            .or_default()
                            .insert_batch(
                                nodes,
                                |expanded_node| &expanded_node.1,
                                score.current_steps,
                                score.current_duration,
                            ),
                    );
                }
            }
            let batch = Batch {
                score,
                nodes: non_dominated_nodes
                    .iter()
                    .enumerate()
                    .map(|(idx, node)| {
                        let state = node.1;
                        let self_idx = self.visited_nodes.len() + idx;
                        (state, node.2, self_idx)
                    })
                    .collect(),
            };
            for expanded_node in non_dominated_nodes {
                let first_action_preference = self.first_action_preference(
                    expanded_node.0.parent_idx(),
                    expanded_node.0.action(),
                );
                self.push_visited(
                    expanded_node.0,
                    Some((expanded_node.1, expanded_node.2)),
                    first_action_preference,
                );
            }
            Some(batch)
        } else {
            None
        }
    }

    fn push_visited(
        &mut self,
        node: SearchNode,
        state: Option<(SimulationState, Condition)>,
        first_action_preference: FirstActionPreference,
    ) {
        self.visited_nodes.push(node);
        self.visited_states.push(state);
        if let Some(preferences) = self.visited_first_action_preferences.as_mut() {
            preferences.push(first_action_preference);
        }
        if let Some(has_action) = self.visited_has_action.as_mut() {
            has_action.push(has_action[node.parent_idx()] || !node.action().actions().is_empty());
        }
        debug_assert_eq!(self.visited_nodes.len(), self.visited_states.len());
        debug_assert!(
            self.visited_first_action_preferences
                .as_ref()
                .is_none_or(|preferences| self.visited_nodes.len() == preferences.len())
        );
        debug_assert!(
            self.visited_has_action
                .as_ref()
                .is_none_or(|has_action| self.visited_nodes.len() == has_action.len())
        );
    }

    fn first_action_preference(
        &self,
        parent_idx: usize,
        action: ActionCombo,
    ) -> FirstActionPreference {
        let Some(has_action) = self.visited_has_action.as_ref() else {
            return FirstActionPreference::Other;
        };
        if has_action[parent_idx] {
            return self.visited_first_action_preferences.as_ref().unwrap()[parent_idx];
        }
        let Some(&first_action) = action.actions().first() else {
            return FirstActionPreference::Other;
        };
        let Ok(next) = self.initial_state.use_action(
            first_action,
            self.initial_condition,
            &self.settings.simulator_settings,
        ) else {
            return FirstActionPreference::Other;
        };
        if next.quality > self.initial_state.quality {
            FirstActionPreference::Quality
        } else if next.progress > self.initial_state.progress {
            FirstActionPreference::Progress
        } else {
            FirstActionPreference::Other
        }
    }

    pub(super) fn retained_first_action_preference(
        &self,
        node_idx: usize,
    ) -> FirstActionPreference {
        self.visited_first_action_preferences
            .as_ref()
            .map_or(FirstActionPreference::Other, |preferences| {
                preferences[node_idx]
            })
    }

    fn reconstruct_node(&self, node: SearchNode) -> (SimulationState, Condition) {
        let (state, condition) = self.state_at(node.parent_idx());
        execute_search_action(&self.settings, state, condition, node.action())
    }

    fn state_at(&self, mut idx: usize) -> (SimulationState, Condition) {
        let mut suffix = SmallVec::<[ActionCombo; 8]>::new();
        loop {
            if let Some(cached) = self.visited_states.get(idx).and_then(|state| *state) {
                let mut state = cached.0;
                let mut condition = cached.1;
                for action in suffix.into_iter().rev() {
                    (state, condition) =
                        execute_search_action(&self.settings, state, condition, action);
                }
                return (state, condition);
            }
            if idx == 0 {
                let mut state = self.initial_state;
                let mut condition = self.initial_condition;
                for action in suffix.into_iter().rev() {
                    (state, condition) =
                        execute_search_action(&self.settings, state, condition, action);
                }
                return (state, condition);
            }
            suffix.push(self.visited_nodes[idx].action());
            idx = self.visited_nodes[idx].parent_idx();
        }
    }

    pub fn get_actions_from_node_idx(&self, mut idx: usize) -> SmallVec<[ActionCombo; 56]> {
        let mut actions = SmallVec::new();
        while idx > 0 {
            let search_node = self.visited_nodes[idx];
            actions.push(search_node.action());
            idx = search_node.parent_idx();
        }
        actions.reverse();
        actions
    }

    pub fn runtime_stats(&self) -> SearchQueueStats {
        SearchQueueStats {
            inserted_nodes: self.num_inserted_nodes,
            processed_nodes: self.visited_nodes.len(),
        }
    }
}

fn execute_search_action(
    settings: &SolverSettings,
    state: SimulationState,
    condition: Condition,
    action: ActionCombo,
) -> (SimulationState, Condition) {
    if condition == Condition::Normal {
        (
            use_action_combo(settings, state, action).unwrap(),
            Condition::Normal,
        )
    } else {
        use_action_combo_with_condition(settings, state, action, condition).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raphael_sim::{Action, ActionMask, Effects, Settings};

    fn settings() -> SolverSettings {
        SolverSettings {
            simulator_settings: Settings {
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
            },
            allow_non_max_quality_solutions: true,
        }
    }

    #[test]
    fn reconstruction_replays_zero_step_and_excellent_poor_phases() {
        let settings = settings();
        let root = SimulationState {
            effects: Effects::initial(&settings.simulator_settings)
                .with_heart_and_soul_available(true),
            ..SimulationState::new(&settings.simulator_settings)
        };
        let mut queue = SearchQueue::new(settings, root, Condition::Excellent, false);
        let initial = queue.pop_batch().unwrap().nodes[0];
        assert_eq!(initial.1, Condition::Excellent);

        queue
            .push(
                SearchScore::MAX,
                ActionCombo::Single(Action::HeartAndSoul),
                initial.2,
            )
            .unwrap();
        let after_zero_step = queue.pop_batch().unwrap().nodes[0];
        assert_eq!(after_zero_step.1, Condition::Excellent);
        assert!(after_zero_step.0.effects.heart_and_soul_active());

        queue
            .push(
                SearchScore::MAX,
                ActionCombo::Single(Action::BasicTouch),
                after_zero_step.2,
            )
            .unwrap();
        let poor = queue.pop_batch().unwrap().nodes[0];
        assert_eq!(poor.1, Condition::Poor);
        assert_eq!(poor.0.effects.combo(), raphael_sim::Combo::BasicTouch);

        queue
            .push(
                SearchScore::MAX,
                ActionCombo::Single(Action::StandardTouch),
                poor.2,
            )
            .unwrap();
        let normal = queue.pop_batch().unwrap().nodes[0];
        assert_eq!(normal.1, Condition::Normal);
    }

    #[test]
    fn queued_stellar_transition_reconstructs_every_concrete_action() {
        let mut settings = settings();
        settings.simulator_settings.stellar_steady_hand_charges = 1;
        let root = SimulationState::new(&settings.simulator_settings);
        let mut queue = SearchQueue::new(settings, root, Condition::Normal, false);
        let anchor = queue.pop_batch().unwrap().nodes[0];
        let actions = [
            ActionCombo::Single(Action::StellarSteadyHand),
            ActionCombo::Single(Action::HastyTouch),
            ActionCombo::Single(Action::DaringTouch),
            ActionCombo::Single(Action::HastyTouch),
        ];
        queue
            .push_transition(SearchScore::MAX, &actions, anchor.2)
            .unwrap();
        let endpoint = queue.pop_batch().unwrap().nodes[0];
        let reconstructed = queue
            .get_actions_from_node_idx(endpoint.2)
            .iter()
            .flat_map(|action| action.actions().iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(
            reconstructed,
            [
                Action::StellarSteadyHand,
                Action::HastyTouch,
                Action::DaringTouch,
                Action::HastyTouch,
            ]
        );
        assert_eq!(endpoint.0.effects.stellar_steady_hand(), 0);
        assert!(endpoint.0.quality > 0);
    }

    #[test]
    fn seeded_prefix_does_not_replace_unrestricted_anchor() {
        let settings = settings();
        let root = SimulationState::new(&settings.simulator_settings);
        let mut queue = SearchQueue::new(settings, root, Condition::Normal, false);
        queue
            .seed_prefix(
                &[Action::BasicSynthesis],
                SearchScore {
                    quality_upper_bound: settings.max_quality(),
                    steps_lower_bound: 1,
                    duration_lower_bound: 3,
                    current_steps: 1,
                    current_duration: 3,
                },
            )
            .unwrap();

        let anchor = queue.pop_batch().unwrap();
        assert_eq!(anchor.nodes.len(), 1);
        assert_eq!(anchor.nodes[0].0, root);
        let seed = queue.pop_batch().unwrap();
        assert!(seed.nodes[0].0.progress > root.progress);
    }
}
