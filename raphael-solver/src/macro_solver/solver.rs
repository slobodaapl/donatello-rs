use raphael_sim::*;
use rayon::prelude::*;

use super::search_queue::{SearchQueueStats, SearchScore};
use crate::actions::{
    ActionCombo, FULL_SEARCH_ACTIONS, final_appraisal_is_dominated, stellar_window_transitions,
    use_action_combo, use_action_combo_with_condition,
};
use crate::finish_solver::{FinishSolverStats, Finishability};
use crate::macro_solver::search_queue::{Batch, QueueCandidate, SearchQueue};
use crate::quality_upper_bound_solver::{
    QualityUbSolverShard, QualityUbSolverStats, QualityUbStates,
};
use crate::step_lower_bound_solver::{StepLbSolverShard, StepLbSolverStats, StepLbStates};
use crate::utils::AtomicFlag;
use crate::utils::ScopedTimer;
use crate::{
    FinishSolver, ProgressFrontierSolver, ProgressPolicy, ProgressTarget, QualityUbSolver,
    SolverException, SolverSettings, StepLbSolver,
};

use smallvec::{SmallVec, smallvec};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::vec::Vec;
use strum::IntoEnumIterator;

use super::FirstActionPreference;

#[derive(Clone)]
struct Solution {
    score: (SearchScore, u16),
    solver_actions: Vec<ActionCombo>,
    first_action_preference: FirstActionPreference,
}

impl Solution {
    fn actions(&self) -> Vec<Action> {
        let mut actions = Vec::new();
        for solver_action in &self.solver_actions {
            actions.extend_from_slice(solver_action.actions());
        }
        actions
    }
}

type SolutionCallback<'a> = dyn Fn(&[Action]) + Send + 'a;
type ProgressCallback<'a> = dyn Fn(usize) + Send + 'a;

#[derive(Debug, Default, Clone, Copy)]
pub struct MacroSolverStats {
    pub search_queue_stats: SearchQueueStats,
    pub finish_solver_stats: FinishSolverStats,
    pub quality_ub_stats: QualityUbSolverStats,
    pub step_lb_stats: StepLbSolverStats,
}

#[derive(Debug, Clone)]
pub struct MacroSolveOutcome {
    pub actions: Vec<Action>,
    pub optimal: bool,
    pub quality: u16,
    pub quality_upper_bound: u16,
    pub stats: MacroSolverStats,
}

pub struct MacroSolver<'a> {
    settings: SolverSettings,
    solution_callback: Box<SolutionCallback<'a>>,
    progress_callback: Box<ProgressCallback<'a>>,
    finish_solver: FinishSolver,
    quality_ub_solver: QualityUbSolver,
    step_lb_solver: StepLbSolver,
    interrupt_signal: AtomicFlag,
    last_solve_runtime_stats: MacroSolverStats,
    improved_solution_found: Arc<AtomicBool>,
    complete_solution_found: Arc<AtomicBool>,
    progress_frontier_cache: Option<ProgressFrontierCache>,
}

struct ProgressFrontierCache {
    root: SimulationState,
    condition: Condition,
    completion: Option<Vec<Action>>,
    one_short_prefixes: Vec<Vec<Action>>,
    one_short_attempted: bool,
}

struct OneShortWorker {
    interrupt: AtomicFlag,
    handle: Option<std::thread::JoinHandle<Vec<Vec<Action>>>>,
}

impl OneShortWorker {
    const EXPANSION_LIMIT: usize = 100;

    fn start(settings: SolverSettings, root: SimulationState, condition: Condition) -> Self {
        let interrupt = AtomicFlag::new();
        let worker_interrupt = interrupt.clone();
        let handle = std::thread::spawn(move || {
            ProgressFrontierSolver::new(settings, worker_interrupt)
                .solve_with_expansion_limit(
                    root,
                    condition,
                    ProgressTarget::OneShort,
                    ProgressPolicy::Pareto,
                    Some(Self::EXPANSION_LIMIT),
                )
                .into_iter()
                .map(|endpoint| endpoint.actions)
                .collect()
        });
        Self {
            interrupt,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> Vec<Vec<Action>> {
        self.handle
            .take()
            .and_then(|handle| handle.join().ok())
            .unwrap_or_default()
    }
}

impl Drop for OneShortWorker {
    fn drop(&mut self) {
        self.interrupt.set();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl<'a> MacroSolver<'a> {
    pub fn new(
        settings: SolverSettings,
        solution_callback: Box<SolutionCallback<'a>>,
        progress_callback: Box<ProgressCallback<'a>>,
        interrupt_signal: AtomicFlag,
    ) -> Self {
        let quality_ub_solver = QualityUbSolver::new(settings, interrupt_signal.clone());
        let step_lb_solver = StepLbSolver::new(settings, interrupt_signal.clone());
        let mut finish_solver = FinishSolver::new(settings);
        finish_solver.set_interrupt_signal(interrupt_signal.clone());
        Self {
            settings,
            solution_callback,
            progress_callback,
            finish_solver,
            quality_ub_solver,
            step_lb_solver,
            interrupt_signal,
            last_solve_runtime_stats: MacroSolverStats::default(),
            improved_solution_found: Arc::new(AtomicBool::new(false)),
            complete_solution_found: Arc::new(AtomicBool::new(false)),
            progress_frontier_cache: None,
        }
    }

    pub fn solve(&mut self) -> Result<Vec<Action>, SolverException> {
        self.solve_from_state(SimulationState::new(&self.settings.simulator_settings))
    }

    pub fn set_interrupt_signal(&mut self, interrupt_signal: AtomicFlag) {
        self.interrupt_signal = interrupt_signal;
        self.finish_solver
            .set_interrupt_signal(self.interrupt_signal.clone());
        self.quality_ub_solver
            .set_interrupt_signal(self.interrupt_signal.clone());
        self.step_lb_solver
            .set_interrupt_signal(self.interrupt_signal.clone());
    }

    pub fn improved_solution_signal(&self) -> Arc<AtomicBool> {
        self.improved_solution_found.clone()
    }

    pub fn complete_solution_signal(&self) -> Arc<AtomicBool> {
        self.complete_solution_found.clone()
    }

    /// Solve the complete remaining synthesis from an authoritative live state.
    /// The state's current resources, progress, quality, effects, charges, and combo are preserved.
    pub fn solve_from_state(
        &mut self,
        initial_state: SimulationState,
    ) -> Result<Vec<Action>, SolverException> {
        self.solve_from_state_with_condition(initial_state, Condition::Normal)
    }

    /// Solve from a live state while applying the already-observed deterministic condition
    /// prefix action-by-action. Future unknown conditions become Normal.
    pub fn solve_from_state_with_condition(
        &mut self,
        initial_state: SimulationState,
        condition: Condition,
    ) -> Result<Vec<Action>, SolverException> {
        self.solve_from_state_with_condition_and_incumbent(initial_state, condition, &[])
    }

    /// Solve from a live state, using a known completing suffix as the initial branch-and-bound
    /// floor. Invalid or non-completing incumbents are ignored.
    pub fn solve_from_state_with_condition_and_incumbent(
        &mut self,
        initial_state: SimulationState,
        condition: Condition,
        incumbent_actions: &[Action],
    ) -> Result<Vec<Action>, SolverException> {
        self.solve_from_state_with_condition_and_incumbent_objective(
            initial_state,
            condition,
            incumbent_actions,
            true,
        )
    }

    /// Solve from a live state. When `minimize_steps` is false, secondary step/duration
    /// optimization is disabled and the first proven maximum-quality solution terminates search.
    pub fn solve_from_state_with_condition_and_incumbent_objective(
        &mut self,
        initial_state: SimulationState,
        condition: Condition,
        incumbent_actions: &[Action],
        minimize_steps: bool,
    ) -> Result<Vec<Action>, SolverException> {
        self.solve_from_state_with_condition_and_incumbent_anytime(
            initial_state,
            condition,
            incumbent_actions,
            minimize_steps,
        )
        .map(|outcome| outcome.actions)
    }

    pub fn solve_from_state_with_condition_and_incumbent_anytime(
        &mut self,
        initial_state: SimulationState,
        condition: Condition,
        incumbent_actions: &[Action],
        minimize_steps: bool,
    ) -> Result<MacroSolveOutcome, SolverException> {
        self.solve_from_state_with_condition_and_incumbent_anytime_preference(
            initial_state,
            condition,
            incumbent_actions,
            minimize_steps,
            false,
        )
    }

    pub fn solve_from_state_with_condition_and_incumbent_anytime_preference(
        &mut self,
        initial_state: SimulationState,
        condition: Condition,
        incumbent_actions: &[Action],
        minimize_steps: bool,
        prefer_quality_first: bool,
    ) -> Result<MacroSolveOutcome, SolverException> {
        self.solve_from_state_with_condition_and_lanes_anytime(
            initial_state,
            condition,
            incumbent_actions,
            minimize_steps,
            prefer_quality_first,
            None,
        )
    }

    /// Optimize only continuations of the supplied exact progress prefixes. The unrestricted
    /// root is intentionally excluded; callers must supply a validated completing incumbent.
    pub fn solve_from_state_with_condition_and_seeded_prefixes_anytime(
        &mut self,
        initial_state: SimulationState,
        condition: Condition,
        incumbent_actions: &[Action],
        minimize_steps: bool,
        prefixes: Vec<Vec<Action>>,
    ) -> Result<MacroSolveOutcome, SolverException> {
        self.solve_from_state_with_condition_and_seeded_prefixes_anytime_preference(
            initial_state,
            condition,
            incumbent_actions,
            minimize_steps,
            false,
            prefixes,
        )
    }

    pub fn solve_from_state_with_condition_and_seeded_prefixes_anytime_preference(
        &mut self,
        initial_state: SimulationState,
        condition: Condition,
        incumbent_actions: &[Action],
        minimize_steps: bool,
        prefer_quality_first: bool,
        prefixes: Vec<Vec<Action>>,
    ) -> Result<MacroSolveOutcome, SolverException> {
        self.solve_from_state_with_condition_and_lanes_anytime(
            initial_state,
            condition,
            incumbent_actions,
            minimize_steps,
            prefer_quality_first,
            Some(prefixes),
        )
    }

    fn solve_from_state_with_condition_and_lanes_anytime(
        &mut self,
        initial_state: SimulationState,
        condition: Condition,
        incumbent_actions: &[Action],
        minimize_steps: bool,
        prefer_quality_first: bool,
        seeded_prefixes: Option<Vec<Vec<Action>>>,
    ) -> Result<MacroSolveOutcome, SolverException> {
        log::debug!(
            "rayon::current_num_threads() = {}",
            rayon::current_num_threads()
        );

        self.last_solve_runtime_stats = MacroSolverStats::default();
        self.improved_solution_found.store(false, Ordering::Release);
        self.complete_solution_found.store(false, Ordering::Release);
        let _total_time = ScopedTimer::new("Total Time");

        let seeded_only = seeded_prefixes.is_some();
        let mut incumbent = self.incumbent_solution(
            initial_state,
            condition,
            incumbent_actions,
            prefer_quality_first,
        );
        if !seeded_only {
            let completion = self
                .progress_frontier_cache
                .as_ref()
                .filter(|cache| cache.root == initial_state && cache.condition == condition)
                .and_then(|cache| cache.completion.clone());
            let completion = if completion.is_some() {
                completion
            } else {
                let frontier_solver =
                    ProgressFrontierSolver::new(self.settings, self.interrupt_signal.clone());
                let completion = frontier_solver
                    .solve_with_expansion_limit(
                        initial_state,
                        condition,
                        ProgressTarget::Complete,
                        ProgressPolicy::Fastest,
                        Some(100_000),
                    )
                    .into_iter()
                    .next()
                    .map(|endpoint| endpoint.actions);
                if completion.is_some() {
                    self.progress_frontier_cache = Some(ProgressFrontierCache {
                        root: initial_state,
                        condition,
                        completion: completion.clone(),
                        one_short_prefixes: Vec::new(),
                        one_short_attempted: false,
                    });
                }
                completion
            };
            if let Some(actions) = completion {
                let completion = self
                    .incumbent_solution(initial_state, condition, &actions, prefer_quality_first)
                    .expect("progress frontier returned a non-completing endpoint");
                if incumbent.as_ref().is_none_or(|current| {
                    completion.score > current.score
                        || prefer_quality_first
                            && completion.score == current.score
                            && completion
                                .first_action_preference
                                .strictly_preferred_to(current.first_action_preference)
                }) {
                    incumbent = Some(completion);
                }
            }
        }
        if incumbent.is_some() {
            self.complete_solution_found.store(true, Ordering::Release);
        }
        let cached_one_short = (!seeded_only)
            .then(|| {
                self.progress_frontier_cache
                    .as_ref()
                    .filter(|cache| cache.root == initial_state && cache.condition == condition)
                    .filter(|cache| cache.one_short_attempted)
                    .map(|cache| cache.one_short_prefixes.clone())
            })
            .flatten();
        let mut one_short_worker =
            (!seeded_only && cached_one_short.is_none() && !self.interrupt_signal.is_set())
                .then(|| OneShortWorker::start(self.settings, initial_state, condition));
        let timer = ScopedTimer::new("Finish Solver");
        if let Err(error) = self.finish_solver.precompute() {
            self.finish_solver.discard_precompute();
            return self.interrupted_outcome(error, incumbent);
        }
        if condition == Condition::Normal
            && self.finish_solver.can_finish(&initial_state)? == Finishability::Impossible
        {
            self.last_solve_runtime_stats.finish_solver_stats = self.finish_solver.runtime_stats();
            if incumbent.is_none() {
                return Err(SolverException::NoSolution);
            }
        }
        drop(timer);

        let timer = ScopedTimer::new("Quality UB Solver");
        if !self.quality_ub_solver.is_precomputed()
            && let Err(error) = self.quality_ub_solver.precompute()
        {
            self.quality_ub_solver.discard_precompute();
            return self.interrupted_outcome(error, incumbent);
        }
        drop(timer);

        // The StepLbSolver is only queried when a state has the potential to reach max_quality.
        // If the quality upper-bound of the initial state is less than max_quality, then no
        // subsequent state can reach max_quality, which in turn means the StepLbSolver is not needed.
        let initial_state_quality_ub = if condition == Condition::Normal {
            let mut shard = self.quality_ub_solver.create_shard();
            let result = match shard.quality_upper_bound_for_progress(initial_state) {
                Ok(result) => result,
                Err(error) => return self.interrupted_outcome(error, incumbent),
            };
            self.quality_ub_solver
                .extend_solved_states(shard.solved_states());
            result
        } else {
            self.settings.max_quality()
        };
        if initial_state_quality_ub >= self.settings.max_quality() {
            let _timer = ScopedTimer::new("Step LB Solver");
            if !self.step_lb_solver.is_precomputed()
                && let Err(error) = self.step_lb_solver.precompute()
            {
                self.step_lb_solver.discard_precompute();
                return self.interrupted_outcome(error, incumbent);
            }
        }
        let one_short_prefixes = if let Some(prefixes) = seeded_prefixes {
            prefixes
        } else if let Some(prefixes) = cached_one_short {
            prefixes
        } else if self.interrupt_signal.is_set() {
            Vec::new()
        } else {
            let prefixes = one_short_worker
                .take()
                .map(OneShortWorker::finish)
                .unwrap_or_default();
            if let Some(cache) = self.progress_frontier_cache.as_mut()
                && cache.root == initial_state
                && cache.condition == condition
            {
                cache.one_short_prefixes.clone_from(&prefixes);
                cache.one_short_attempted = true;
            }
            prefixes
        };

        let timer = ScopedTimer::new("Search");
        let search_queue = if seeded_only {
            SearchQueue::seeded(
                self.settings,
                initial_state,
                condition,
                prefer_quality_first,
            )
        } else {
            SearchQueue::new(
                self.settings,
                initial_state,
                condition,
                prefer_quality_first,
            )
        };
        let (solution, optimal, quality_upper_bound) = self.do_solve(
            search_queue,
            incumbent,
            one_short_prefixes,
            minimize_steps,
            prefer_quality_first,
        )?;
        drop(timer);

        log::debug!("{:?}", self.runtime_stats());

        let quality = solution.score.0.quality_upper_bound;
        Ok(MacroSolveOutcome {
            actions: solution.actions(),
            optimal,
            quality,
            quality_upper_bound: quality_upper_bound.max(quality),
            stats: self.runtime_stats(),
        })
    }

    fn interrupted_outcome(
        &mut self,
        error: SolverException,
        incumbent: Option<Solution>,
    ) -> Result<MacroSolveOutcome, SolverException> {
        if error != SolverException::Interrupted {
            return Err(error);
        }
        let solution = incumbent.ok_or(SolverException::Interrupted)?;
        self.last_solve_runtime_stats = MacroSolverStats {
            search_queue_stats: SearchQueueStats::default(),
            finish_solver_stats: self.finish_solver.runtime_stats(),
            quality_ub_stats: self.quality_ub_solver.runtime_stats(),
            step_lb_stats: self.step_lb_solver.runtime_stats(),
        };
        let quality = solution.score.0.quality_upper_bound;
        Ok(MacroSolveOutcome {
            actions: solution.actions(),
            optimal: false,
            quality,
            quality_upper_bound: self.settings.max_quality(),
            stats: self.runtime_stats(),
        })
    }

    fn incumbent_solution(
        &self,
        mut state: SimulationState,
        mut condition: Condition,
        actions: &[Action],
        prefer_quality_first: bool,
    ) -> Option<Solution> {
        let mut solver_actions = Vec::with_capacity(actions.len());
        let mut duration = 0_u8;
        let mut first_action_preference = FirstActionPreference::Other;
        for &action in actions {
            if state.is_final(&self.settings.simulator_settings) {
                break;
            }
            let previous = state;
            (state, condition) = use_action_combo_with_condition(
                &self.settings,
                state,
                ActionCombo::Single(action),
                condition,
            )
            .ok()?;
            if prefer_quality_first && solver_actions.is_empty() {
                first_action_preference = if state.quality > previous.quality {
                    FirstActionPreference::Quality
                } else if state.progress > previous.progress {
                    FirstActionPreference::Progress
                } else {
                    FirstActionPreference::Other
                };
            }
            duration = duration.checked_add(action.time_cost())?;
            solver_actions.push(ActionCombo::Single(action));
        }
        if state.progress < self.settings.max_progress() {
            return None;
        }
        let steps = u8::try_from(solver_actions.len()).ok()?;
        let score = SearchScore {
            quality_upper_bound: state.quality.min(self.settings.max_quality()),
            steps_lower_bound: steps,
            duration_lower_bound: duration,
            current_steps: steps,
            current_duration: duration,
        };
        Some(Solution {
            score: (score, 0),
            solver_actions,
            first_action_preference,
        })
    }

    fn do_solve(
        &mut self,
        mut search_queue: SearchQueue,
        incumbent: Option<Solution>,
        one_short_prefixes: Vec<Vec<Action>>,
        minimize_steps: bool,
        prefer_quality_first: bool,
    ) -> Result<(Solution, bool, u16), SolverException> {
        for prefix in one_short_prefixes {
            let current_steps = u8::try_from(prefix.len()).unwrap_or(u8::MAX);
            let current_duration = prefix.iter().fold(0_u8, |duration, action| {
                duration.saturating_add(action.time_cost())
            });
            search_queue.seed_prefix(
                &prefix,
                SearchScore {
                    quality_upper_bound: self.settings.max_quality(),
                    steps_lower_bound: current_steps,
                    duration_lower_bound: current_duration,
                    current_steps,
                    current_duration,
                },
            )?;
        }
        let mut solution = incumbent;
        if !minimize_steps
            && solution.as_ref().is_some_and(|solution| {
                solution.score.0.quality_upper_bound >= self.settings.max_quality()
                    && (!prefer_quality_first
                        || solution.first_action_preference != FirstActionPreference::Progress)
            })
        {
            let solution = solution.unwrap();
            return Ok((solution, true, self.settings.max_quality()));
        }
        let score_floor = |solution: &Solution| {
            if minimize_steps {
                solution.score.0
            } else {
                SearchScore {
                    quality_upper_bound: if prefer_quality_first
                        && solution.first_action_preference == FirstActionPreference::Progress
                    {
                        solution.score.0.quality_upper_bound
                    } else {
                        solution.score.0.quality_upper_bound.saturating_add(1)
                    },
                    ..SearchScore::MIN
                }
            }
        };
        let mut min_accepted_score = solution.as_ref().map_or(SearchScore::MIN, score_floor);

        while let Some(Batch {
            score,
            nodes: batch,
        }) = search_queue.pop_batch()
            && score >= min_accepted_score
        {
            if self.interrupt_signal.is_set() {
                let solution = solution.ok_or(SolverException::Interrupted)?;
                return Ok((solution, false, score.quality_upper_bound));
            }

            let create_worker_data = || WorkerData {
                settings: &self.settings,
                interrupt_signal: &self.interrupt_signal,
                finish_solver: &self.finish_solver,
                quality_ub_solver_shard: self.quality_ub_solver.create_shard(),
                step_lb_solver_shard: self.step_lb_solver.create_shard(),
                search_queue: &search_queue,
                min_accepted_score,
                candidate_states: Vec::new(),
                best_intermediate_solution: None,
            };

            let worker_results = batch
                .into_par_iter()
                .try_fold(
                    create_worker_data,
                    |mut worker_data, (state, condition, backtrack_id)| {
                        let first_action_preference =
                            search_queue.retained_first_action_preference(backtrack_id);
                        worker_data.process_state(
                            state,
                            condition,
                            score,
                            backtrack_id,
                            first_action_preference,
                        )?;
                        Ok(worker_data)
                    },
                )
                .collect::<Result<Vec<_>, SolverException>>();
            let worker_results = match worker_results {
                Ok(results) => results,
                Err(SolverException::Interrupted) => {
                    let solution = solution.ok_or(SolverException::Interrupted)?;
                    return Ok((solution, false, score.quality_upper_bound));
                }
                Err(error) => return Err(error),
            };

            // Finalize the workers to drop all shared references to `self` to satisfy the borrow checker.
            let worker_results = worker_results
                .into_iter()
                .map(WorkerData::finalize)
                .collect::<Vec<_>>();

            // Update the current best intermediate solution.
            for worker_data in &worker_results {
                if let Some(worker_solution) = worker_data.best_intermediate_solution.as_ref()
                    && (solution.is_none()
                        || worker_solution.score.0.quality_upper_bound
                            > solution.as_ref().unwrap().score.0.quality_upper_bound
                        || (minimize_steps
                            && Some(worker_solution.score) > solution.as_ref().map(|s| s.score))
                        || (prefer_quality_first
                            && Some(worker_solution.score.0)
                                == solution.as_ref().map(|s| s.score.0)
                            && worker_solution
                                .first_action_preference
                                .strictly_preferred_to(
                                    solution.as_ref().unwrap().first_action_preference,
                                )))
                {
                    solution = Some(worker_solution.clone());
                    self.complete_solution_found.store(true, Ordering::Release);
                    self.improved_solution_found.store(true, Ordering::Release);
                    (self.solution_callback)(&solution.as_ref().unwrap().actions());
                }
            }

            min_accepted_score = worker_results
                .iter()
                .map(|result| result.min_accepted_score)
                .max()
                .unwrap_or(min_accepted_score);
            if !minimize_steps {
                if let Some(best) = solution.as_ref() {
                    min_accepted_score = score_floor(best);
                }
                if solution.as_ref().is_some_and(|solution| {
                    solution.score.0.quality_upper_bound >= self.settings.max_quality()
                        && (!prefer_quality_first
                            || solution.first_action_preference != FirstActionPreference::Progress)
                }) {
                    break;
                }
            }
            search_queue.drop_nodes_below_score(min_accepted_score);

            // Filter once, then group equal scores so the ordered queue is updated once per
            // distinct score instead of once per candidate node.
            let candidate_count = worker_results
                .iter()
                .map(|worker| worker.candidate_states.len())
                .sum::<usize>();
            if candidate_count >= 4096 {
                let candidates = worker_results
                    .iter()
                    .flat_map(|worker| worker.candidate_states.iter())
                    .filter(|candidate| candidate.score >= min_accepted_score)
                    .cloned()
                    .collect();
                search_queue.push_batch(candidates)?;
            } else {
                for candidate in worker_results
                    .iter()
                    .flat_map(|worker| worker.candidate_states.iter())
                    .filter(|candidate| candidate.score >= min_accepted_score)
                {
                    search_queue.push_transition(
                        candidate.score,
                        &candidate.actions,
                        candidate.parent_idx,
                    )?;
                }
            }

            // Extend inner solvers with local states from all workers.
            for worker_result in worker_results {
                self.quality_ub_solver
                    .extend_solved_states(worker_result.quality_ub_states);
                self.step_lb_solver
                    .extend_solved_states(worker_result.step_lb_states);
            }

            (self.progress_callback)(search_queue.runtime_stats().processed_nodes);
        }

        self.last_solve_runtime_stats = MacroSolverStats {
            search_queue_stats: search_queue.runtime_stats(),
            finish_solver_stats: self.finish_solver.runtime_stats(),
            quality_ub_stats: self.quality_ub_solver.runtime_stats(),
            step_lb_stats: self.step_lb_solver.runtime_stats(),
        };

        if let Some(solution) = &solution
            && solution.score.0.quality_upper_bound < self.settings.max_quality()
            && !self.settings.allow_non_max_quality_solutions
        {
            return Err(SolverException::NoSolution);
        }

        let solution = solution.ok_or(SolverException::NoSolution)?;
        let quality_upper_bound = solution.score.0.quality_upper_bound;
        Ok((solution, true, quality_upper_bound))
    }

    pub fn runtime_stats(&self) -> MacroSolverStats {
        self.last_solve_runtime_stats
    }

    pub fn estimated_retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.finish_solver.estimated_retained_bytes()
            + self.quality_ub_solver.estimated_retained_bytes()
            + self.step_lb_solver.estimated_retained_bytes()
    }
}

struct WorkerResult {
    quality_ub_states: QualityUbStates,
    step_lb_states: StepLbStates,
    min_accepted_score: SearchScore,
    candidate_states: Vec<QueueCandidate>,
    best_intermediate_solution: Option<Solution>,
}

struct WorkerData<'main> {
    settings: &'main SolverSettings,
    interrupt_signal: &'main AtomicFlag,
    finish_solver: &'main FinishSolver,
    quality_ub_solver_shard: QualityUbSolverShard<'main>,
    step_lb_solver_shard: StepLbSolverShard<'main>,
    search_queue: &'main SearchQueue,
    min_accepted_score: SearchScore,
    candidate_states: Vec<QueueCandidate>,
    best_intermediate_solution: Option<Solution>,
}

impl WorkerData<'_> {
    fn finalize(self) -> WorkerResult {
        WorkerResult {
            quality_ub_states: self.quality_ub_solver_shard.solved_states(),
            step_lb_states: self.step_lb_solver_shard.solved_states(),
            min_accepted_score: self.min_accepted_score,
            candidate_states: self.candidate_states,
            best_intermediate_solution: self.best_intermediate_solution,
        }
    }

    fn update_min_score(&mut self, score: SearchScore) {
        self.min_accepted_score = std::cmp::max(self.min_accepted_score, score);
    }

    fn add_candidate_state(
        &mut self,
        state: SimulationState,
        score: SearchScore,
        transition_actions: SmallVec<[ActionCombo; 4]>,
        parent_id: usize,
        first_action_preference: FirstActionPreference,
    ) {
        if state.progress >= self.settings.max_progress() {
            if self
                .best_intermediate_solution
                .as_ref()
                .is_none_or(|solution| {
                    solution.score < (score, state.quality)
                        || solution.score == (score, state.quality)
                            && first_action_preference
                                .strictly_preferred_to(solution.first_action_preference)
                })
            {
                let mut actions = self.search_queue.get_actions_from_node_idx(parent_id);
                actions.extend_from_slice(&transition_actions);
                self.best_intermediate_solution = Some(Solution {
                    score: (score, state.quality),
                    solver_actions: actions.into_vec(),
                    first_action_preference,
                });
            }
        } else if score >= self.min_accepted_score {
            self.candidate_states.push(QueueCandidate {
                score,
                actions: transition_actions,
                parent_idx: parent_id,
            });
        }
    }

    fn process_state(
        &mut self,
        state: SimulationState,
        condition: Condition,
        score: SearchScore,
        backtrack_id: usize,
        first_action_preference: FirstActionPreference,
    ) -> Result<(), SolverException> {
        if self.interrupt_signal.is_set() {
            return Err(SolverException::Interrupted);
        }
        let stellar_active = state.effects.stellar_steady_hand() != 0;
        if !stellar_active && condition == Condition::Normal {
            for action in FULL_SEARCH_ACTIONS
                .into_iter()
                .filter(|action| !action.is_stellar_window_action())
                .filter(|action| {
                    *action != ActionCombo::Single(Action::FinalAppraisal)
                        || !final_appraisal_is_dominated(self.settings, state, Condition::Normal)
                })
            {
                if let Ok(state) = use_action_combo(self.settings, state, action) {
                    self.process_child(
                        state,
                        Condition::Normal,
                        score,
                        smallvec![action],
                        backtrack_id,
                        first_action_preference,
                    )?;
                }
            }
        } else if !stellar_active {
            // Prefix expansion uses individual actions so zero-step actions and combo transitions
            // consume conditions exactly as the game does. Future random conditions are absent.
            for action in Action::iter()
                .map(ActionCombo::Single)
                .filter(|action| !action.is_stellar_window_action())
                .filter(|action| {
                    *action != ActionCombo::Single(Action::FinalAppraisal)
                        || !final_appraisal_is_dominated(self.settings, state, condition)
                })
            {
                if let Ok((state, next_condition)) =
                    use_action_combo_with_condition(self.settings, state, action, condition)
                {
                    self.process_child(
                        state,
                        next_condition,
                        score,
                        smallvec![action],
                        backtrack_id,
                        first_action_preference,
                    )?;
                }
            }
        }
        for transition in stellar_window_transitions(self.settings, state, condition, None) {
            self.process_child(
                transition.state,
                transition.condition,
                score,
                transition.actions,
                backtrack_id,
                first_action_preference,
            )?;
        }
        Ok(())
    }

    fn process_child(
        &mut self,
        state: SimulationState,
        condition: Condition,
        score: SearchScore,
        actions: SmallVec<[ActionCombo; 4]>,
        backtrack_id: usize,
        first_action_preference: FirstActionPreference,
    ) -> Result<(), SolverException> {
        let transition_steps = actions.iter().map(|action| action.steps()).sum::<u8>();
        let transition_duration = actions.iter().map(|action| action.duration()).sum::<u8>();
        let current_steps = score.current_steps.saturating_add(transition_steps);
        let current_duration = score.current_duration.saturating_add(transition_duration);
        if state.is_final(&self.settings.simulator_settings) {
            if state.progress >= self.settings.max_progress() {
                let solution_score = SearchScore {
                    quality_upper_bound: state.quality.min(self.settings.max_quality()),
                    steps_lower_bound: current_steps,
                    duration_lower_bound: current_duration,
                    current_steps,
                    current_duration,
                };
                self.update_min_score(solution_score);
                self.add_candidate_state(
                    state,
                    solution_score,
                    actions,
                    backtrack_id,
                    first_action_preference,
                );
            }
            return Ok(());
        }

        if condition != Condition::Normal {
            // Normal-only bounds are not admissible while a special condition remains active.
            let child_score = SearchScore {
                quality_upper_bound: self.settings.max_quality(),
                steps_lower_bound: current_steps,
                duration_lower_bound: current_duration,
                current_steps,
                current_duration,
            };
            self.add_candidate_state(
                state,
                child_score,
                actions,
                backtrack_id,
                first_action_preference,
            );
            return Ok(());
        }

        let finishability = self.finish_solver.can_finish(&state)?;
        if finishability == Finishability::Impossible {
            return Ok(());
        }

        // Only a proven completion path makes current quality a valid global lower bound. Unknown
        // live states must remain searchable without raising the floor from a dead-end branch.
        if finishability == Finishability::Proven {
            self.update_min_score(SearchScore {
                quality_upper_bound: state.quality.min(self.settings.max_quality()),
                ..SearchScore::MIN
            });
        }

        let quality_upper_bound = if state.quality >= self.settings.max_quality() {
            self.settings.max_quality()
        } else {
            std::cmp::min(
                score.quality_upper_bound,
                self.quality_ub_solver_shard.quality_upper_bound(state)?,
            )
        };
        if !self.settings.allow_non_max_quality_solutions
            && quality_upper_bound < self.settings.max_quality()
        {
            return Ok(());
        }

        let step_lb_hint = score.steps_lower_bound.saturating_sub(current_steps);
        let steps_lower_bound = if quality_upper_bound >= self.settings.max_quality() {
            self.step_lb_solver_shard
                .step_lower_bound(state, step_lb_hint)?
                .saturating_add(current_steps)
        } else {
            current_steps
        };
        let child_score = SearchScore {
            quality_upper_bound,
            steps_lower_bound,
            duration_lower_bound: current_duration + 3,
            current_steps,
            current_duration,
        };
        self.add_candidate_state(
            state,
            child_score,
            actions,
            backtrack_id,
            first_action_preference,
        );
        Ok(())
    }
}
