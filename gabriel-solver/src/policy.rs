use std::cell::RefCell;
use std::sync::{Arc, Mutex, OnceLock};

use raphael_sim::{Action, Combo, Condition};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::model::ActorModel;
use crate::sim::{CounterStream, RecipeModel, State, TerminalStatus, derive_seed};

const ACTION_COUNT: usize = 36;
const FEATURE_COUNT: usize = 64;
const POLICY_PILOT_ROLLOUTS: usize = 4;
const POLICY_SCREEN_ROLLOUTS: usize = 16;
const POLICY_FINAL_ROLLOUTS: usize = 64;
const POLICY_FINALISTS: usize = 4;
// Checkpoint input scales only. They neither terminate search nor gate a recipe/action.
const ACTOR_STEP_SCALE: f32 = 55.0;
const ACTOR_DECISION_SCALE: f32 = 64.0;

pub const MAX_WORKER_THREADS: usize = 256;
const DEFAULT_WORKER_THREADS: usize = 4;

struct PoolEntry {
    workers: usize,
    pool: Arc<ThreadPool>,
}

static WORKER_POOL: OnceLock<Mutex<Option<PoolEntry>>> = OnceLock::new();

#[derive(Clone, Copy)]
struct StateScratch {
    state: State,
    unbuffed_progress: u16,
    unbuffed_basic: u16,
    quality_finisher: u16,
    legal: [bool; ACTION_COUNT],
}

thread_local! {
    static STATE_SCRATCH: RefCell<Option<StateScratch>> = const { RefCell::new(None) };
}

const BASIC_SYNTHESIS: usize = 0;
const MASTERS_MEND: usize = 2;
const HASTY_TOUCH: usize = 3;
const RAPID_SYNTHESIS: usize = 4;
const OBSERVE: usize = 5;
const TRICKS_OF_THE_TRADE: usize = 6;
const VENERATION: usize = 8;
const STANDARD_TOUCH: usize = 9;
const GREAT_STRIDES: usize = 10;
const INNOVATION: usize = 11;
const FINAL_APPRAISAL: usize = 12;
const BYREGOT: usize = 14;
const PRECISE_TOUCH: usize = 15;
const MUSCLE_MEMORY: usize = 16;
const CAREFUL_OBSERVATION: usize = 17;
const CAREFUL_SYNTHESIS: usize = 18;
const MANIPULATION: usize = 19;
const ADVANCED_TOUCH: usize = 21;
const REFLECT: usize = 22;
const GROUNDWORK: usize = 24;
const DELICATE_SYNTHESIS: usize = 25;
const INTENSIVE_SYNTHESIS: usize = 26;
const TRAINED_EYE: usize = 27;
const HEART_AND_SOUL: usize = 28;
const PRUDENT_SYNTHESIS: usize = 29;
const TRAINED_FINESSE: usize = 30;
const QUICK_INNOVATION: usize = 32;
const DARING_TOUCH: usize = 33;
const IMMACULATE_MEND: usize = 34;

const ACTIONS: [Action; ACTION_COUNT] = [
    Action::BasicSynthesis,
    Action::BasicTouch,
    Action::MasterMend,
    Action::HastyTouch,
    Action::RapidSynthesis,
    Action::Observe,
    Action::TricksOfTheTrade,
    Action::WasteNot,
    Action::Veneration,
    Action::StandardTouch,
    Action::GreatStrides,
    Action::Innovation,
    Action::FinalAppraisal,
    Action::WasteNot2,
    Action::ByregotsBlessing,
    Action::PreciseTouch,
    Action::MuscleMemory,
    Action::CarefulObservation,
    Action::CarefulSynthesis,
    Action::Manipulation,
    Action::PrudentTouch,
    Action::AdvancedTouch,
    Action::Reflect,
    Action::PreparatoryTouch,
    Action::Groundwork,
    Action::DelicateSynthesis,
    Action::IntensiveSynthesis,
    Action::TrainedEye,
    Action::HeartAndSoul,
    Action::PrudentSynthesis,
    Action::TrainedFinesse,
    Action::RefinedTouch,
    Action::QuickInnovation,
    Action::DaringTouch,
    Action::ImmaculateMend,
    Action::TrainedPerfection,
];

const BASE_CP: [u16; ACTION_COUNT] = [
    0, 18, 88, 0, 0, 7, 0, 56, 18, 32, 32, 18, 1, 98, 24, 18, 6, 0, 7, 96, 25, 46, 6, 40, 18, 32,
    6, 250, 0, 18, 32, 24, 0, 0, 112, 0,
];

const BASE_DURABILITY: [u16; ACTION_COUNT] = [
    10, 10, 0, 10, 10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 10, 10, 0, 10, 0, 5, 10, 10, 20, 20, 10, 10,
    0, 0, 5, 0, 10, 0, 10, 0, 0,
];

#[derive(Debug, Clone, Copy)]
pub struct Recommendation {
    pub action: Action,
    pub planned: bool,
    pub failure_closure: bool,
    pub candidate_count: usize,
    pub rollout_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ProbabilityEstimate {
    pub successes: usize,
    pub samples: usize,
}

struct ActionEvaluation {
    action: usize,
    values: Vec<f64>,
    actor_score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ConditionOpportunity {
    tier: u8,
    value: f32,
}

impl ProbabilityEstimate {
    pub fn probability(self) -> f64 {
        self.successes as f64 / self.samples.max(1) as f64
    }
}

pub fn recommend(model: &RecipeModel, state: State, seed: u64) -> Result<Recommendation, String> {
    recommend_with_worker_threads(model, state, seed, default_worker_threads())
}

pub fn recommend_with_worker_threads(
    model: &RecipeModel,
    state: State,
    seed: u64,
    worker_threads: usize,
) -> Result<Recommendation, String> {
    model.validate()?;
    match model.status(state) {
        TerminalStatus::Horizon if failure_closure_legal(model, state) => {
            return Ok(Recommendation {
                action: ACTIONS[BASIC_SYNTHESIS],
                planned: false,
                failure_closure: true,
                candidate_count: 1,
                rollout_count: 0,
            });
        }
        TerminalStatus::Active => {}
        _ => {
            return Err(String::from(
                "Gabriel cannot recommend an action for a game-terminal state",
            ));
        }
    }
    let actor = ActorModel::bundled()?;
    pool(worker_threads)?.install(|| recommend_with_actor(model, state, seed, actor))
}

pub fn estimate_full_quality_probability(
    model: &RecipeModel,
    initial: State,
    samples: usize,
    seed: u64,
) -> Result<ProbabilityEstimate, String> {
    estimate_full_quality_probability_with_worker_threads(
        model,
        initial,
        samples,
        seed,
        default_worker_threads(),
    )
}

pub fn estimate_full_quality_probability_with_worker_threads(
    model: &RecipeModel,
    initial: State,
    samples: usize,
    seed: u64,
    worker_threads: usize,
) -> Result<ProbabilityEstimate, String> {
    model.validate()?;
    if samples == 0 {
        return Err(String::from(
            "Gabriel probability estimate requires at least one sample",
        ));
    }
    let actor = ActorModel::bundled()?;
    let successes = pool(worker_threads)?.install(|| {
        (0..samples)
            .into_par_iter()
            .filter(|sample| {
                let game_seed = derive_seed(seed, *sample as u64);
                let policy_seed = derive_seed(seed ^ 0xA076_1D64_78BD_642F, *sample as u64);
                simulate_policy(model, initial, game_seed, policy_seed, actor).is_ok_and(
                    |final_state| {
                        model.status(final_state) == TerminalStatus::Complete
                            && final_state.simulation.quality >= model.settings.max_quality
                    },
                )
            })
            .count()
    });
    Ok(ProbabilityEstimate { successes, samples })
}

fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(DEFAULT_WORKER_THREADS)
}

fn pool(worker_threads: usize) -> Result<Arc<ThreadPool>, String> {
    if !(1..=MAX_WORKER_THREADS).contains(&worker_threads) {
        return Err(format!(
            "Gabriel worker threads must be within 1..={MAX_WORKER_THREADS}"
        ));
    }

    let slot = WORKER_POOL.get_or_init(|| Mutex::new(None));
    let mut entry = slot
        .lock()
        .map_err(|_| String::from("Gabriel worker pool lock poisoned"))?;
    if let Some(current) = entry.as_ref()
        && current.workers == worker_threads
    {
        return Ok(Arc::clone(&current.pool));
    }

    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(worker_threads)
            .thread_name(|index| format!("gabriel-{index}"))
            .build()
            .map_err(|error| format!("failed to create Gabriel worker pool: {error}"))?,
    );
    *entry = Some(PoolEntry {
        workers: worker_threads,
        pool: Arc::clone(&pool),
    });
    Ok(pool)
}

fn simulate_policy(
    model: &RecipeModel,
    mut state: State,
    game_seed: u64,
    policy_seed: u64,
    actor: &ActorModel,
) -> Result<State, String> {
    let mut stream = CounterStream::new(game_seed);
    while model.status(state) == TerminalStatus::Active {
        let decision_seed = derive_seed(policy_seed, u64::from(state.decisions));
        let recommendation = recommend_with_actor(model, state, decision_seed, actor)?;
        state = model
            .apply_counter(state, recommendation.action, &mut stream)
            .map_err(|error| format!("Gabriel policy produced an unusable action: {error:?}"))?;
    }
    Ok(state)
}

fn recommend_with_actor(
    model: &RecipeModel,
    state: State,
    seed: u64,
    actor: &ActorModel,
) -> Result<Recommendation, String> {
    if let Some(action) = immediate_progress_finish(model, state) {
        return Ok(Recommendation {
            action: ACTIONS[action],
            planned: true,
            failure_closure: false,
            candidate_count: 1,
            rollout_count: 0,
        });
    }
    if !(0..ACTION_COUNT).any(|action| planner_legal(model, state, action)) {
        if failure_closure_legal(model, state) {
            return Ok(Recommendation {
                action: ACTIONS[BASIC_SYNTHESIS],
                planned: false,
                failure_closure: true,
                candidate_count: 1,
                rollout_count: 0,
            });
        }
        return Err(String::from("Gabriel found no game-legal action"));
    }
    if state.simulation.quality >= model.required_quality {
        let Some(action) = progress_greedy(model, state) else {
            if failure_closure_legal(model, state) {
                return Ok(Recommendation {
                    action: ACTIONS[BASIC_SYNTHESIS],
                    planned: false,
                    failure_closure: true,
                    candidate_count: 1,
                    rollout_count: 0,
                });
            }
            return Err(String::from(
                "Gabriel found no legal full-quality progress action",
            ));
        };
        return Ok(Recommendation {
            action: ACTIONS[action],
            planned: false,
            failure_closure: false,
            candidate_count: 1,
            rollout_count: 0,
        });
    }
    let (action, candidate_count, rollout_count) =
        match policy_improvement_action(model, state, seed, actor) {
            Ok(result) => result,
            Err(_) if failure_closure_legal(model, state) => {
                return Ok(Recommendation {
                    action: ACTIONS[BASIC_SYNTHESIS],
                    planned: false,
                    failure_closure: true,
                    candidate_count: 1,
                    rollout_count: 0,
                });
            }
            Err(error) => return Err(error),
        };
    Ok(Recommendation {
        action: ACTIONS[action],
        planned: true,
        failure_closure: false,
        candidate_count,
        rollout_count,
    })
}

fn failure_closure_legal(model: &RecipeModel, state: State) -> bool {
    required_action_sequence_legal(model, state, BASIC_SYNTHESIS)
        && state
            .simulation
            .use_action_with_outcome(
                ACTIONS[BASIC_SYNTHESIS],
                state.condition,
                &model.settings,
                true,
            )
            .is_ok()
}

fn policy_improvement_action(
    model: &RecipeModel,
    state: State,
    seed: u64,
    actor: &ActorModel,
) -> Result<(usize, usize, usize), String> {
    let (candidates, actor_scores) = candidate_actions(model, state, actor);
    if candidates.is_empty() {
        return Err(String::from(
            "Gabriel policy improvement found no legal candidate",
        ));
    }
    let incumbent = shielded_actor_action_finish(model, state, actor)
        .ok_or_else(|| String::from("Gabriel actor found no legal incumbent action"))?;
    let condition_prior = condition_prior_action(model, state, &candidates, &actor_scores);
    let candidate_count = candidates.len();
    let mut evaluations = candidates
        .par_iter()
        .map(|action| ActionEvaluation {
            action: *action,
            values: policy_rollout_values(
                model,
                state,
                *action,
                seed,
                actor,
                0..POLICY_PILOT_ROLLOUTS,
            ),
            actor_score: actor_scores[*action],
        })
        .collect::<Vec<_>>();
    let mut rollout_count = candidate_count * POLICY_PILOT_ROLLOUTS;
    evaluations.par_iter_mut().for_each(|evaluation| {
        evaluation.values.extend(policy_rollout_values(
            model,
            state,
            evaluation.action,
            seed,
            actor,
            POLICY_PILOT_ROLLOUTS..POLICY_SCREEN_ROLLOUTS,
        ));
    });
    rollout_count += candidate_count * (POLICY_SCREEN_ROLLOUTS - POLICY_PILOT_ROLLOUTS);
    evaluations.sort_unstable_by(|left, right| {
        let left_score = mean(&left.values) + 1e-7 * f64::from(left.actor_score);
        let right_score = mean(&right.values) + 1e-7 * f64::from(right.actor_score);
        right_score
            .total_cmp(&left_score)
            .then_with(|| left.action.cmp(&right.action))
    });
    let finalist_actions = select_finalist_actions(&evaluations, incumbent, condition_prior);
    let mut finalists = evaluations
        .into_iter()
        .filter(|evaluation| finalist_actions.contains(&evaluation.action))
        .collect::<Vec<_>>();
    finalists.par_iter_mut().for_each(|evaluation| {
        evaluation.values.extend(policy_rollout_values(
            model,
            state,
            evaluation.action,
            seed,
            actor,
            POLICY_SCREEN_ROLLOUTS..POLICY_FINAL_ROLLOUTS,
        ));
    });
    rollout_count += finalists.len() * (POLICY_FINAL_ROLLOUTS - POLICY_SCREEN_ROLLOUTS);
    let selected = best_action(&finalists).unwrap_or(incumbent);
    let incumbent_evaluation = finalists
        .iter()
        .find(|evaluation| evaluation.action == incumbent)
        .expect("actor incumbent must remain a finalist");
    let selected_evaluation = finalists
        .iter()
        .find(|evaluation| evaluation.action == selected)
        .expect("selected action must be a finalist");
    let action =
        if selected == incumbent || paired_improvement(selected_evaluation, incumbent_evaluation) {
            selected
        } else {
            incumbent
        };
    Ok((action, candidate_count, rollout_count))
}

fn policy_rollout_values(
    model: &RecipeModel,
    root: State,
    first_action: usize,
    seed: u64,
    actor: &ActorModel,
    rollouts: std::ops::Range<usize>,
) -> Vec<f64> {
    rollouts
        .map(|rollout| {
            rollout_utility(
                model,
                root,
                first_action,
                derive_seed(seed, rollout as u64),
                actor,
            )
        })
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn select_finalist_actions(
    ranked: &[ActionEvaluation],
    incumbent: usize,
    condition_prior: Option<usize>,
) -> Vec<usize> {
    let mut finalists = Vec::with_capacity(POLICY_FINALISTS + 1);
    finalists.push(incumbent);
    for evaluation in ranked {
        if !finalists.contains(&evaluation.action) {
            finalists.push(evaluation.action);
            if finalists.len() == POLICY_FINALISTS {
                break;
            }
        }
    }
    if let Some(action) = condition_prior
        && !finalists.contains(&action)
    {
        finalists.push(action);
    }
    finalists
}

fn completion_count(values: &[f64]) -> usize {
    values.iter().filter(|value| **value >= 1_000.0).count()
}

fn best_action(evaluations: &[ActionEvaluation]) -> Option<usize> {
    let mut best = None;
    let mut best_successes = 0;
    let mut best_mean = f64::NEG_INFINITY;
    let mut best_actor_score = f32::NEG_INFINITY;
    for evaluation in evaluations {
        let successes = completion_count(&evaluation.values);
        let candidate_mean = mean(&evaluation.values);
        if best.is_none()
            || successes > best_successes
            || successes == best_successes
                && (candidate_mean.total_cmp(&best_mean).is_gt()
                    || candidate_mean.total_cmp(&best_mean).is_eq()
                        && evaluation.actor_score.total_cmp(&best_actor_score).is_gt())
        {
            best = Some(evaluation.action);
            best_successes = successes;
            best_mean = candidate_mean;
            best_actor_score = evaluation.actor_score;
        }
    }
    best
}

fn paired_improvement(challenger: &ActionEvaluation, incumbent: &ActionEvaluation) -> bool {
    debug_assert_eq!(challenger.values.len(), incumbent.values.len());
    let mut gains = 0;
    let mut losses = 0;
    let differences = challenger
        .values
        .iter()
        .zip(&incumbent.values)
        .map(|(candidate, baseline)| {
            let candidate_success = *candidate >= 1_000.0;
            let baseline_success = *baseline >= 1_000.0;
            gains += usize::from(candidate_success && !baseline_success);
            losses += usize::from(!candidate_success && baseline_success);
            candidate - baseline
        })
        .collect::<Vec<_>>();
    if gains != losses {
        return gains > losses;
    }
    let paired_mean = mean(&differences);
    if paired_mean <= 0.0 {
        return false;
    }
    let variance = differences
        .iter()
        .map(|difference| (difference - paired_mean).powi(2))
        .sum::<f64>()
        / differences.len().saturating_sub(1).max(1) as f64;
    let standard_error = (variance / differences.len().max(1) as f64).sqrt();
    paired_mean > 1.96 * standard_error
}

fn rollout_utility(
    model: &RecipeModel,
    mut state: State,
    first_action: usize,
    seed: u64,
    actor: &ActorModel,
) -> f64 {
    let mut stream = CounterStream::new(seed);
    let Ok(next) = model.apply_counter(state, ACTIONS[first_action], &mut stream) else {
        return -1_000.0;
    };
    state = next;
    while model.status(state) == TerminalStatus::Active {
        let action = if state.simulation.quality >= model.required_quality {
            progress_greedy(model, state)
        } else {
            shielded_actor_action_finish(model, state, actor)
        };
        let Some(action) = action else {
            break;
        };
        let Ok(next) = model.apply_counter(state, ACTIONS[action], &mut stream) else {
            break;
        };
        state = next;
    }
    terminal_utility(model, state)
}

fn terminal_utility(model: &RecipeModel, state: State) -> f64 {
    let quality =
        (f64::from(state.simulation.quality) / f64::from(model.required_quality)).min(1.0);
    let progress =
        (f64::from(state.simulation.progress) / f64::from(model.settings.max_progress)).min(1.0);
    let success = f64::from(model.status(state) == TerminalStatus::Complete);
    1_000.0 * success + 30.0 * quality.min(progress).powi(10) + 10.0 * (quality * progress).powi(5)
}

fn progress_greedy(model: &RecipeModel, state: State) -> Option<usize> {
    if let Some(action) = immediate_progress_finish(model, state) {
        return Some(action);
    }
    if state.simulation.effects.heart_and_soul_active() {
        return [INTENSIVE_SYNTHESIS, TRICKS_OF_THE_TRADE, PRECISE_TOUCH]
            .into_iter()
            .find(|&action| planner_legal(model, state, action));
    }
    if state.simulation.effects.combo() == Combo::SynthesisBegin {
        return [MUSCLE_MEMORY, REFLECT]
            .into_iter()
            .find(|&action| planner_legal(model, state, action));
    }
    let remaining = model
        .settings
        .max_progress
        .saturating_sub(state.simulation.progress);
    if state.condition == Condition::Good
        && legal_with_durability(model, state, INTENSIVE_SYNTHESIS)
    {
        return Some(INTENSIVE_SYNTHESIS);
    }
    if state.condition == Condition::Malleable {
        if legal_with_durability(model, state, GROUNDWORK) {
            return Some(GROUNDWORK);
        }
        if legal_with_durability(model, state, CAREFUL_SYNTHESIS) {
            return Some(CAREFUL_SYNTHESIS);
        }
    }
    if state.condition == Condition::Centered
        && legal_with_durability(model, state, RAPID_SYNTHESIS)
    {
        return Some(RAPID_SYNTHESIS);
    }
    if state.condition == Condition::Pliant {
        if state.simulation.durability <= 20 && planner_legal(model, state, IMMACULATE_MEND) {
            return Some(IMMACULATE_MEND);
        }
        if state.simulation.effects.manipulation() == 0
            && state.simulation.durability <= 35
            && planner_legal(model, state, MANIPULATION)
        {
            return Some(MANIPULATION);
        }
        if state.simulation.durability <= 25 && planner_legal(model, state, MASTERS_MEND) {
            return Some(MASTERS_MEND);
        }
        if veneration_worthwhile(model, state, remaining) {
            return Some(VENERATION);
        }
    }
    if state.condition == Condition::Primed {
        if state.simulation.effects.manipulation() == 0
            && state.simulation.durability <= 25
            && planner_legal(model, state, MANIPULATION)
        {
            return Some(MANIPULATION);
        }
        if veneration_worthwhile(model, state, remaining) {
            return Some(VENERATION);
        }
    }
    if state.simulation.durability <= 10 {
        if state.simulation.effects.manipulation() > 0 && planner_legal(model, state, OBSERVE) {
            return Some(OBSERVE);
        }
        for action in [IMMACULATE_MEND, MASTERS_MEND, MANIPULATION] {
            if planner_legal(model, state, action) {
                return Some(action);
            }
        }
    }
    if veneration_worthwhile(model, state, remaining) {
        return Some(VENERATION);
    }
    for action in [CAREFUL_SYNTHESIS, PRUDENT_SYNTHESIS] {
        if legal_with_durability(model, state, action) {
            return Some(action);
        }
    }
    [BASIC_SYNTHESIS, RAPID_SYNTHESIS, OBSERVE]
        .into_iter()
        .find(|&action| planner_legal(model, state, action))
}

fn veneration_worthwhile(model: &RecipeModel, state: State, remaining: u16) -> bool {
    if state.simulation.effects.veneration() > 0 || !planner_legal(model, state, VENERATION) {
        return false;
    }
    let best_immediate_gain = (0..ACTION_COUNT)
        .filter(|action| is_progress(*action) && legal_with_durability(model, state, *action))
        .map(|action| progress_gain(model, state, action))
        .max()
        .unwrap_or(0);
    best_immediate_gain > 0 && remaining > best_immediate_gain
}

fn candidate_actions(
    model: &RecipeModel,
    state: State,
    actor: &ActorModel,
) -> (Vec<usize>, [f32; ACTION_COUNT]) {
    let scratch = scratch(model, state);
    let logits = actor.logits(&features(model, state));
    let candidates = (0..ACTION_COUNT)
        .filter(|action| scratch.legal[*action])
        .collect();
    (candidates, logits)
}

fn condition_prior_action(
    model: &RecipeModel,
    state: State,
    candidates: &[usize],
    actor_scores: &[f32; ACTION_COUNT],
) -> Option<usize> {
    let mut best: Option<(usize, ConditionOpportunity)> = None;
    for &action in candidates {
        let Some(opportunity) = condition_opportunity(model, state, action) else {
            continue;
        };
        let replace = best.is_none_or(|(best_action, best_opportunity)| {
            opportunity.tier > best_opportunity.tier
                || opportunity.tier == best_opportunity.tier
                    && (opportunity.value.total_cmp(&best_opportunity.value).is_gt()
                        || opportunity.value.total_cmp(&best_opportunity.value).is_eq()
                            && (actor_scores[action]
                                .total_cmp(&actor_scores[best_action])
                                .is_gt()
                                || actor_scores[action]
                                    .total_cmp(&actor_scores[best_action])
                                    .is_eq()
                                    && action < best_action))
        });
        if replace {
            best = Some((action, opportunity));
        }
    }
    best.map(|(action, _)| action)
}

fn condition_opportunity(
    model: &RecipeModel,
    state: State,
    action: usize,
) -> Option<ConditionOpportunity> {
    let normal = State {
        condition: Condition::Normal,
        ..state
    };
    match state.condition {
        Condition::Good | Condition::Excellent => {
            let current_next = model
                .apply_outcome(state, ACTIONS[action], true, 0.0)
                .ok()?;
            let normal_next = model.apply_outcome(normal, ACTIONS[action], true, 0.0).ok();
            let current_value = immediate_opportunity_value(model, state, action);
            let normal_value = immediate_opportunity_value(model, normal, action);
            let preserves_heart_and_soul = normal_next.is_some_and(|next| {
                state.simulation.effects.heart_and_soul_active()
                    && current_next.simulation.effects.heart_and_soul_active()
                    && !next.simulation.effects.heart_and_soul_active()
            });
            if (normal_next.is_none() || preserves_heart_and_soul) && current_value > 0.0 {
                return Some(ConditionOpportunity {
                    tier: 2,
                    value: current_value,
                });
            }
            positive_opportunity(1, current_value - normal_value)
        }
        Condition::Centered | Condition::Malleable => positive_opportunity(
            1,
            expected_productive_value(model, state, action)
                - expected_productive_value(model, normal, action),
        ),
        Condition::Sturdy | Condition::Robust => {
            if !has_useful_success_transition(model, state, action) {
                return None;
            }
            positive_opportunity(
                1,
                f32::from(
                    durability_cost(normal, action).saturating_sub(durability_cost(state, action)),
                ) / f32::from(model.settings.max_durability.max(1)),
            )
        }
        Condition::Pliant => {
            if !has_useful_success_transition(model, state, action) {
                return None;
            }
            positive_opportunity(
                1,
                f32::from(cp_cost(normal, action).saturating_sub(cp_cost(state, action)))
                    / f32::from(model.settings.max_cp.max(1)),
            )
        }
        Condition::Primed => {
            positive_opportunity(1, primed_duration_opportunity(model, state, normal, action))
        }
        Condition::GoodOmen => {
            positive_opportunity(1, good_omen_followup_opportunity(model, state, action))
        }
        Condition::Normal | Condition::Poor => None,
    }
}

fn positive_opportunity(tier: u8, value: f32) -> Option<ConditionOpportunity> {
    (value > f32::EPSILON).then_some(ConditionOpportunity { tier, value })
}

fn expected_productive_value(model: &RecipeModel, state: State, action: usize) -> f32 {
    let success =
        f32::from(ACTIONS[action].success_rate(&state.simulation, state.condition)) / 100.0;
    let remaining_progress = model
        .settings
        .max_progress
        .saturating_sub(state.simulation.progress);
    let remaining_quality = model
        .required_quality
        .saturating_sub(state.simulation.quality);
    let progress = progress_gain(model, state, action).min(remaining_progress);
    let quality = quality_gain(model, state, action).min(remaining_quality);
    success
        * (f32::from(progress) / f32::from(model.settings.max_progress.max(1))
            + f32::from(quality) / f32::from(model.required_quality.max(1)))
}

fn immediate_opportunity_value(model: &RecipeModel, state: State, action: usize) -> f32 {
    let success =
        f32::from(ACTIONS[action].success_rate(&state.simulation, state.condition)) / 100.0;
    let Some(next) = model.apply_outcome(state, ACTIONS[action], true, 0.0).ok() else {
        return 0.0;
    };
    expected_productive_value(model, state, action)
        + success
            * (f32::from(next.simulation.cp.saturating_sub(state.simulation.cp))
                / f32::from(model.settings.max_cp.max(1))
                + f32::from(
                    next.simulation
                        .durability
                        .saturating_sub(state.simulation.durability),
                ) / f32::from(model.settings.max_durability.max(1)))
}

fn has_useful_success_transition(model: &RecipeModel, state: State, action: usize) -> bool {
    let Ok(next) = model.apply_outcome(state, ACTIONS[action], true, 0.0) else {
        return false;
    };
    let before = state.simulation;
    let after = next.simulation;
    let quality_needed = before.quality < model.required_quality;
    let progress_needed = before.progress < model.settings.max_progress;
    after.progress > before.progress
        || after.quality > before.quality
        || after.cp > before.cp
        || after.durability > before.durability
        || quality_needed
            && (after.effects.inner_quiet() > before.effects.inner_quiet()
                || after.effects.innovation() > before.effects.innovation()
                || after.effects.great_strides() > before.effects.great_strides()
                || after.effects.expedience() && !before.effects.expedience())
        || progress_needed
            && (after.effects.veneration() > before.effects.veneration()
                || after.effects.muscle_memory() > before.effects.muscle_memory())
        || (quality_needed || progress_needed)
            && (after.effects.waste_not() > before.effects.waste_not()
                || after.effects.manipulation() > before.effects.manipulation()
                || after.effects.final_appraisal() > before.effects.final_appraisal()
                || after.effects.trained_perfection_active()
                    && !before.effects.trained_perfection_active()
                || after.effects.heart_and_soul_active() && !before.effects.heart_and_soul_active())
}

fn primed_duration_opportunity(
    model: &RecipeModel,
    state: State,
    normal: State,
    action: usize,
) -> f32 {
    let Ok(primed_next) = model.apply_outcome(state, ACTIONS[action], true, 0.0) else {
        return 0.0;
    };
    let Ok(normal_next) = model.apply_outcome(normal, ACTIONS[action], true, 0.0) else {
        return 0.0;
    };
    let primed = primed_next.simulation.effects;
    let ordinary = normal_next.simulation.effects;
    let before = state.simulation.effects;
    let quality_needed = state.simulation.quality < model.required_quality;
    let progress_needed = state.simulation.progress < model.settings.max_progress;
    let useful_added_turns = |before: u8, ordinary: u8, primed: u8| {
        u16::from(ordinary > before) * u16::from(primed.saturating_sub(ordinary))
    };
    let mut added_turns = 0u16;
    if quality_needed || progress_needed {
        added_turns +=
            useful_added_turns(before.waste_not(), ordinary.waste_not(), primed.waste_not());
        added_turns += useful_added_turns(
            before.manipulation(),
            ordinary.manipulation(),
            primed.manipulation(),
        );
        added_turns += useful_added_turns(
            before.final_appraisal(),
            ordinary.final_appraisal(),
            primed.final_appraisal(),
        );
    }
    if quality_needed {
        added_turns += useful_added_turns(
            before.innovation(),
            ordinary.innovation(),
            primed.innovation(),
        );
        added_turns += useful_added_turns(
            before.great_strides(),
            ordinary.great_strides(),
            primed.great_strides(),
        );
    }
    if progress_needed {
        added_turns += useful_added_turns(
            before.veneration(),
            ordinary.veneration(),
            primed.veneration(),
        );
        added_turns += useful_added_turns(
            before.muscle_memory(),
            ordinary.muscle_memory(),
            primed.muscle_memory(),
        );
    }
    f32::from(added_turns) / 2.0
}

fn good_omen_followup_opportunity(model: &RecipeModel, state: State, action: usize) -> f32 {
    let success =
        f32::from(ACTIONS[action].success_rate(&state.simulation, state.condition)) / 100.0;
    let successful = model
        .apply_outcome(state, ACTIONS[action], true, 0.0)
        .ok()
        .map_or(0.0, |next| best_good_followup_opportunity(model, next));
    let failed = if success < 1.0 {
        model
            .apply_outcome(state, ACTIONS[action], false, 0.0)
            .ok()
            .map_or(0.0, |next| best_good_followup_opportunity(model, next))
    } else {
        0.0
    };
    success * successful + (1.0 - success) * failed
}

fn best_good_followup_opportunity(model: &RecipeModel, state: State) -> f32 {
    if state.condition != Condition::Good || model.status(state) != TerminalStatus::Active {
        return 0.0;
    }
    (0..ACTION_COUNT)
        .filter(|action| compute_planner_legal(model, state, *action))
        .map(|action| immediate_opportunity_value(model, state, action))
        .max_by(f32::total_cmp)
        .unwrap_or(0.0)
}

fn shielded_actor_action_finish(
    model: &RecipeModel,
    state: State,
    actor: &ActorModel,
) -> Option<usize> {
    if let Some(action) = immediate_progress_finish(model, state) {
        return Some(action);
    }
    let action = actor_safety_override(
        model,
        state,
        byregot_override(model, state, raw_actor_action(model, state, actor)?),
    );
    Some(protect_subquality_completion(model, state, action))
}

fn actor_safety_override(model: &RecipeModel, state: State, action: usize) -> usize {
    let remaining_progress = model
        .settings
        .max_progress
        .saturating_sub(state.simulation.progress);
    let remaining_quality = model
        .required_quality
        .saturating_sub(state.simulation.quality);
    let progress_endgame_span = unbuffed_progress_gain(model, state)
        .saturating_add(unbuffed_basic_synthesis_gain(model, state));
    if is_progress(action)
        && remaining_progress <= progress_endgame_span
        && remaining_quality > prepared_quality_finisher_gain(model, state)
    {
        for replacement in [
            DARING_TOUCH,
            PRECISE_TOUCH,
            TRAINED_FINESSE,
            HASTY_TOUCH,
            OBSERVE,
        ] {
            if matches!(replacement, DARING_TOUCH | PRECISE_TOUCH | HASTY_TOUCH) {
                let hasty_is_worthwhile = replacement != HASTY_TOUCH
                    || matches!(
                        state.condition,
                        Condition::Centered | Condition::Sturdy | Condition::Robust
                    )
                    || state.simulation.effects.waste_not() > 0;
                if hasty_is_worthwhile && legal_with_durability(model, state, replacement) {
                    return replacement;
                }
            } else if planner_legal(model, state, replacement) {
                return replacement;
            }
        }
    }
    let quality_completion_cp_reserve = cp_cost(state, MASTERS_MEND)
        .saturating_add(cp_cost(state, GREAT_STRIDES))
        .saturating_add(cp_cost(state, INNOVATION))
        .saturating_add(cp_cost(state, BYREGOT));
    if action == TRAINED_FINESSE && state.simulation.cp < quality_completion_cp_reserve {
        for replacement in [DARING_TOUCH, HASTY_TOUCH] {
            let hasty_is_worthwhile = replacement != HASTY_TOUCH
                || matches!(
                    state.condition,
                    Condition::Centered | Condition::Sturdy | Condition::Robust
                )
                || state.simulation.effects.waste_not() > 0
                || state.simulation.durability
                    >= durability_cost(state, HASTY_TOUCH).saturating_mul(3);
            if hasty_is_worthwhile && legal_with_durability(model, state, replacement) {
                return replacement;
            }
        }
    }
    action
}

fn raw_actor_action(model: &RecipeModel, state: State, actor: &ActorModel) -> Option<usize> {
    let scratch = scratch(model, state);
    let logits = actor.logits(&features(model, state));
    first_max_action(
        (0..ACTION_COUNT).filter(|action| scratch.legal[*action]),
        |action| logits[action],
    )
}

fn first_max_action<I, F>(mut actions: I, score: F) -> Option<usize>
where
    I: Iterator<Item = usize>,
    F: Fn(usize) -> f32,
{
    let mut best = actions.next()?;
    let mut best_score = score(best);
    for action in actions {
        let action_score = score(action);
        if action_score.total_cmp(&best_score).is_gt() {
            best = action;
            best_score = action_score;
        }
    }
    Some(best)
}

fn byregot_override(model: &RecipeModel, state: State, base_action: usize) -> usize {
    if state.simulation.effects.inner_quiet() != 10 {
        return base_action;
    }
    if planner_legal(model, state, BYREGOT)
        && state
            .simulation
            .quality
            .saturating_add(quality_gain(model, state, BYREGOT))
            >= model.required_quality
    {
        return BYREGOT;
    }
    let mut configured = state;
    configured.simulation.effects = configured
        .simulation
        .effects
        .with_great_strides(3)
        .with_innovation(4);
    configured.condition = Condition::Good;
    let good_gain = quality_gain(model, configured, BYREGOT);
    configured.condition = Condition::Normal;
    let normal_gain = quality_gain(model, configured, BYREGOT);
    if state.simulation.quality.saturating_add(good_gain) < model.required_quality {
        return base_action;
    }
    let mut required_cp = 24;
    if state.simulation.effects.great_strides() == 0 {
        required_cp += 32;
    }
    if state.simulation.effects.innovation() == 0 {
        required_cp += 18;
    }
    if state.simulation.cp < required_cp {
        return base_action;
    }
    if state.simulation.effects.great_strides() == 0 && planner_legal(model, state, GREAT_STRIDES) {
        return GREAT_STRIDES;
    }
    if state.simulation.effects.great_strides() > 0 && state.simulation.effects.innovation() == 0 {
        if planner_legal(model, state, INNOVATION) {
            return INNOVATION;
        }
    }
    if state.simulation.effects.great_strides() > 0 && state.simulation.effects.innovation() > 0 {
        if state.condition == Condition::Good && planner_legal(model, state, BYREGOT) {
            return BYREGOT;
        }
        if state.simulation.quality.saturating_add(normal_gain) >= model.required_quality
            && planner_legal(model, state, BYREGOT)
        {
            return BYREGOT;
        }
        if state.simulation.effects.great_strides() > 1
            && state.simulation.cp >= 31
            && planner_legal(model, state, OBSERVE)
        {
            return OBSERVE;
        }
        if planner_legal(model, state, BYREGOT) {
            return BYREGOT;
        }
    }
    base_action
}

fn protect_subquality_completion(model: &RecipeModel, state: State, action: usize) -> usize {
    if !is_progress(action) || state.simulation.quality >= model.required_quality {
        return action;
    }
    let Ok(next) = model.apply_outcome(state, ACTIONS[action], true, 0.0) else {
        return action;
    };
    if next.simulation.progress < model.settings.max_progress {
        return action;
    }
    for replacement in [
        DARING_TOUCH,
        PRECISE_TOUCH,
        HASTY_TOUCH,
        TRAINED_FINESSE,
        OBSERVE,
    ] {
        if matches!(replacement, DARING_TOUCH | PRECISE_TOUCH | HASTY_TOUCH) {
            if legal_with_durability(model, state, replacement) {
                return replacement;
            }
        } else if planner_legal(model, state, replacement) {
            return replacement;
        }
    }
    action
}

fn immediate_progress_finish(model: &RecipeModel, state: State) -> Option<usize> {
    if state.simulation.quality < model.required_quality {
        return None;
    }
    (0..ACTION_COUNT)
        .filter(|action| {
            is_progress(*action)
                && planner_legal(model, state, *action)
                && ACTIONS[*action].success_rate(&state.simulation, state.condition) == 100
                && state
                    .simulation
                    .progress
                    .saturating_add(progress_gain(model, state, *action))
                    >= model.settings.max_progress
        })
        .min_by_key(|action| {
            (
                cp_cost(state, *action),
                durability_cost(state, *action),
                u16::MAX - progress_gain(model, state, *action),
            )
        })
}

fn planner_legal(model: &RecipeModel, state: State, action: usize) -> bool {
    scratch(model, state).legal[action]
}

fn compute_planner_legal(model: &RecipeModel, state: State, action: usize) -> bool {
    // The actor checkpoint retains its original 36 output slots, but Gabriel must never
    // search or emit actions forbidden by its policy.
    if matches!(
        action,
        FINAL_APPRAISAL | TRAINED_EYE | CAREFUL_OBSERVATION | QUICK_INNOVATION
    ) || model.status(state) != TerminalStatus::Active
    {
        return false;
    }
    if !required_action_sequence_legal(model, state, action) {
        return false;
    }
    let Ok(next) = model.apply_outcome(state, ACTIONS[action], true, 0.0) else {
        return false;
    };
    if next.simulation.progress >= model.settings.max_progress
        && next.simulation.quality < model.required_quality
    {
        return false;
    }
    let durability = durability_cost(state, action);
    if durability >= state.simulation.durability {
        return is_progress(action) && next.simulation.progress >= model.settings.max_progress;
    }
    true
}

fn legal_with_durability(model: &RecipeModel, state: State, action: usize) -> bool {
    planner_legal(model, state, action)
        && durability_cost(state, action) < state.simulation.durability
}

fn progress_gain(model: &RecipeModel, state: State, action: usize) -> u16 {
    state
        .simulation
        .use_action_with_outcome(ACTIONS[action], state.condition, &model.settings, true)
        .map_or(0, |next| {
            next.progress.saturating_sub(state.simulation.progress)
        })
}

fn quality_gain(model: &RecipeModel, state: State, action: usize) -> u16 {
    state
        .simulation
        .use_action_with_outcome(ACTIONS[action], state.condition, &model.settings, true)
        .map_or(0, |next| {
            next.quality.saturating_sub(state.simulation.quality)
        })
}

fn cp_cost(state: State, action: usize) -> u16 {
    let mut cost = BASE_CP[action];
    if (action == STANDARD_TOUCH && state.simulation.effects.combo() == Combo::BasicTouch)
        || (action == ADVANCED_TOUCH
            && matches!(state.simulation.effects.combo(), Combo::StandardTouch))
    {
        cost = 18;
    }
    if state.condition == Condition::Pliant {
        cost.div_ceil(2)
    } else {
        cost
    }
}

fn durability_cost(state: State, action: usize) -> u16 {
    if state.simulation.effects.trained_perfection_active() {
        return 0;
    }
    let mut divisor = 1;
    if state.simulation.effects.waste_not() > 0 {
        divisor *= 2;
    }
    if matches!(state.condition, Condition::Sturdy | Condition::Robust) {
        divisor *= 2;
    }
    BASE_DURABILITY[action].div_ceil(divisor)
}

fn is_progress(action: usize) -> bool {
    matches!(
        action,
        BASIC_SYNTHESIS
            | RAPID_SYNTHESIS
            | MUSCLE_MEMORY
            | CAREFUL_SYNTHESIS
            | GROUNDWORK
            | DELICATE_SYNTHESIS
            | INTENSIVE_SYNTHESIS
            | PRUDENT_SYNTHESIS
    )
}

fn is_heart_and_soul_followup(action: usize) -> bool {
    matches!(
        action,
        TRICKS_OF_THE_TRADE | INTENSIVE_SYNTHESIS | PRECISE_TOUCH
    )
}

fn required_action_sequence_legal(model: &RecipeModel, state: State, action: usize) -> bool {
    if state.simulation.effects.combo() == Combo::SynthesisBegin
        && !matches!(action, MUSCLE_MEMORY | REFLECT)
    {
        return false;
    }
    if state.simulation.effects.heart_and_soul_active() && !is_heart_and_soul_followup(action) {
        return false;
    }
    action != HEART_AND_SOUL
        || state.condition == Condition::Normal
            && state.decisions.saturating_add(1) < model.max_decisions
}

fn scratch(model: &RecipeModel, state: State) -> StateScratch {
    STATE_SCRATCH.with(|slot| {
        if let Some(cached) = *slot.borrow()
            && cached.state == state
        {
            return cached;
        }
        let computed = StateScratch {
            state,
            unbuffed_progress: compute_unbuffed_progress_gain(model, state),
            unbuffed_basic: compute_unbuffed_basic_synthesis_gain(model, state),
            quality_finisher: compute_prepared_quality_finisher_gain(model, state),
            legal: std::array::from_fn(|action| compute_planner_legal(model, state, action)),
        };
        *slot.borrow_mut() = Some(computed);
        computed
    })
}

fn unbuffed_progress_gain(model: &RecipeModel, state: State) -> u16 {
    scratch(model, state).unbuffed_progress
}

fn compute_unbuffed_progress_gain(model: &RecipeModel, state: State) -> u16 {
    let mut configured = state;
    configured.simulation.cp = model.settings.max_cp;
    configured.simulation.durability = model.settings.max_durability;
    configured.simulation.effects = configured
        .simulation
        .effects
        .with_veneration(0)
        .with_muscle_memory(0);
    configured.condition = Condition::Normal;
    (0..ACTION_COUNT)
        .filter(|action| is_progress(*action))
        .map(|action| progress_gain(model, configured, action))
        .max()
        .unwrap_or(0)
}

fn unbuffed_basic_synthesis_gain(model: &RecipeModel, state: State) -> u16 {
    scratch(model, state).unbuffed_basic
}

fn compute_unbuffed_basic_synthesis_gain(model: &RecipeModel, state: State) -> u16 {
    let mut configured = state;
    configured.simulation.cp = model.settings.max_cp;
    configured.simulation.durability = model.settings.max_durability;
    configured.simulation.effects = configured
        .simulation
        .effects
        .with_veneration(0)
        .with_muscle_memory(0);
    configured.condition = Condition::Normal;
    progress_gain(model, configured, BASIC_SYNTHESIS)
}

fn prepared_quality_finisher_gain(model: &RecipeModel, state: State) -> u16 {
    scratch(model, state).quality_finisher
}

fn compute_prepared_quality_finisher_gain(model: &RecipeModel, state: State) -> u16 {
    let mut configured = state;
    configured.simulation.cp = model.settings.max_cp;
    configured.simulation.durability = model.settings.max_durability;
    configured.simulation.effects = configured
        .simulation
        .effects
        .with_inner_quiet(10)
        .with_great_strides(3)
        .with_innovation(4);
    configured.condition = Condition::Normal;
    (0..ACTION_COUNT)
        .map(|action| quality_gain(model, configured, action))
        .max()
        .unwrap_or(0)
}

fn features(model: &RecipeModel, state: State) -> [f32; FEATURE_COUNT] {
    let simulation = state.simulation;
    let effects = simulation.effects;
    let mut values = [0.0; FEATURE_COUNT];
    values[0] = f32::from(simulation.cp) / f32::from(model.settings.max_cp);
    values[1] = f32::from(simulation.durability) / f32::from(model.settings.max_durability);
    values[2] = f32::from(simulation.progress) / f32::from(model.settings.max_progress);
    values[3] = f32::from(simulation.quality) / f32::from(model.required_quality);
    values[4] = f32::from(
        model
            .settings
            .max_progress
            .saturating_sub(simulation.progress),
    ) / f32::from(model.settings.max_progress);
    values[5] = f32::from(model.required_quality.saturating_sub(simulation.quality))
        / f32::from(model.required_quality);
    values[6] = f32::from(effects.inner_quiet()) / 10.0;
    values[7] = f32::from(effects.waste_not()) / 10.0;
    values[8] = f32::from(effects.veneration()) / 6.0;
    values[9] = f32::from(effects.innovation()) / 6.0;
    values[10] = f32::from(effects.great_strides()) / 5.0;
    values[11] = f32::from(effects.muscle_memory()) / 7.0;
    values[12] = f32::from(effects.manipulation()) / 10.0;
    values[13] = f32::from(effects.final_appraisal()) / 7.0;
    values[14] = f32::from(effects.trained_perfection_active());
    values[15] = f32::from(effects.trained_perfection_available());
    values[16] = f32::from(effects.heart_and_soul_active());
    values[17] = f32::from(effects.heart_and_soul_available());
    values[18] = f32::from(effects.quick_innovation_available());
    values[19] = f32::from(effects.careful_observation_charges()) / 3.0;
    values[20] = f32::from(effects.crafter_delineations()) / 2.0;
    values[21] = f32::from(effects.combo() == Combo::BasicTouch);
    values[22] = f32::from(effects.combo() == Combo::StandardTouch);
    values[23] = f32::from(effects.expedience());
    values[24] = (f32::from(state.step) / ACTOR_STEP_SCALE).min(1.0);
    values[25] = (f32::from(state.decisions) / ACTOR_DECISION_SCALE).min(1.0);
    values[26 + condition_index(state.condition)] = 1.0;
    values[37] = f32::from(simulation.durability + 5 * u16::from(effects.manipulation()))
        / f32::from(model.settings.max_durability);
    values[38] = f32::from(effects.inner_quiet() == 10);
    values[39] = f32::from(effects.innovation() > 0);
    values[40] = f32::from(effects.great_strides() > 0);
    values[41] = f32::from(effects.veneration() > 0);
    values[42] = values[3] * values[2];
    values[43] = values[3].min(values[2]);
    values[44] = values[3].max(values[2]);
    values[45] = values[0] * values[3];
    values[46] = values[1] * values[3];
    values[47] = values[0] * values[2];
    values[48] = values[1] * values[2];
    let remaining_progress = model
        .settings
        .max_progress
        .saturating_sub(simulation.progress);
    let remaining_quality = model.required_quality.saturating_sub(simulation.quality);
    let quality_finisher_gain = prepared_quality_finisher_gain(model, state);
    values[49] = f32::from(
        remaining_progress
            <= unbuffed_progress_gain(model, state)
                .saturating_add(unbuffed_basic_synthesis_gain(model, state)),
    );
    values[50] = f32::from(remaining_quality <= quality_finisher_gain.saturating_mul(2));
    values[51] = f32::from(remaining_quality <= quality_finisher_gain);
    values[52] = f32::from(simulation.durability <= 10);
    values[53] = f32::from(simulation.cp <= 100);
    values[54] = f32::from(state.condition == Condition::Robust && effects.innovation() > 0);
    values[55] = f32::from(
        state.condition == Condition::Good
            && effects.great_strides() > 0
            && effects.innovation() > 0,
    );
    values[56] = f32::from(state.condition == Condition::Pliant && effects.manipulation() == 0);
    values[57] = f32::from(state.condition == Condition::Primed && effects.manipulation() == 0);
    values[58] = values[6] * values[3];
    values[59] = values[8] * values[4];
    values[60] = values[9] * values[5];
    values[61] = values[10] * values[5];
    values[62] = 1.0;
    values[63] = (f32::from(state.decisions) - f32::from(state.step)) / 9.0;
    values
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

#[cfg(test)]
mod tests {
    use super::*;
    use raphael_sim::{ActionMask, Effects, Settings, SimulationState};

    fn fixture() -> (RecipeModel, State) {
        let settings = Settings {
            max_cp: 700,
            max_durability: 60,
            max_progress: 11_250,
            max_quality: 31_520,
            base_progress: 323,
            base_quality: 322,
            job_level: 100,
            allowed_actions: ActionMask::all()
                .remove(Action::TrainedEye)
                .remove(Action::StellarSteadyHand),
            adversarial: false,
            backload_progress: false,
            stellar_steady_hand_charges: 0,
        };
        let effects = Effects::initial(&settings)
            .with_careful_observation_charges(3)
            .with_crafter_delineations(2)
            .with_splendor_cosmic(true);
        (
            RecipeModel {
                settings,
                required_quality: 31_520,
                condition_probabilities_bps: [
                    2_000, 1_000, 0, 0, 1_500, 1_000, 1_500, 1_000, 1_000, 0, 1_000,
                ],
                max_steps: 55,
                max_decisions: 64,
            },
            State {
                simulation: SimulationState {
                    cp: 700,
                    durability: 60,
                    progress: 0,
                    quality: 0,
                    unreliable_quality: 0,
                    effects,
                },
                condition: Condition::Normal,
                step: 0,
                decisions: 0,
            },
        )
    }

    #[test]
    fn policy_improvement_scores_current_legal_candidates() {
        let (model, state) = fixture();
        let recommendation = recommend(&model, state, 1).unwrap();
        assert!(recommendation.planned);
        assert!(recommendation.candidate_count > 1);
        assert!(
            recommendation.rollout_count >= recommendation.candidate_count * POLICY_PILOT_ROLLOUTS
        );
        assert!(
            model
                .apply_outcome(state, recommendation.action, true, 0.0)
                .is_ok()
        );
    }

    #[test]
    fn opening_action_is_muscle_memory_or_reflect() {
        let (model, state) = fixture();
        let legal = (0..ACTION_COUNT)
            .filter(|action| planner_legal(&model, state, *action))
            .collect::<Vec<_>>();
        assert_eq!(legal, vec![MUSCLE_MEMORY, REFLECT]);

        let recommendation = recommend(&model, state, 7).unwrap();
        assert!(matches!(
            recommendation.action,
            Action::MuscleMemory | Action::Reflect
        ));
    }

    #[test]
    fn heart_and_soul_requires_normal_and_an_immediate_followup() {
        let (model, mut state) = fixture();
        state.step = 1;
        state.decisions = 1;
        state.simulation.effects = state.simulation.effects.with_combo(Combo::None);

        state.condition = Condition::Centered;
        assert!(!planner_legal(&model, state, HEART_AND_SOUL));

        state.condition = Condition::Normal;
        assert!(planner_legal(&model, state, HEART_AND_SOUL));
        let active = model
            .apply_outcome(state, Action::HeartAndSoul, true, 0.0)
            .unwrap();
        assert!(active.simulation.effects.heart_and_soul_active());
        let legal = (0..ACTION_COUNT)
            .filter(|action| planner_legal(&model, active, *action))
            .collect::<Vec<_>>();
        assert!(!legal.is_empty());
        assert!(
            legal
                .iter()
                .all(|action| is_heart_and_soul_followup(*action))
        );
        assert!(matches!(
            recommend(&model, active, 11).unwrap().action,
            Action::TricksOfTheTrade | Action::IntensiveSynthesis | Action::PreciseTouch
        ));

        state.decisions = model.max_decisions - 1;
        assert!(!planner_legal(&model, state, HEART_AND_SOUL));
    }

    #[test]
    fn forbidden_actions_are_not_gabriel_actions() {
        let (model, mut state) = fixture();
        state.simulation.effects = state
            .simulation
            .effects
            .with_careful_observation_charges(3)
            .with_crafter_delineations(2)
            .with_quick_innovation_available(true);
        assert!(
            model
                .settings
                .allowed_actions
                .has(Action::CarefulObservation)
        );
        assert!(model.settings.allowed_actions.has(Action::QuickInnovation));
        assert!(model.settings.allowed_actions.has(Action::FinalAppraisal));
        assert!(!planner_legal(&model, state, FINAL_APPRAISAL));
        assert!(!planner_legal(&model, state, CAREFUL_OBSERVATION));
        assert!(!planner_legal(&model, state, QUICK_INNOVATION));
        assert!(!ACTIONS.contains(&Action::StellarSteadyHand));

        let actor = ActorModel::bundled().unwrap();
        let (candidates, _) = candidate_actions(&model, state, actor);
        assert!(!candidates.contains(&FINAL_APPRAISAL));
        assert!(!candidates.contains(&CAREFUL_OBSERVATION));
        assert!(!candidates.contains(&QUICK_INNOVATION));
    }

    #[test]
    fn dense_terminal_value_depends_on_relative_recipe_state() {
        let (mut first_model, mut first_state) = fixture();
        first_model.settings.max_progress = 100;
        first_model.settings.max_quality = 1_000;
        first_model.required_quality = 1_000;
        first_state.simulation.progress = 50;
        first_state.simulation.quality = 500;

        let mut second_model = first_model;
        second_model.settings.max_progress = 200;
        second_model.settings.max_quality = 2_000;
        second_model.required_quality = 2_000;
        let mut second_state = first_state;
        second_state.simulation.progress = 100;
        second_state.simulation.quality = 1_000;

        assert_eq!(
            terminal_utility(&first_model, first_state),
            terminal_utility(&second_model, second_state)
        );
    }

    #[test]
    fn probability_estimate_is_bounded() {
        let (model, state) = fixture();
        let estimate = estimate_full_quality_probability(&model, state, 8, 1).unwrap();
        assert!(estimate.successes <= estimate.samples);
    }

    #[test]
    fn worker_pool_honors_requested_size_and_reconfigures() {
        let one_worker = pool(1).unwrap();
        assert_eq!(one_worker.current_num_threads(), 1);

        let two_workers = pool(2).unwrap();
        assert_eq!(two_workers.current_num_threads(), 2);

        let error = match pool(0) {
            Ok(_) => panic!("zero Gabriel workers unexpectedly accepted"),
            Err(error) => error,
        };
        assert!(error.contains("within 1..=256"));
    }

    #[test]
    fn actor_proposal_remains_a_viable_rollout_incumbent() {
        let (model, initial) = fixture();
        let actor = ActorModel::bundled().unwrap();
        let successes = (0..1_000)
            .filter(|sample| {
                let mut state = initial;
                let mut stream = CounterStream::new(derive_seed(1, *sample));
                while model.status(state) == TerminalStatus::Active {
                    let Some(action) = shielded_actor_action_finish(&model, state, actor) else {
                        return false;
                    };
                    let Ok(next) = model.apply_counter(state, ACTIONS[action], &mut stream) else {
                        return false;
                    };
                    state = next;
                }
                model.status(state) == TerminalStatus::Complete
            })
            .count();

        assert!(successes > 0, "actor proposal produced no legal successes");
    }

    #[test]
    fn paired_policy_admission_requires_objective_evidence() {
        let evaluation = |values| ActionEvaluation {
            action: 0,
            values,
            actor_score: 0.0,
        };
        assert!(paired_improvement(
            &evaluation(vec![1_000.0, 10.0, 10.0]),
            &evaluation(vec![0.0, 0.0, 0.0])
        ));
        assert!(!paired_improvement(
            &evaluation(vec![0.0, 0.0, 0.0]),
            &evaluation(vec![1_000.0, 10.0, 10.0])
        ));
        assert!(paired_improvement(
            &evaluation(vec![10.0; 8]),
            &evaluation(vec![0.0; 8])
        ));
        assert!(!paired_improvement(
            &evaluation(vec![10.0, 0.0]),
            &evaluation(vec![0.0, 10.0])
        ));
    }

    #[test]
    fn condition_opportunities_follow_expert_condition_mechanics() {
        let (model, mut state) = fixture();
        let action = |action| {
            ACTIONS
                .iter()
                .position(|candidate| *candidate == action)
                .unwrap()
        };
        let basic_touch = action(Action::BasicTouch);
        let tricks = action(Action::TricksOfTheTrade);

        state.condition = Condition::Good;
        assert_eq!(
            condition_opportunity(&model, state, PRECISE_TOUCH)
                .unwrap()
                .tier,
            2
        );
        assert_eq!(
            condition_opportunity(&model, state, basic_touch)
                .unwrap()
                .tier,
            1
        );
        assert_eq!(condition_opportunity(&model, state, tricks), None);

        state.condition = Condition::Centered;
        assert!(condition_opportunity(&model, state, HASTY_TOUCH).is_some());
        assert_eq!(condition_opportunity(&model, state, basic_touch), None);

        state.condition = Condition::Sturdy;
        assert!(condition_opportunity(&model, state, GROUNDWORK).is_some());
        assert_eq!(condition_opportunity(&model, state, VENERATION), None);

        state.condition = Condition::Robust;
        assert!(condition_opportunity(&model, state, GROUNDWORK).is_some());

        state.condition = Condition::Pliant;
        assert!(condition_opportunity(&model, state, INNOVATION).is_some());
        assert_eq!(condition_opportunity(&model, state, BASIC_SYNTHESIS), None);

        state.condition = Condition::Malleable;
        assert!(condition_opportunity(&model, state, GROUNDWORK).is_some());
        assert_eq!(condition_opportunity(&model, state, basic_touch), None);

        state.condition = Condition::Primed;
        assert!(condition_opportunity(&model, state, INNOVATION).is_some());
        assert_eq!(condition_opportunity(&model, state, basic_touch), None);
        state.simulation.effects = state.simulation.effects.with_innovation(4);
        assert_eq!(condition_opportunity(&model, state, INNOVATION), None);
        state.simulation.effects = state.simulation.effects.with_innovation(0);

        state.condition = Condition::GoodOmen;
        assert!(condition_opportunity(&model, state, GREAT_STRIDES).is_some());

        state.condition = Condition::Normal;
        assert_eq!(condition_opportunity(&model, state, GROUNDWORK), None);
    }

    #[test]
    fn condition_prior_adds_evaluation_without_evicting_empirical_finalists() {
        let evaluation = |action, value| ActionEvaluation {
            action,
            values: vec![value; POLICY_SCREEN_ROLLOUTS],
            actor_score: 0.0,
        };
        let ranked = vec![
            evaluation(1, 40.0),
            evaluation(2, 30.0),
            evaluation(3, 20.0),
            evaluation(4, 10.0),
            evaluation(5, 0.0),
        ];

        let finalists = select_finalist_actions(&ranked, 4, Some(5));

        assert_eq!(finalists, vec![4, 1, 2, 3, 5]);
        let evaluated = ranked
            .into_iter()
            .filter(|evaluation| finalists.contains(&evaluation.action))
            .collect::<Vec<_>>();
        assert_eq!(best_action(&evaluated), Some(1));
    }

    #[test]
    fn impossible_full_quality_root_returns_game_terminal_failure_closure() {
        let (model, mut state) = fixture();
        state.simulation = SimulationState {
            cp: 0,
            durability: 5,
            progress: 9_000,
            quality: 20_000,
            unreliable_quality: 0,
            effects: Effects::new().with_splendor_cosmic(true),
        };
        state.condition = Condition::Centered;
        state.step = 42;
        state.decisions = 44;

        let recommendation = recommend(&model, state, 1).unwrap();
        assert_eq!(recommendation.action, Action::BasicSynthesis);
        assert!(recommendation.failure_closure);
    }

    #[test]
    fn subquality_progress_completion_is_only_a_failure_closure() {
        let (model, mut state) = fixture();
        state.simulation.progress = model.settings.max_progress - 1;
        state.simulation.quality = 0;
        state.simulation.effects = state.simulation.effects.with_combo(Combo::None);
        state.step = 1;
        state.decisions = 1;

        assert!(!planner_legal(&model, state, BASIC_SYNTHESIS));
        assert!(failure_closure_legal(&model, state));
    }

    #[test]
    fn completed_quality_with_no_safe_progress_candidate_uses_game_legal_closure() {
        let (model, mut state) = fixture();
        state.simulation = SimulationState {
            cp: 0,
            durability: 5,
            progress: 9_000,
            quality: model.required_quality,
            unreliable_quality: 0,
            effects: Effects::new().with_splendor_cosmic(true),
        };
        state.condition = Condition::Centered;
        state.step = 42;
        state.decisions = 44;

        let recommendation = recommend(&model, state, 1).unwrap();
        assert_eq!(recommendation.action, Action::BasicSynthesis);
        assert!(recommendation.failure_closure);
    }
}
