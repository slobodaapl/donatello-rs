#![recursion_limit = "256"]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use raphael_data::{CrafterStats, RECIPES, RLVLS, get_game_settings};
use raphael_sim::{Action, ActionMask, Condition, Settings, SimulationState};
use raphael_solver::{AtomicFlag, MacroSolver, SolverSettings};
use serde::{Deserialize, Serialize};

const CONDITIONS: [Condition; 11] = [
    Condition::Normal,
    Condition::Good,
    Condition::Excellent,
    Condition::Poor,
    Condition::Centered,
    Condition::Sturdy,
    Condition::Pliant,
    Condition::Malleable,
    Condition::Primed,
    Condition::GoodOmen,
    Condition::Robust,
];
static DEADLINE_MILLIS: AtomicU64 = AtomicU64::new(0);
static INITIAL_ONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static QUIET_FAILURES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[derive(Clone, Copy)]
enum Mode {
    Quick,
    Full,
}

#[derive(Clone, Copy)]
struct Fixture {
    name: &'static str,
    max_cp: u16,
    max_durability: u16,
    max_progress: u16,
    max_quality: u16,
    base_progress: u16,
    base_quality: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PlanScore {
    completes: bool,
    quality: u16,
    actions: u16,
    duration: u16,
}

impl PlanScore {
    fn strictly_better_than(self, incumbent: Self) -> bool {
        if self.completes != incumbent.completes {
            return self.completes;
        }
        if self.quality != incumbent.quality {
            return self.quality > incumbent.quality;
        }
        if self.actions != incumbent.actions {
            return self.actions < incumbent.actions;
        }
        self.duration < incumbent.duration
    }

    fn ties_or_beats(self, incumbent: Self) -> bool {
        self == incumbent || self.strictly_better_than(incumbent)
    }
}

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConditionSummary {
    scenarios: u64,
    strict_wins: u64,
    ties: u64,
    losses: u64,
    rejected_candidates: u64,
    solve_failures: u64,
    completion_regressions: u64,
    non_optimal: u64,
    total_quality_bound_gap: u64,
    max_quality_bound_gap: u16,
    total_quality_gain: u64,
    max_quality_gain: u16,
    total_action_reduction: i64,
    total_duration_reduction: i64,
    solve_micros: u128,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FixtureSummary {
    name: String,
    job_level: Option<u8>,
    expert: Option<bool>,
    baseline_actions: usize,
    baseline_score: PlanScore,
    elapsed_millis: u128,
    conditions: BTreeMap<String, ConditionSummary>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    mode: &'static str,
    deadline_millis: u64,
    fixtures: Vec<FixtureSummary>,
    totals: ConditionSummary,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Corpus {
    version: u32,
    cases: Vec<CorpusCase>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CorpusCase {
    name: String,
    job_level: Option<u8>,
    expert: Option<bool>,
    #[serde(default)]
    recipe_id: Option<u32>,
    #[serde(default)]
    item_id: Option<u32>,
    #[serde(default)]
    crafter_stats: Option<CrafterStats>,
    #[serde(default)]
    recipe_level: Option<raphael_data::RecipeLevel>,
    baseline_settings: Settings,
    adaptive_settings: Settings,
    #[serde(default)]
    crafter_delineations: u8,
}

struct RealFixture {
    name: String,
    recipe_id: u32,
    item_id: u32,
    job_level: u8,
    expert: bool,
    crafter_stats: CrafterStats,
    recipe_level: raphael_data::RecipeLevel,
    baseline_settings: Settings,
    adaptive_settings: Settings,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReferenceResults {
    version: u32,
    cases: Vec<ReferenceCase>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReferenceCase {
    name: String,
    cold_micros: u128,
    warm_micros: u128,
    action_ids: Vec<u32>,
    score: PlanScore,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeFfiResponse {
    ok: bool,
    action_ids: Option<Vec<u32>>,
    optimal: Option<bool>,
    quality_upper_bound: Option<u16>,
    achieved_quality: Option<u16>,
    progress_boundary: Option<RuntimeProgressBoundary>,
    error: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeProgressBoundary {
    action_count: usize,
    target: String,
}

struct RuntimeSolveOutcome {
    actions: Vec<Action>,
    optimal: bool,
    quality_bound_gap: u16,
    staged_progress_plan: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonCase {
    name: String,
    reference_score: PlanScore,
    candidate_score: PlanScore,
    standalone_score: PlanScore,
    reference_cold_micros: u128,
    reference_warm_micros: u128,
    candidate_cold_micros: u128,
    candidate_warm_micros: u128,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonReport {
    cases: Vec<ComparisonCase>,
    reference_cold_p95_micros: u128,
    reference_warm_p95_micros: u128,
    candidate_cold_p95_micros: u128,
    candidate_warm_p95_micros: u128,
}

fn corpus_case_variants(
    name: String,
    job_level: Option<u8>,
    expert: Option<bool>,
    recipe_id: Option<u32>,
    item_id: Option<u32>,
    crafter_stats: Option<CrafterStats>,
    recipe_level: Option<raphael_data::RecipeLevel>,
    baseline_settings: Settings,
    adaptive_settings: Settings,
    repeat_specialist: bool,
) -> Vec<CorpusCase> {
    let mut cases = vec![CorpusCase {
        name: name.clone(),
        job_level,
        expert,
        recipe_id,
        item_id,
        crafter_stats,
        recipe_level,
        baseline_settings,
        adaptive_settings,
        crafter_delineations: 0,
    }];
    if repeat_specialist {
        for crafter_delineations in 0..=2 {
            let mut specialist_baseline = baseline_settings;
            specialist_baseline.allowed_actions = specialist_baseline
                .allowed_actions
                .add(Action::HeartAndSoul)
                .add(Action::QuickInnovation);
            let mut specialist_adaptive = adaptive_settings;
            specialist_adaptive.allowed_actions = specialist_adaptive
                .allowed_actions
                .add(Action::HeartAndSoul)
                .add(Action::QuickInnovation);
            cases.push(CorpusCase {
                name: format!("{name}-specialist-d{crafter_delineations}"),
                job_level,
                expert,
                recipe_id,
                item_id,
                crafter_stats,
                recipe_level,
                baseline_settings: specialist_baseline,
                adaptive_settings: specialist_adaptive,
                crafter_delineations,
            });
        }
    }
    cases
}

fn quick_actions() -> ActionMask {
    ActionMask::none()
        .add(Action::BasicSynthesis)
        .add(Action::CarefulSynthesis)
        .add(Action::BasicTouch)
        .add(Action::StandardTouch)
        .add(Action::AdvancedTouch)
        .add(Action::PreciseTouch)
        .add(Action::IntensiveSynthesis)
        .add(Action::TricksOfTheTrade)
        .add(Action::Innovation)
        .add(Action::ByregotsBlessing)
        .add(Action::MasterMend)
        .add(Action::FinalAppraisal)
}

fn action_mask(mode: Mode, include_final_appraisal: bool) -> ActionMask {
    let mask = match mode {
        Mode::Quick => quick_actions(),
        Mode::Full => ActionMask::regular()
            .remove(Action::RapidSynthesis)
            .remove(Action::HastyTouch)
            .remove(Action::DaringTouch),
    };
    if include_final_appraisal {
        mask.add(Action::FinalAppraisal)
    } else {
        mask.remove(Action::FinalAppraisal)
    }
}

fn settings(fixture: Fixture, mode: Mode, include_final_appraisal: bool) -> Settings {
    Settings {
        max_cp: fixture.max_cp,
        max_durability: fixture.max_durability,
        max_progress: fixture.max_progress,
        max_quality: fixture.max_quality,
        base_progress: fixture.base_progress,
        base_quality: fixture.base_quality,
        job_level: 100,
        allowed_actions: action_mask(mode, include_final_appraisal),
        adversarial: false,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    }
}

fn solver(settings: Settings) -> MacroSolver<'static> {
    MacroSolver::new(
        SolverSettings {
            simulator_settings: settings,
            allow_non_max_quality_solutions: true,
        },
        Box::new(|_| {}),
        Box::new(|_| {}),
        AtomicFlag::new(),
    )
}

fn condition_id(condition: Condition) -> u8 {
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

fn solve_runtime_ffi_staged(
    settings: Settings,
    root: SimulationState,
    condition: Condition,
    incumbent: &[Action],
    crafter_delineations: u8,
    progress_first: bool,
) -> Result<RuntimeSolveOutcome, String> {
    let deadline = DEADLINE_MILLIS.load(Ordering::Relaxed);
    let request = serde_json::json!({
        "abiVersion": 9,
        "maxCp": settings.max_cp,
        "maxDurability": settings.max_durability,
        "maxProgress": settings.max_progress,
        "maxQuality": settings.max_quality,
        "baseProgress": settings.base_progress,
        "baseQuality": settings.base_quality,
        "jobLevel": settings.job_level,
        "manipulation": settings.allowed_actions.has(Action::Manipulation),
        "specialist": settings.allowed_actions.has(Action::HeartAndSoul)
            || settings.allowed_actions.has(Action::QuickInnovation),
        "allowCarefulObservation": false,
        "solveMode": 2,
        "minimizeSteps": true,
        "progressFirst": progress_first,
        "stellarSteadyHandCharges": root.effects.stellar_steady_hand_charges(),
        "incumbentActionIds": incumbent.iter().map(|action| action.action_id()).collect::<Vec<_>>(),
        "softDeadlineMillis": deadline,
        "hardDeadlineMillis": deadline,
        "bypassSolutionCache": true,
        "root": {
            "cp": root.cp,
            "durability": root.durability,
            "progress": root.progress,
            "quality": root.quality,
            "innerQuiet": root.effects.inner_quiet(),
            "wasteNot": root.effects.waste_not(),
            "manipulation": root.effects.manipulation(),
            "innovation": root.effects.innovation(),
            "veneration": root.effects.veneration(),
            "greatStrides": root.effects.great_strides(),
            "muscleMemory": root.effects.muscle_memory(),
            "finalAppraisal": root.effects.final_appraisal(),
            "carefulObservationCharges": root.effects.careful_observation_charges(),
            "combo": root.effects.combo().into_bits(),
            "heartAndSoulActive": root.effects.heart_and_soul_active(),
            "heartAndSoulAvailable": root.effects.heart_and_soul_available(),
            "quickInnovationAvailable": root.effects.quick_innovation_available(),
            "trainedPerfectionActive": root.effects.trained_perfection_active(),
            "trainedPerfectionAvailable": root.effects.trained_perfection_available(),
            "stellarSteadyHandCharges": root.effects.stellar_steady_hand_charges(),
            "stellarSteadyHand": root.effects.stellar_steady_hand(),
            "splendorCosmic": root.effects.splendor_cosmic(),
            "expedience": root.effects.expedience(),
            "condition": condition_id(condition),
            "crafterDelineations": crafter_delineations,
        }
    });
    let response: RuntimeFfiResponse = serde_json::from_str(&donatello_ffi::solve_json(
        &serde_json::to_vec(&request).map_err(|error| error.to_string())?,
    ))
    .map_err(|error| error.to_string())?;
    if !response.ok {
        return Err(response.error.unwrap_or_else(|| String::from("unknown FFI solve failure")));
    }
    let actions = response
        .action_ids
        .ok_or_else(|| String::from("FFI response omitted actions"))?
        .into_iter()
        .map(|id| Action::from_action_id(id).ok_or_else(|| format!("unknown FFI action {id}")))
        .collect::<Result<Vec<_>, _>>()?;
    let staged_progress_plan = response.progress_boundary.is_some_and(|boundary| {
        boundary.target == "oneShort"
            && boundary.action_count > 0
            && boundary.action_count < actions.len()
    });
    Ok(RuntimeSolveOutcome {
        actions,
        optimal: response.optimal.unwrap_or(false),
        quality_bound_gap: response
            .quality_upper_bound
            .unwrap_or_default()
            .saturating_sub(response.achieved_quality.unwrap_or_default()),
        staged_progress_plan,
    })
}

fn start_deadline(
    interrupt: AtomicFlag,
) -> (
    std::sync::mpsc::Sender<()>,
    Option<std::thread::JoinHandle<()>>,
) {
    start_deadline_for(interrupt, DEADLINE_MILLIS.load(Ordering::Relaxed))
}

fn start_deadline_for(
    interrupt: AtomicFlag,
    deadline_millis: u64,
) -> (
    std::sync::mpsc::Sender<()>,
    Option<std::thread::JoinHandle<()>>,
) {
    let (done, wake) = std::sync::mpsc::channel::<()>();
    let worker = (deadline_millis > 0).then(|| {
        std::thread::spawn(move || {
            if wake
                .recv_timeout(Duration::from_millis(deadline_millis))
                .is_err_and(|error| error == std::sync::mpsc::RecvTimeoutError::Timeout)
            {
                interrupt.set();
            }
        })
    });
    (done, worker)
}

fn finish_deadline(done: std::sync::mpsc::Sender<()>, worker: Option<std::thread::JoinHandle<()>>) {
    drop(done);
    if let Some(worker) = worker {
        let _ = worker.join();
    }
}

fn solve_timed(
    solver: &mut MacroSolver<'_>,
    root: SimulationState,
) -> (Result<Vec<Action>, String>, Duration) {
    let interrupt = AtomicFlag::new();
    solver.set_interrupt_signal(interrupt.clone());
    let (done, worker) = start_deadline(interrupt);
    let started = Instant::now();
    let result = solver
        .solve_from_state_with_condition_and_incumbent_anytime(root, Condition::Normal, &[], false)
        .map(|outcome| outcome.actions)
        .map_err(|error| format!("{error:?}"));
    let elapsed = started.elapsed();
    finish_deadline(done, worker);
    (result, elapsed)
}

fn solve_with_incumbent_timed(
    solver: &mut MacroSolver<'_>,
    root: SimulationState,
    incumbent: &[Action],
) -> (Result<Vec<Action>, String>, Duration) {
    let interrupt = AtomicFlag::new();
    solver.set_interrupt_signal(interrupt.clone());
    let (done, worker) = start_deadline(interrupt);
    let started = Instant::now();
    let result = solver
        .solve_from_state_with_condition_and_incumbent_anytime(
            root,
            Condition::Normal,
            incumbent,
            false,
        )
        .map(|outcome| outcome.actions)
        .map_err(|error| format!("{error:?}"));
    let elapsed = started.elapsed();
    finish_deadline(done, worker);
    (result, elapsed)
}

fn percentile_95(values: &mut [u128]) -> u128 {
    assert!(!values.is_empty());
    values.sort_unstable();
    values[(values.len() * 95).div_ceil(100) - 1]
}

fn compare_reference(corpus_cases: Vec<CorpusCase>, reference_path: &str) -> ComparisonReport {
    let bytes = std::fs::read(reference_path).expect("failed to read reference results");
    let reference: ReferenceResults =
        serde_json::from_slice(&bytes).expect("invalid reference results JSON");
    assert_eq!(
        reference.version, 1,
        "unsupported reference results version"
    );
    let mut references = reference
        .cases
        .into_iter()
        .map(|case| (case.name.clone(), case))
        .collect::<BTreeMap<_, _>>();
    let mut cases = Vec::with_capacity(corpus_cases.len());
    for case in corpus_cases {
        eprintln!("comparison fixture: {}", case.name);
        let reference = references
            .remove(&case.name)
            .unwrap_or_else(|| panic!("missing reference result for {}", case.name));
        let reference_actions = reference
            .action_ids
            .iter()
            .map(|&id| {
                Action::from_action_id(id).unwrap_or_else(|| panic!("unknown action ID {id}"))
            })
            .collect::<Vec<_>>();
        let mut root = SimulationState::new(&case.adaptive_settings);
        root.effects
            .set_crafter_delineations(case.crafter_delineations);
        root.effects = root.effects.canonicalize_specialist_resources();
        let replayed_reference = evaluate(
            &case.adaptive_settings,
            root,
            Condition::Normal,
            &reference_actions,
        );
        assert_eq!(
            replayed_reference, reference.score,
            "{} reference plan changed semantics in current raphael-sim",
            case.name
        );
        let mut candidate_solver = solver(case.adaptive_settings);
        let (standalone, cold_duration) = solve_timed(&mut candidate_solver, root);
        let standalone = standalone
            .unwrap_or_else(|error| panic!("{} standalone cold solve failed: {error}", case.name));
        let standalone_score = evaluate(
            &case.adaptive_settings,
            root,
            Condition::Normal,
            &standalone,
        );
        eprintln!(
            "standalone cold: {:?}, {}us, {:?}",
            standalone_score,
            cold_duration.as_micros(),
            candidate_solver.runtime_stats()
        );
        assert!(
            standalone_score.completes,
            "{} standalone cold candidate did not complete",
            case.name
        );
        assert!(
            standalone_score.ties_or_beats(reference.score),
            "{} standalone cold candidate regressed: reference={:?}, candidate={:?}",
            case.name,
            reference.score,
            standalone_score
        );
        let (warm_standalone, warm_duration) = solve_timed(&mut candidate_solver, root);
        let warm_standalone = warm_standalone
            .unwrap_or_else(|error| panic!("{} standalone warm solve failed: {error}", case.name));
        let warm_standalone_score = evaluate(
            &case.adaptive_settings,
            root,
            Condition::Normal,
            &warm_standalone,
        );
        assert!(
            warm_standalone_score.completes,
            "{} standalone warm candidate did not complete",
            case.name
        );
        assert!(
            warm_standalone_score.ties_or_beats(reference.score),
            "{} standalone warm candidate regressed: reference={:?}, candidate={:?}",
            case.name,
            reference.score,
            warm_standalone_score
        );
        let (semantic, _) =
            solve_with_incumbent_timed(&mut candidate_solver, root, &reference_actions);
        let semantic = semantic
            .unwrap_or_else(|error| panic!("{} incumbent solve failed: {error}", case.name));
        let candidate_score = evaluate(&case.adaptive_settings, root, Condition::Normal, &semantic);
        assert!(
            candidate_score.ties_or_beats(reference.score),
            "{} regressed: reference={:?}, candidate={:?}",
            case.name,
            reference.score,
            candidate_score
        );
        cases.push(ComparisonCase {
            name: case.name,
            reference_score: reference.score,
            candidate_score,
            standalone_score,
            reference_cold_micros: reference.cold_micros,
            reference_warm_micros: reference.warm_micros,
            candidate_cold_micros: cold_duration.as_micros(),
            candidate_warm_micros: warm_duration.as_micros(),
        });
    }
    let mut reference_cold = cases
        .iter()
        .map(|case| case.reference_cold_micros)
        .collect::<Vec<_>>();
    let mut reference_warm = cases
        .iter()
        .map(|case| case.reference_warm_micros)
        .collect::<Vec<_>>();
    let mut candidate_cold = cases
        .iter()
        .map(|case| case.candidate_cold_micros)
        .collect::<Vec<_>>();
    let mut candidate_warm = cases
        .iter()
        .map(|case| case.candidate_warm_micros)
        .collect::<Vec<_>>();
    let report = ComparisonReport {
        reference_cold_p95_micros: percentile_95(&mut reference_cold),
        reference_warm_p95_micros: percentile_95(&mut reference_warm),
        candidate_cold_p95_micros: percentile_95(&mut candidate_cold),
        candidate_warm_p95_micros: percentile_95(&mut candidate_warm),
        cases,
    };
    assert!(references.is_empty(), "reference contained unmatched cases");
    report
}

fn evaluate(
    settings: &Settings,
    root: SimulationState,
    initial_condition: Condition,
    actions: &[Action],
) -> PlanScore {
    let mut state = root;
    let mut condition = initial_condition;
    let mut executed = 0_u16;
    let mut duration = 0_u16;
    for &action in actions {
        if state.is_final(settings) || state.durability == 0 {
            break;
        }
        let Ok(next) = state.use_action(action, condition, settings) else {
            break;
        };
        state = next;
        executed += 1;
        duration += u16::from(action.time_cost());
        if action.advances_condition() {
            condition = condition.deterministic_successor();
        }
    }
    PlanScore {
        completes: state.is_final(settings),
        quality: state.quality.min(settings.max_quality),
        actions: executed,
        duration,
    }
}

fn baseline_boundaries(
    settings: &Settings,
    root: SimulationState,
    actions: &[Action],
) -> Vec<SimulationState> {
    let mut states = Vec::with_capacity(actions.len());
    let mut state = root;
    for &action in actions {
        if state.is_final(settings) || state.durability == 0 {
            break;
        }
        states.push(state);
        state = state
            .use_action(action, Condition::Normal, settings)
            .expect("baseline Raphael plan must replay under Normal");
    }
    states
}

fn record_scenario(
    summary: &mut ConditionSummary,
    incumbent: PlanScore,
    candidate: Result<PlanScore, ()>,
    solve_duration: Duration,
    optimal: bool,
    quality_bound_gap: u16,
    accept_staged_candidate: bool,
) {
    summary.scenarios += 1;
    summary.solve_micros += solve_duration.as_micros();
    if !optimal {
        summary.non_optimal += 1;
        summary.total_quality_bound_gap += u64::from(quality_bound_gap);
        summary.max_quality_bound_gap = summary.max_quality_bound_gap.max(quality_bound_gap);
    }
    let Ok(candidate) = candidate else {
        summary.solve_failures += 1;
        summary.ties += 1;
        return;
    };
    let selected = if candidate.strictly_better_than(incumbent) || accept_staged_candidate {
        candidate
    } else {
        incumbent
    };
    if selected.strictly_better_than(incumbent) {
        summary.strict_wins += 1;
        summary.total_quality_gain += u64::from(selected.quality.saturating_sub(incumbent.quality));
        summary.max_quality_gain = summary
            .max_quality_gain
            .max(selected.quality.saturating_sub(incumbent.quality));
        summary.total_action_reduction +=
            i64::from(incumbent.actions) - i64::from(selected.actions);
        summary.total_duration_reduction +=
            i64::from(incumbent.duration) - i64::from(selected.duration);
    } else if selected == incumbent {
        summary.ties += 1;
        if candidate != incumbent {
            summary.rejected_candidates += 1;
        }
    } else {
        summary.losses += 1;
    }
    if incumbent.completes && !selected.completes {
        summary.completion_regressions += 1;
    }
}

#[cfg(test)]
fn select_candidate(incumbent: PlanScore, candidate: PlanScore) -> PlanScore {
    if candidate.strictly_better_than(incumbent) {
        candidate
    } else {
        incumbent
    }
}

fn merge(target: &mut ConditionSummary, source: &ConditionSummary) {
    target.scenarios += source.scenarios;
    target.strict_wins += source.strict_wins;
    target.ties += source.ties;
    target.losses += source.losses;
    target.rejected_candidates += source.rejected_candidates;
    target.solve_failures += source.solve_failures;
    target.completion_regressions += source.completion_regressions;
    target.non_optimal += source.non_optimal;
    target.total_quality_bound_gap += source.total_quality_bound_gap;
    target.max_quality_bound_gap = target
        .max_quality_bound_gap
        .max(source.max_quality_bound_gap);
    target.total_quality_gain += source.total_quality_gain;
    target.max_quality_gain = target.max_quality_gain.max(source.max_quality_gain);
    target.total_action_reduction += source.total_action_reduction;
    target.total_duration_reduction += source.total_duration_reduction;
    target.solve_micros += source.solve_micros;
}

fn run_case(
    name: String,
    job_level: Option<u8>,
    expert: Option<bool>,
    baseline_settings: Settings,
    adaptive_settings: Settings,
    crafter_delineations: u8,
    runtime_ffi: bool,
) -> Option<FixtureSummary> {
    let started = Instant::now();
    eprintln!("running fixture: {name}");
    let mut root = SimulationState::new(&baseline_settings);
    root.effects.set_crafter_delineations(crafter_delineations);
    root.effects = root.effects.canonicalize_specialist_resources();
    let mut raphael = solver(baseline_settings);
    let baseline_interrupt = AtomicFlag::new();
    raphael.set_interrupt_signal(baseline_interrupt.clone());
    let (deadline_done, deadline_worker) = start_deadline_for(baseline_interrupt, 30_000);
    let baseline = raphael.solve_from_state(root);
    finish_deadline(deadline_done, deadline_worker);
    let Ok(baseline) = baseline else {
        eprintln!("skipping unsolved baseline fixture: {name}");
        return None;
    };
    let baseline_score = evaluate(&baseline_settings, root, Condition::Normal, &baseline);
    assert!(
        baseline_score.completes,
        "baseline Raphael plan did not complete"
    );
    let boundaries = if INITIAL_ONLY.load(Ordering::Relaxed) {
        vec![root]
    } else {
        baseline_boundaries(&baseline_settings, root, &baseline)
    };
    let mut donatello = solver(adaptive_settings);
    let mut conditions = BTreeMap::new();

    let conditions_to_run = if INITIAL_ONLY.load(Ordering::Relaxed) {
        &CONDITIONS[..1]
    } else {
        &CONDITIONS
    };
    for &condition in conditions_to_run {
        let mut summary = ConditionSummary::default();
        for (index, &root) in boundaries.iter().enumerate() {
            let incumbent = evaluate(&adaptive_settings, root, condition, &baseline[index..]);
            let solve_started = Instant::now();
            let solved = if runtime_ffi {
                solve_runtime_ffi_staged(
                    adaptive_settings,
                    root,
                    condition,
                    &baseline[index..],
                    crafter_delineations,
                    expert != Some(true),
                )
            } else {
                let interrupt = AtomicFlag::new();
                donatello.set_interrupt_signal(interrupt.clone());
                let (deadline_done, deadline_worker) = start_deadline(interrupt);
                let result = donatello
                    .solve_from_state_with_condition_and_incumbent_anytime(
                        root,
                        condition,
                        &baseline[index..],
                        true,
                    )
                    .map(|outcome| RuntimeSolveOutcome {
                        actions: outcome.actions,
                        optimal: outcome.optimal,
                        quality_bound_gap: outcome
                            .quality_upper_bound
                            .saturating_sub(outcome.quality),
                        staged_progress_plan: false,
                    })
                    .map_err(|error| format!("{error:?}"));
                finish_deadline(deadline_done, deadline_worker);
                result
            };
            let (optimal, quality_bound_gap, _staged_progress_plan) = solved
                .as_ref()
                .map_or((false, 0, false), |outcome| {
                    (
                        outcome.optimal,
                        outcome.quality_bound_gap,
                        outcome.staged_progress_plan,
                    )
                });
            let candidate = solved
                .map(|outcome| evaluate(&adaptive_settings, root, condition, &outcome.actions))
                .map_err(|error| {
                    if !QUIET_FAILURES.load(Ordering::Relaxed) {
                        eprintln!(
                            "solve failure: fixture={}, condition={condition:?}, boundary={index}, root={root:?}, incumbent={incumbent:?}, incumbent_actions={:?}, error={error:?}",
                            name,
                            &baseline[index..],
                        );
                    }
                });
            record_scenario(
                &mut summary,
                incumbent,
                candidate,
                solve_started.elapsed(),
                optimal,
                quality_bound_gap,
                false,
            );
        }
        assert_eq!(summary.completion_regressions, 0);
        conditions.insert(format!("{condition:?}"), summary);
    }

    Some(FixtureSummary {
        name,
        job_level,
        expert,
        baseline_actions: baseline.len(),
        baseline_score,
        elapsed_millis: started.elapsed().as_millis(),
        conditions,
    })
}

fn real_fixtures(mode: Mode) -> Vec<RealFixture> {
    const CRAFTER_CURVE: [(u8, u16, u16, u16); 10] = [
        (10, 50, 45, 200),
        (20, 110, 100, 230),
        (30, 180, 165, 260),
        (40, 250, 230, 295),
        (50, 350, 325, 340),
        (60, 950, 850, 400),
        (70, 1550, 1450, 470),
        (80, 2450, 2250, 515),
        (90, 3350, 3150, 550),
        (100, 5200, 4800, 630),
    ];
    let mut result = Vec::new();
    for (job_level, craftsmanship, control, cp) in CRAFTER_CURVE {
        let bracket_start = job_level.saturating_sub(9);
        let crafter = CrafterStats {
            craftsmanship,
            control,
            cp,
            level: job_level,
            manipulation: job_level >= 65,
            heart_and_soul: false,
            quick_innovation: false,
        };
        let mut candidates = RECIPES
            .entries()
            .filter(|(_, recipe)| {
                let recipe_job_level = RLVLS[recipe.recipe_level as usize].job_level;
                recipe.max_level_scaling == 0
                    && !recipe.is_expert
                    && recipe_job_level >= bracket_start
                    && recipe_job_level <= job_level
                    && recipe.req_craftsmanship <= crafter.craftsmanship
                    && recipe.req_control <= crafter.control
            })
            .map(|(recipe_id, recipe)| {
                let mut settings = get_game_settings(*recipe, None, crafter, None, None);
                settings.allowed_actions = action_mask(mode, true);
                (recipe_id, *recipe, settings)
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(recipe_id, recipe, settings)| {
            std::cmp::Reverse((
                RLVLS[recipe.recipe_level as usize].job_level,
                recipe.recipe_level,
                (settings.max_progress as u32) * (settings.max_quality as u32),
                *recipe_id,
            ))
        });
        assert!(
            candidates.len() >= 10,
            "level bracket {bracket_start}-{job_level} has fewer than ten eligible regular recipes"
        );
        for (recipe_id, recipe, baseline) in candidates.into_iter().take(10) {
            let recipe_level = RLVLS[recipe.recipe_level as usize];
            let mut adaptive = baseline;
            adaptive.allowed_actions = action_mask(mode, true);
            result.push(RealFixture {
                name: format!(
                    "L{job_level}-regular-recipe{recipe_id}-item{}",
                    recipe.item_id
                ),
                recipe_id,
                item_id: recipe.item_id,
                job_level,
                expert: false,
                crafter_stats: crafter,
                recipe_level,
                baseline_settings: baseline,
                adaptive_settings: adaptive,
            });
        }
    }

    for (job_level, craftsmanship, control, cp) in CRAFTER_CURVE
        .into_iter()
        .filter(|(job_level, _, _, _)| matches!(job_level, 80 | 100))
    {
        let expert_crafter = CrafterStats {
            craftsmanship,
            control,
            cp,
            level: job_level,
            manipulation: true,
            heart_and_soul: false,
            quick_innovation: false,
        };
        let mut experts = RECIPES
            .entries()
            .filter(|(_, recipe)| {
                recipe.max_level_scaling == 0
                    && recipe.is_expert
                    && RLVLS[recipe.recipe_level as usize].job_level == job_level
                    && recipe.req_craftsmanship <= expert_crafter.craftsmanship
                    && recipe.req_control <= expert_crafter.control
            })
            .map(|(recipe_id, recipe)| {
                let mut settings = get_game_settings(*recipe, None, expert_crafter, None, None);
                settings.allowed_actions = action_mask(mode, true);
                (recipe_id, *recipe, settings)
            })
            .collect::<Vec<_>>();
        experts.sort_by_key(|(recipe_id, _, settings)| {
            (
                (settings.max_progress as u32) * (settings.max_quality as u32),
                *recipe_id,
            )
        });
        assert!(
            !experts.is_empty(),
            "no eligible level-{job_level} expert recipes"
        );
        let sample_count = experts.len().min(10);
        let mut sampled_indices = if sample_count == 1 {
            vec![0]
        } else {
            (0..sample_count)
                .map(|index| index * (experts.len() - 1) / (sample_count - 1))
                .collect::<Vec<_>>()
        };
        if job_level == 100
            && let Some(recipe_38202_index) = experts.iter().position(|(id, _, _)| *id == 38202)
            && !sampled_indices.contains(&recipe_38202_index)
        {
            let replacement = sampled_indices
                .iter()
                .enumerate()
                .min_by_key(|(_, sampled)| (**sampled).abs_diff(recipe_38202_index))
                .map(|(index, _)| index)
                .unwrap();
            sampled_indices[replacement] = recipe_38202_index;
            sampled_indices.sort_unstable();
        }
        for index in sampled_indices {
            let (recipe_id, recipe, mut baseline) = experts[index];
            let mut selected_crafter = expert_crafter;
            if recipe_id == 38202 {
                let logged_crafter = CrafterStats {
                    craftsmanship: 5328,
                    control: 4779,
                    cp: 573,
                    ..expert_crafter
                };
                baseline = get_game_settings(recipe, None, logged_crafter, None, None);
                baseline.allowed_actions = action_mask(mode, true);
                selected_crafter = logged_crafter;
            }
            let mut adaptive = baseline;
            adaptive.allowed_actions = action_mask(mode, true);
            result.push(RealFixture {
                name: if recipe_id == 38202 {
                    String::from("recipe38202-logged-stats")
                } else {
                    format!("L{job_level}-expert-recipe{recipe_id}-item{}", recipe.item_id)
                },
                recipe_id,
                item_id: recipe.item_id,
                job_level,
                expert: true,
                crafter_stats: selected_crafter,
                recipe_level: RLVLS[recipe.recipe_level as usize],
                baseline_settings: baseline,
                adaptive_settings: adaptive,
            });
        }
    }
    if matches!(mode, Mode::Full) {
        let logged_crafter = CrafterStats {
            craftsmanship: 1425,
            control: 1314,
            cp: 387,
            level: 91,
            manipulation: false,
            heart_and_soul: false,
            quick_innovation: false,
        };
        for recipe_id in [35643, 35644] {
            let recipe = *RECIPES
                .get(recipe_id)
                .unwrap_or_else(|| panic!("missing logged regression recipe {recipe_id}"));
            let mut baseline = get_game_settings(recipe, None, logged_crafter, None, None);
            baseline.allowed_actions = action_mask(mode, true);
            result.push(RealFixture {
                name: format!("regression-recipe{recipe_id}-logged-l91"),
                recipe_id,
                item_id: recipe.item_id,
                job_level: 91,
                expert: false,
                crafter_stats: logged_crafter,
                recipe_level: RLVLS[recipe.recipe_level as usize],
                baseline_settings: baseline,
                adaptive_settings: baseline,
            });
        }
    }
    result
}

fn fixtures(mode: Mode) -> &'static [Fixture] {
    const QUICK: [Fixture; 3] = [
        Fixture {
            name: "balanced-40",
            max_cp: 180,
            max_durability: 40,
            max_progress: 420,
            max_quality: 900,
            base_progress: 110,
            base_quality: 95,
        },
        Fixture {
            name: "durability-tight-35",
            max_cp: 220,
            max_durability: 35,
            max_progress: 520,
            max_quality: 1100,
            base_progress: 125,
            base_quality: 105,
        },
        Fixture {
            name: "quality-heavy-60",
            max_cp: 260,
            max_durability: 60,
            max_progress: 650,
            max_quality: 1800,
            base_progress: 130,
            base_quality: 110,
        },
    ];
    const FULL: [Fixture; 3] = QUICK;
    match mode {
        Mode::Quick => &QUICK,
        Mode::Full => &FULL,
    }
}

fn main() {
    let mut mode = Mode::Quick;
    let mut json = false;
    let mut real = false;
    let mut expert_high = false;
    let mut summary_only = false;
    let mut emit_corpus = None;
    let mut corpus_path = None;
    let mut reference_results_path = None;
    let mut comparison_output_path = None;
    let mut output_path = None;
    let mut name_filter = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--quick" => mode = Mode::Quick,
            "--full" => mode = Mode::Full,
            "--json" => json = true,
            "--real" => real = true,
            "--expert-high" => {
                real = true;
                expert_high = true;
            }
            "--summary" => summary_only = true,
            "--deadline-ms" => {
                let value = arguments.next().expect("--deadline-ms requires a value");
                DEADLINE_MILLIS.store(value.parse().expect("invalid deadline"), Ordering::Relaxed);
            }
            "--initial-only" => INITIAL_ONLY.store(true, Ordering::Relaxed),
            "--emit-corpus" => {
                emit_corpus = Some(arguments.next().expect("--emit-corpus requires a path"));
            }
            "--corpus" => {
                corpus_path = Some(arguments.next().expect("--corpus requires a path"));
            }
            "--compare-reference" => {
                reference_results_path = Some(
                    arguments
                        .next()
                        .expect("--compare-reference requires a path"),
                );
            }
            "--comparison-output" => {
                comparison_output_path = Some(
                    arguments
                        .next()
                        .expect("--comparison-output requires a path"),
                );
            }
            "--output" => {
                output_path = Some(arguments.next().expect("--output requires a path"));
            }
            "--filter" => {
                name_filter = Some(arguments.next().expect("--filter requires text"));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: donatello-bench [--quick|--full] [--real|--expert-high] [--summary] [--json] [--output PATH] [--deadline-ms N] [--initial-only] [--filter TEXT] [--emit-corpus PATH|--corpus PATH] [--compare-reference PATH] [--comparison-output PATH]"
                );
                return;
            }
            _ => panic!("unknown argument {argument}"),
        }
    }

    QUIET_FAILURES.store(summary_only, Ordering::Relaxed);
    let mut real_cases = real_fixtures(mode);
    if expert_high {
        real_cases.retain(|case| case.expert && case.job_level >= 90);
        real_cases.reverse();
        real_cases.truncate(if matches!(mode, Mode::Quick) { 2 } else { 10 });
    }

    let mut corpus_cases = if let Some(path) = corpus_path.as_ref() {
        let bytes = std::fs::read(path).expect("failed to read corpus");
        let corpus: Corpus = serde_json::from_slice(&bytes).expect("invalid corpus JSON");
        assert_eq!(corpus.version, 1, "unsupported corpus version");
        corpus.cases
    } else if real {
        let mut cases = real_cases
            .into_iter()
            .flat_map(|case| {
                corpus_case_variants(
                    case.name,
                    Some(case.job_level),
                    Some(case.expert),
                    Some(case.recipe_id),
                    Some(case.item_id),
                    Some(case.crafter_stats),
                    Some(case.recipe_level),
                    case.baseline_settings,
                    case.adaptive_settings,
                    case.job_level >= 90,
                )
            })
            .collect::<Vec<_>>();
        for fixture in fixtures(mode) {
            cases.extend(corpus_case_variants(
                fixture.name.to_owned(),
                None,
                None,
                None,
                None,
                None,
                None,
                settings(*fixture, mode, true),
                settings(*fixture, mode, true),
                true,
            ));
        }
        cases
    } else {
        fixtures(mode)
            .iter()
            .copied()
            .map(|fixture| CorpusCase {
                name: fixture.name.to_owned(),
                job_level: None,
                expert: None,
                recipe_id: None,
                item_id: None,
                crafter_stats: None,
                recipe_level: None,
                baseline_settings: settings(fixture, mode, true),
                adaptive_settings: settings(fixture, mode, true),
                crafter_delineations: 0,
            })
            .collect()
    };
    if let Some(filter) = name_filter {
        corpus_cases.retain(|case| case.name.contains(&filter));
    }
    corpus_cases.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    assert!(
        !real || matches!(mode, Mode::Full),
        "--real exercises the runtime FFI action set and therefore requires --full"
    );
    if let Some(path) = emit_corpus {
        let mut bytes = serde_json::to_vec_pretty(&Corpus {
            version: 1,
            cases: corpus_cases,
        })
        .unwrap();
        bytes.push(b'\n');
        std::fs::write(path, bytes).expect("failed to write corpus");
        return;
    }
    if let Some(reference_path) = reference_results_path {
        assert!(
            corpus_path.is_some(),
            "--compare-reference requires --corpus"
        );
        let report = compare_reference(corpus_cases, &reference_path);
        let mut json = serde_json::to_vec_pretty(&report).unwrap();
        json.push(b'\n');
        if let Some(path) = comparison_output_path {
            std::fs::write(path, json).expect("failed to write comparison report");
        } else {
            print!("{}", String::from_utf8(json).unwrap());
        }
        assert!(
            report.candidate_warm_p95_micros * 10 <= report.reference_warm_p95_micros * 11,
            "warm P95 regressed by more than 10%: reference={}us candidate={}us",
            report.reference_warm_p95_micros,
            report.candidate_warm_p95_micros
        );
        return;
    }

    let mut report = Report {
        mode: match (mode, real, expert_high, corpus_path.is_some()) {
            (_, _, _, true) => "corpus",
            (_, _, true, false) => "expert-high",
            (Mode::Quick, false, false, false) => "quick",
            (Mode::Full, false, false, false) => "full",
            (Mode::Quick, true, false, false) => "real-quick",
            (Mode::Full, true, false, false) => "real-full",
        },
        deadline_millis: DEADLINE_MILLIS.load(Ordering::Relaxed),
        fixtures: corpus_cases
            .into_iter()
            .filter_map(|case| {
                run_case(
                    case.name,
                    case.job_level,
                    case.expert,
                    case.baseline_settings,
                    case.adaptive_settings,
                    case.crafter_delineations,
                    real,
                )
            })
            .collect(),
        totals: ConditionSummary::default(),
    };
    for fixture in &report.fixtures {
        for summary in fixture.conditions.values() {
            merge(&mut report.totals, summary);
        }
    }
    assert_eq!(report.totals.completion_regressions, 0);

    if let Some(path) = output_path {
        let mut bytes = serde_json::to_vec_pretty(&report).unwrap();
        bytes.push(b'\n');
        std::fs::write(path, bytes).expect("failed to write benchmark report");
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!(
            "Donatello benchmark ({}) — {} scenarios, {} strict wins, {} ties, {} losses, {} rejected, {} solve failures, {} completion regressions, {} bounded returns, max quality gap {}",
            report.mode,
            report.totals.scenarios,
            report.totals.strict_wins,
            report.totals.ties,
            report.totals.losses,
            report.totals.rejected_candidates,
            report.totals.solve_failures,
            report.totals.completion_regressions,
            report.totals.non_optimal,
            report.totals.max_quality_bound_gap,
        );
        if real {
            let mut cohorts: BTreeMap<(u8, bool), ConditionSummary> = BTreeMap::new();
            for fixture in &report.fixtures {
                let (Some(level), Some(expert)) = (fixture.job_level, fixture.expert) else {
                    continue;
                };
                let key = (level, expert);
                let cohort = cohorts.entry(key).or_default();
                for summary in fixture.conditions.values() {
                    merge(cohort, summary);
                }
            }
            println!("\nRuntime win/tie/loss rate by real-recipe cohort:");
            for ((level, expert), cohort) in cohorts {
                let denominator = cohort.scenarios.max(1) as f64;
                let win_rate = 100.0 * cohort.strict_wins as f64 / denominator;
                let tie_rate = 100.0 * cohort.ties as f64 / denominator;
                let loss_rate = 100.0 * cohort.losses as f64 / denominator;
                let bar = "█".repeat((win_rate / 2.0).round() as usize);
                println!(
                    "  L{level:>3} {:>7}: win {win_rate:>6.2}% / tie {tie_rate:>6.2}% / loss {loss_rate:>6.2}% ({}/{}/{}) / solver failures {} {bar}",
                    if expert { "expert" } else { "regular" },
                    cohort.strict_wins,
                    cohort.ties,
                    cohort.losses,
                    cohort.solve_failures,
                );
            }
        }
        if summary_only {
            return;
        }
        for fixture in &report.fixtures {
            println!(
                "  {}{}: baseline {} actions, quality {}, {} ms",
                fixture.name,
                match (fixture.job_level, fixture.expert) {
                    (Some(level), Some(expert)) =>
                        format!(" [L{level} {}]", if expert { "expert" } else { "regular" }),
                    _ => String::new(),
                },
                fixture.baseline_actions,
                fixture.baseline_score.quality,
                fixture.elapsed_millis,
            );
            for (condition, summary) in &fixture.conditions {
                println!(
                    "    {condition}: {}/{} strict wins, {} ties, {} losses, {} rejected, {} failures",
                    summary.strict_wins,
                    summary.scenarios,
                    summary.ties,
                    summary.losses,
                    summary.rejected_candidates,
                    summary.solve_failures,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_is_completion_first_and_lexicographic() {
        let complete = PlanScore {
            completes: true,
            quality: 100,
            actions: 10,
            duration: 30,
        };
        let incomplete = PlanScore {
            completes: false,
            quality: 1000,
            actions: 1,
            duration: 3,
        };
        assert!(complete.strictly_better_than(incomplete));
        assert!(!incomplete.strictly_better_than(complete));
        assert_eq!(select_candidate(complete, incomplete), complete);

        let worse_quality = PlanScore {
            quality: 99,
            ..complete
        };
        let better_quality = PlanScore {
            quality: 101,
            ..complete
        };
        assert_eq!(select_candidate(complete, worse_quality), complete);
        assert_eq!(select_candidate(complete, better_quality), better_quality);
    }

    #[test]
    fn zero_step_actions_preserve_the_injected_condition() {
        let fixture = fixtures(Mode::Quick)[0];
        let settings = settings(fixture, Mode::Quick, true);
        let root = SimulationState::new(&settings);
        let score = evaluate(
            &settings,
            root,
            Condition::Excellent,
            &[Action::FinalAppraisal, Action::BasicTouch],
        );
        let expected = root
            .use_action(Action::FinalAppraisal, Condition::Excellent, &settings)
            .unwrap()
            .use_action(Action::BasicTouch, Condition::Excellent, &settings)
            .unwrap();
        assert_eq!(score.quality, expected.quality);
    }

    #[test]
    fn real_corpus_has_exact_bracket_and_expert_coverage() {
        let cases = real_fixtures(Mode::Full);
        for level in (10..=100).step_by(10) {
            assert_eq!(
                cases
                    .iter()
                    .filter(|case| case.job_level == level && !case.expert)
                    .count(),
                10,
                "level bracket ending at {level}"
            );
        }
        for (level, expected) in [(80, 8), (100, 10)] {
            assert_eq!(
                cases
                    .iter()
                    .filter(|case| case.job_level == level && case.expert)
                    .count(),
                expected,
                "level-{level} expert recipes"
            );
        }
        let experts = cases
            .iter()
            .filter(|case| case.expert)
            .collect::<Vec<_>>();
        assert!(
            experts
                .iter()
                .any(|case| case.name == "recipe38202-logged-stats")
        );
    }
}
