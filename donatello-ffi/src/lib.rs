use std::collections::VecDeque;
use std::ffi::{CString, c_char};
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use raphael_sim::{
    Action, ActionMask, Combo, Condition, Effects, Settings, SimulationState, SpecialQualityState,
};
use raphael_solver::{AtomicFlag, MacroSolver, SolverSettings};
use serde::{Deserialize, Serialize};

const ABI_VERSION: u32 = 5;
const DEFAULT_CACHE_BUDGET: usize = 512 * 1024 * 1024;

type CachedSolver = Arc<Mutex<MacroSolver<'static>>>;
static SOLVER_CACHE: OnceLock<Mutex<SolverCache>> = OnceLock::new();
type SolutionCacheKey = (SolverSettings, SimulationState, Condition, bool);
static SOLUTION_CACHE: OnceLock<Mutex<SolutionCache>> = OnceLock::new();
static CACHE_BUDGET: AtomicUsize = AtomicUsize::new(DEFAULT_CACHE_BUDGET);

#[derive(Default)]
struct SolutionCache {
    entries: VecDeque<(SolutionCacheKey, Vec<Action>)>,
    retained_bytes: usize,
}

#[derive(Default)]
struct SolverCache {
    entries: VecDeque<(SolverSettings, CachedSolver, usize)>,
    retained_bytes: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CraftSolveRequest {
    abi_version: u32,
    max_cp: u16,
    max_durability: u16,
    max_progress: u16,
    max_quality: u16,
    base_progress: u16,
    base_quality: u16,
    job_level: u8,
    manipulation: bool,
    specialist: bool,
    backload_progress: bool,
    #[serde(default)]
    objective: u8,
    #[serde(default)]
    minimize_steps: bool,
    #[serde(default)]
    stellar_steady_hand_charges: u8,
    #[serde(default)]
    incumbent_action_ids: Vec<u32>,
    root: RootState,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RootState {
    cp: u16,
    durability: u16,
    progress: u16,
    quality: u16,
    inner_quiet: u8,
    waste_not: u8,
    manipulation: u8,
    innovation: u8,
    veneration: u8,
    great_strides: u8,
    muscle_memory: u8,
    final_appraisal: u8,
    careful_observation_charges: u8,
    combo: u8,
    heart_and_soul_active: bool,
    heart_and_soul_available: bool,
    quick_innovation_available: bool,
    trained_perfection_active: bool,
    trained_perfection_available: bool,
    #[serde(default)]
    stellar_steady_hand_charges: u8,
    #[serde(default)]
    stellar_steady_hand: u8,
    #[serde(default)]
    splendor_cosmic: bool,
    expedience: bool,
    condition: u8,
    crafter_delineations: u8,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SolveResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    action_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn condition(value: u8) -> Result<Condition, String> {
    match value {
        0 => Ok(Condition::Normal),
        1 => Ok(Condition::Good),
        2 => Ok(Condition::Excellent),
        3 => Ok(Condition::Poor),
        4 => Ok(Condition::Centered),
        5 => Ok(Condition::Sturdy),
        6 => Ok(Condition::Pliant),
        7 => Ok(Condition::Malleable),
        8 => Ok(Condition::Primed),
        9 => Ok(Condition::GoodOmen),
        10 => Ok(Condition::Robust),
        _ => Err(format!("unsupported condition {value}")),
    }
}

fn build_state(root: &RootState, backload_progress: bool) -> Result<SimulationState, String> {
    let combo = match root.combo {
        0 => Combo::None,
        1 => Combo::BasicTouch,
        2 => Combo::StandardTouch,
        3 => Combo::SynthesisBegin,
        value => return Err(format!("unsupported combo {value}")),
    };
    let effects = Effects::new()
        .with_special_quality_state(if backload_progress && root.progress > 0 {
            SpecialQualityState::Forbidden
        } else {
            SpecialQualityState::Normal
        })
        .with_inner_quiet(root.inner_quiet)
        .with_waste_not(root.waste_not)
        .with_manipulation(root.manipulation)
        .with_innovation(root.innovation)
        .with_veneration(root.veneration)
        .with_great_strides(root.great_strides)
        .with_muscle_memory(root.muscle_memory)
        .with_final_appraisal(root.final_appraisal)
        .with_careful_observation_charges(root.careful_observation_charges)
        .with_combo(combo)
        .with_heart_and_soul_active(root.heart_and_soul_active)
        .with_heart_and_soul_available(root.heart_and_soul_available)
        .with_quick_innovation_available(root.quick_innovation_available)
        .with_trained_perfection_active(root.trained_perfection_active)
        .with_trained_perfection_available(root.trained_perfection_available)
        .with_stellar_steady_hand_charges(root.stellar_steady_hand_charges)
        .with_stellar_steady_hand(root.stellar_steady_hand.min(3))
        .with_splendor_cosmic(root.splendor_cosmic)
        .with_expedience(root.expedience);
    let effects = effects.with_crafter_delineations(root.crafter_delineations.min(2));
    Ok(SimulationState {
        cp: root.cp,
        durability: root.durability,
        progress: root.progress,
        quality: root.quality,
        unreliable_quality: 0,
        effects,
    })
}

fn solve(request: CraftSolveRequest, interrupt: AtomicFlag) -> Result<Vec<u32>, String> {
    if request.abi_version != ABI_VERSION {
        return Err(format!("unsupported ABI version {}", request.abi_version));
    }
    if interrupt.is_set() {
        return Err(String::from("Interrupted"));
    }
    if request.objective > 1 {
        return Err(format!("unsupported objective {}", request.objective));
    }
    let mut allowed_actions = if request.objective == 1 {
        progress_only_actions()
    } else {
        ActionMask::regular().add(Action::FinalAppraisal)
    };
    if request.stellar_steady_hand_charges > 0 {
        allowed_actions = allowed_actions
            .add(Action::StellarSteadyHand)
            .add(Action::RapidSynthesis);
        if request.objective == 0 {
            allowed_actions = allowed_actions
                .add(Action::HastyTouch)
                .add(Action::DaringTouch);
        }
    } else {
        allowed_actions = allowed_actions
            .remove(Action::StellarSteadyHand)
            .remove(Action::RapidSynthesis)
            .remove(Action::HastyTouch)
            .remove(Action::DaringTouch);
    }
    if !request.manipulation {
        allowed_actions = allowed_actions.remove(Action::Manipulation);
    }
    if request.specialist {
        allowed_actions = allowed_actions
            .add(Action::HeartAndSoul)
            .add(Action::QuickInnovation);
    }
    let settings = Settings {
        max_cp: request.max_cp,
        max_durability: request.max_durability,
        max_progress: request.max_progress,
        max_quality: request.max_quality,
        base_progress: request.base_progress,
        base_quality: request.base_quality,
        job_level: request.job_level,
        allowed_actions,
        adversarial: false,
        backload_progress: request.backload_progress,
        stellar_steady_hand_charges: request.stellar_steady_hand_charges.min(3),
    };
    let state = build_state(&request.root, request.backload_progress)?;
    let current_condition = condition(request.root.condition)?;
    let incumbent = request
        .incumbent_action_ids
        .into_iter()
        .map(|action_id| {
            Action::from_action_id(action_id)
                .ok_or_else(|| format!("unsupported incumbent action {action_id}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let solver_settings = SolverSettings {
        simulator_settings: settings,
        allow_non_max_quality_solutions: true,
    };
    let cache_key = (
        solver_settings,
        state,
        current_condition,
        request.minimize_steps,
    );
    if let Some(actions) = cached_solution(&cache_key) {
        return Ok(actions.into_iter().map(Action::action_id).collect());
    }
    let cached_solver = cached_solver(solver_settings);
    let mut solver = cached_solver
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(actions) = cached_solution(&cache_key) {
        return Ok(actions.into_iter().map(Action::action_id).collect());
    }
    solver.set_interrupt_signal(interrupt);
    let result = solver
        .solve_from_state_with_condition_and_incumbent_objective(
            state,
            current_condition,
            &incumbent,
            request.minimize_steps,
        )
        .map_err(|error| format!("{error:?}"));
    let retained_bytes = solver.estimated_retained_bytes();
    drop(solver);
    update_solver_weight(&cached_solver, retained_bytes);
    let actions = result?;
    cache_solution(cache_key, &actions);
    Ok(actions.into_iter().map(Action::action_id).collect())
}

fn progress_only_actions() -> ActionMask {
    ActionMask::none()
        .add(Action::BasicSynthesis)
        .add(Action::MasterMend)
        .add(Action::WasteNot)
        .add(Action::Veneration)
        .add(Action::WasteNot2)
        .add(Action::MuscleMemory)
        .add(Action::CarefulSynthesis)
        .add(Action::Manipulation)
        .add(Action::Groundwork)
        .add(Action::IntensiveSynthesis)
        .add(Action::PrudentSynthesis)
        .add(Action::ImmaculateMend)
        .add(Action::TrainedPerfection)
        .add(Action::FinalAppraisal)
}

fn cached_solution(key: &SolutionCacheKey) -> Option<Vec<Action>> {
    let mut cache = SOLUTION_CACHE
        .get_or_init(|| Mutex::new(SolutionCache::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let index = cache
        .entries
        .iter()
        .position(|(cached_key, _)| cached_key == key)?;
    let entry = cache.entries.remove(index).unwrap();
    let actions = entry.1.clone();
    cache.entries.push_back(entry);
    Some(actions)
}

fn cache_solution(key: SolutionCacheKey, actions: &[Action]) {
    let mut cache = SOLUTION_CACHE
        .get_or_init(|| Mutex::new(SolutionCache::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(index) = cache
        .entries
        .iter()
        .position(|(cached_key, _)| *cached_key == key)
    {
        if let Some((_, old_actions)) = cache.entries.remove(index) {
            cache.retained_bytes = cache
                .retained_bytes
                .saturating_sub(entry_size(&old_actions));
        }
    }
    let actions = actions.to_vec();
    let entry_bytes = entry_size(&actions);
    let budget = CACHE_BUDGET.load(Ordering::Relaxed) / 16;
    if entry_bytes > budget {
        return;
    }
    while cache.retained_bytes.saturating_add(entry_bytes) > budget {
        let Some((_, old_actions)) = cache.entries.pop_front() else {
            break;
        };
        cache.retained_bytes = cache
            .retained_bytes
            .saturating_sub(entry_size(&old_actions));
    }
    cache.retained_bytes += entry_bytes;
    cache.entries.push_back((key, actions));
}

fn entry_size(actions: &Vec<Action>) -> usize {
    size_of::<SolutionCacheKey>() + size_of::<Action>() * actions.capacity()
}

fn trim_solution_cache() {
    let mut cache = SOLUTION_CACHE
        .get_or_init(|| Mutex::new(SolutionCache::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let budget = CACHE_BUDGET.load(Ordering::Relaxed) / 16;
    while cache.retained_bytes > budget {
        let Some((_, actions)) = cache.entries.pop_front() else {
            break;
        };
        cache.retained_bytes = cache.retained_bytes.saturating_sub(entry_size(&actions));
    }
}

fn cached_solver(settings: SolverSettings) -> CachedSolver {
    let cache = SOLVER_CACHE.get_or_init(|| Mutex::new(SolverCache::default()));
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(index) = cache
        .entries
        .iter()
        .position(|(key, _, _)| *key == settings)
    {
        let entry = cache.entries.remove(index).unwrap();
        let solver = Arc::clone(&entry.1);
        cache.entries.push_back(entry);
        return solver;
    }
    let solver = Arc::new(Mutex::new(MacroSolver::new(
        settings,
        Box::new(|_| {}),
        Box::new(|_| {}),
        AtomicFlag::new(),
    )));
    let initial_bytes = size_of::<MacroSolver<'static>>();
    cache.retained_bytes += initial_bytes;
    cache
        .entries
        .push_back((settings, Arc::clone(&solver), initial_bytes));
    solver
}

fn update_solver_weight(solver: &CachedSolver, retained_bytes: usize) {
    let mut cache = SOLVER_CACHE
        .get_or_init(|| Mutex::new(SolverCache::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(index) = cache
        .entries
        .iter()
        .position(|(_, candidate, _)| Arc::ptr_eq(candidate, solver))
    else {
        return;
    };
    let mut entry = cache.entries.remove(index).unwrap();
    cache.retained_bytes = cache.retained_bytes.saturating_sub(entry.2);
    entry.2 = retained_bytes;
    let budget = CACHE_BUDGET.load(Ordering::Relaxed) * 15 / 16;
    if retained_bytes > budget {
        return;
    }
    cache.retained_bytes += retained_bytes;
    cache.entries.push_back(entry);
    while cache.retained_bytes > budget {
        let Some((settings, candidate, bytes)) = cache.entries.pop_front() else {
            break;
        };
        if Arc::strong_count(&candidate) > 1 {
            cache.entries.push_back((settings, candidate, bytes));
            break;
        }
        cache.retained_bytes = cache.retained_bytes.saturating_sub(bytes);
    }
}

fn trim_solver_cache() {
    let Some(cache) = SOLVER_CACHE.get() else {
        return;
    };
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let budget = CACHE_BUDGET.load(Ordering::Relaxed) * 15 / 16;
    let mut checked = 0;
    while cache.retained_bytes > budget && checked < cache.entries.len() {
        let Some((settings, solver, bytes)) = cache.entries.pop_front() else {
            break;
        };
        if Arc::strong_count(&solver) == 1 {
            cache.retained_bytes = cache.retained_bytes.saturating_sub(bytes);
        } else {
            cache.entries.push_back((settings, solver, bytes));
            checked += 1;
        }
    }
}

fn response_json(result: Result<Vec<u32>, String>) -> CString {
    let response = match result {
        Ok(action_ids) => SolveResponse {
            ok: true,
            action_ids: Some(action_ids),
            error: None,
        },
        Err(error) => SolveResponse {
            ok: false,
            action_ids: None,
            error: Some(error),
        },
    };
    let json = serde_json::to_string(&response).unwrap_or_else(|_| {
        String::from(r#"{"ok":false,"error":"response serialization failed"}"#)
    });
    CString::new(json).unwrap()
}

fn solve_json_with_flag(data: &[u8], interrupt: AtomicFlag) -> String {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let request = serde_json::from_slice(data).map_err(|error| error.to_string())?;
        solve(request, interrupt)
    }))
    .unwrap_or_else(|_| Err(String::from("native solver panic")));
    response_json(result).into_string().unwrap()
}

pub fn solve_json(data: &[u8]) -> String {
    solve_json_with_flag(data, AtomicFlag::new())
}

fn gathering_response_json(data: &[u8]) -> CString {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let request: gathering_solvers::SolveRequest =
            serde_json::from_slice(data).map_err(|error| error.to_string())?;
        gathering_solvers::solve(&request)
    }))
    .unwrap_or_else(|_| Err(String::from("native gathering solver panic")));
    let json = match result {
        Ok(decision) => serde_json::json!({ "ok": true, "decision": decision }),
        Err(error) => serde_json::json!({ "ok": false, "error": error }),
    };
    CString::new(json.to_string()).unwrap()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn donatello_gathering_solve_json(
    data: *const u8,
    len: usize,
) -> *mut c_char {
    if data.is_null() {
        return CString::new(r#"{"ok":false,"error":"null request pointer"}"#)
            .unwrap()
            .into_raw();
    }
    let bytes = unsafe { slice::from_raw_parts(data, len) };
    gathering_response_json(bytes).into_raw()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn donatello_solve_json(data: *const u8, len: usize) -> *mut c_char {
    if data.is_null() {
        return response_json(Err(String::from("null request pointer"))).into_raw();
    }
    let bytes = unsafe { slice::from_raw_parts(data, len) };
    CString::new(solve_json(bytes)).unwrap().into_raw()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn donatello_solve_json_interruptible(
    data: *const u8,
    len: usize,
    interrupt: *mut AtomicFlag,
) -> *mut c_char {
    if data.is_null() || interrupt.is_null() {
        return response_json(Err(String::from("null request or interrupt pointer"))).into_raw();
    }
    let bytes = unsafe { slice::from_raw_parts(data, len) };
    let flag = unsafe { &*interrupt }.clone();
    CString::new(solve_json_with_flag(bytes, flag))
        .unwrap()
        .into_raw()
}

#[unsafe(no_mangle)]
pub extern "C" fn donatello_interrupt_create() -> *mut AtomicFlag {
    Box::into_raw(Box::new(AtomicFlag::new()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn donatello_interrupt_set(interrupt: *mut AtomicFlag) {
    if let Some(interrupt) = unsafe { interrupt.as_ref() } {
        interrupt.set();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn donatello_interrupt_free(interrupt: *mut AtomicFlag) {
    if !interrupt.is_null() {
        drop(unsafe { Box::from_raw(interrupt) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn donatello_string_free(value: *mut c_char) {
    if !value.is_null() {
        drop(unsafe { CString::from_raw(value) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn donatello_abi_version() -> u32 {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn donatello_cache_set_budget_bytes(bytes: usize) {
    CACHE_BUDGET.store(bytes, Ordering::Relaxed);
    trim_solution_cache();
    trim_solver_cache();
}

#[unsafe(no_mangle)]
pub extern "C" fn donatello_cache_clear() {
    if let Some(cache) = SOLUTION_CACHE.get() {
        *cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = SolutionCache::default();
    }
    if let Some(cache) = SOLVER_CACHE.get() {
        *cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = SolverCache::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(condition: u8) -> Vec<u8> {
        format!(
            r#"{{"abiVersion":5,"maxCp":500,"maxDurability":40,"maxProgress":500,"maxQuality":500,"baseProgress":100,"baseQuality":100,"jobLevel":100,"manipulation":true,"specialist":false,"backloadProgress":false,"objective":0,"minimizeSteps":false,"stellarSteadyHandCharges":0,"root":{{"cp":500,"durability":40,"progress":0,"quality":0,"innerQuiet":0,"wasteNot":0,"manipulation":0,"innovation":0,"veneration":0,"greatStrides":0,"muscleMemory":0,"finalAppraisal":0,"carefulObservationCharges":0,"combo":3,"heartAndSoulActive":false,"heartAndSoulAvailable":false,"quickInnovationAvailable":false,"trainedPerfectionActive":false,"trainedPerfectionAvailable":true,"stellarSteadyHandCharges":0,"stellarSteadyHand":0,"expedience":false,"condition":{condition},"crafterDelineations":0}}}}"#
        )
        .into_bytes()
    }

    #[test]
    fn rejects_unknown_condition_without_normalizing_it() {
        let response = solve_json(&request(11));
        assert!(response.contains(r#""ok":false"#));
        assert!(response.contains("unsupported condition 11"));
    }

    #[test]
    fn accepts_robust_condition() {
        assert_eq!(condition(10).unwrap(), Condition::Robust);
    }

    #[test]
    fn pre_set_interrupt_is_reported_as_solver_failure() {
        let interrupt = AtomicFlag::new();
        interrupt.set();
        let response = solve_json_with_flag(&request(0), interrupt);
        assert!(response.contains(r#""ok":false"#));
        assert!(response.contains("Interrupted"));
    }

    #[test]
    fn initial_root_returns_a_solution_without_the_cli() {
        let response = solve_json(&request(0));
        assert!(response.contains(r#""ok":true"#), "{response}");
        assert!(response.contains(r#""actionIds":["#), "{response}");
    }

    #[test]
    fn progress_only_request_returns_only_progress_mask_actions() {
        let mut request: serde_json::Value = serde_json::from_slice(&request(0)).unwrap();
        request["objective"] = 1.into();
        request["maxQuality"] = 0.into();

        let response = solve_json(&serde_json::to_vec(&request).unwrap());
        let response: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(response["ok"], true, "{response}");
        let actions = response["actionIds"].as_array().unwrap();
        assert!(!actions.is_empty());
        for action_id in actions {
            let action = Action::from_action_id(action_id.as_u64().unwrap() as u32).unwrap();
            assert!(progress_only_actions().has(action), "unexpected {action:?}");
        }
    }

    #[test]
    fn abi_version_is_stable() {
        assert_eq!(donatello_abi_version(), 5);
    }

    #[test]
    fn camel_case_v2_request_preserves_shared_specialist_resource() {
        let mut request: CraftSolveRequest = serde_json::from_slice(&request(0)).unwrap();
        request.root.crafter_delineations = 1;
        request.root.heart_and_soul_available = true;
        request.root.quick_innovation_available = true;
        let state = build_state(&request.root, false).unwrap();
        assert_eq!(state.effects.crafter_delineations(), 1);
    }

    #[test]
    fn progressed_backload_root_forbids_further_quality_actions() {
        let mut request: CraftSolveRequest = serde_json::from_slice(&request(0)).unwrap();
        request.root.progress = 1;
        let state = build_state(&request.root, true).unwrap();
        assert!(!state.effects.quality_actions_allowed());
    }

    #[test]
    fn arbitrary_root_preserves_every_represented_resource_and_effect() {
        let mut request: CraftSolveRequest = serde_json::from_slice(&request(0)).unwrap();
        request.root.cp = 237;
        request.root.durability = 17;
        request.root.progress = 123;
        request.root.quality = 456;
        request.root.inner_quiet = 8;
        request.root.waste_not = 2;
        request.root.manipulation = 3;
        request.root.innovation = 4;
        request.root.veneration = 5;
        request.root.great_strides = 2;
        request.root.muscle_memory = 1;
        request.root.final_appraisal = 7;
        request.root.careful_observation_charges = 2;
        request.root.combo = 1;
        request.root.heart_and_soul_active = true;
        request.root.heart_and_soul_available = false;
        request.root.quick_innovation_available = true;
        request.root.trained_perfection_active = true;
        request.root.trained_perfection_available = false;
        request.root.expedience = true;
        request.root.stellar_steady_hand_charges = 2;
        request.root.stellar_steady_hand = 3;
        request.root.splendor_cosmic = true;

        let state = build_state(&request.root, false).unwrap();
        assert_eq!(state.cp, 237);
        assert_eq!(state.durability, 17);
        assert_eq!(state.progress, 123);
        assert_eq!(state.quality, 456);
        assert_eq!(state.effects.inner_quiet(), 8);
        assert_eq!(state.effects.waste_not(), 2);
        assert_eq!(state.effects.manipulation(), 3);
        assert_eq!(state.effects.innovation(), 4);
        assert_eq!(state.effects.veneration(), 5);
        assert_eq!(state.effects.great_strides(), 2);
        assert_eq!(state.effects.muscle_memory(), 1);
        assert_eq!(state.effects.final_appraisal(), 7);
        assert_eq!(state.effects.careful_observation_charges(), 2);
        assert_eq!(state.effects.combo(), Combo::BasicTouch);
        assert!(state.effects.heart_and_soul_active());
        assert!(!state.effects.heart_and_soul_available());
        assert!(state.effects.quick_innovation_available());
        assert!(state.effects.trained_perfection_active());
        assert!(!state.effects.trained_perfection_available());
        assert!(state.effects.expedience());
        assert_eq!(state.effects.stellar_steady_hand_charges(), 2);
        assert_eq!(state.effects.stellar_steady_hand(), 3);
        assert!(state.effects.splendor_cosmic());
    }

    #[test]
    fn progress_only_action_set_excludes_every_quality_action() {
        let actions = progress_only_actions();
        for action in [
            Action::BasicTouch,
            Action::HastyTouch,
            Action::DelicateSynthesis,
            Action::ByregotsBlessing,
            Action::TrainedEye,
        ] {
            assert!(!actions.has(action), "{action:?} must be excluded");
        }
        assert!(actions.has(Action::BasicSynthesis));
        assert!(actions.has(Action::Groundwork));
    }

    #[test]
    fn identical_recipe_settings_reuse_prepared_solver() {
        let settings = SolverSettings {
            simulator_settings: Settings {
                max_cp: 10,
                max_durability: 10,
                max_progress: 10,
                max_quality: 10,
                base_progress: 10,
                base_quality: 10,
                job_level: 1,
                allowed_actions: ActionMask::none(),
                adversarial: false,
                backload_progress: false,
                stellar_steady_hand_charges: 0,
            },
            allow_non_max_quality_solutions: true,
        };
        assert!(Arc::ptr_eq(
            &cached_solver(settings),
            &cached_solver(settings)
        ));
    }

    #[test]
    fn exact_result_key_distinguishes_condition_state_and_objective() {
        let settings = SolverSettings {
            simulator_settings: Settings {
                max_cp: 10,
                max_durability: 10,
                max_progress: 10,
                max_quality: 10,
                base_progress: 10,
                base_quality: 10,
                job_level: 1,
                allowed_actions: ActionMask::none(),
                adversarial: false,
                backload_progress: false,
                stellar_steady_hand_charges: 0,
            },
            allow_non_max_quality_solutions: true,
        };
        let state = SimulationState::new(&settings.simulator_settings);
        let key = (settings, state, Condition::Normal, false);
        assert_ne!(key, (settings, state, Condition::Good, false));
        assert_ne!(
            key,
            (
                settings,
                SimulationState { cp: 9, ..state },
                Condition::Normal,
                false
            )
        );
        assert_ne!(key, (settings, state, Condition::Normal, true));
    }
}
