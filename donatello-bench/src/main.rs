use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use raphael_data::{CrafterStats, RECIPES, RLVLS, get_game_settings};
use raphael_sim::{Action, ActionMask, Condition, Settings, SimulationState};
use raphael_solver::{AtomicFlag, MacroSolver, SolverSettings};
use serde::Serialize;

const CONDITIONS: [Condition; 9] = [
    Condition::Good,
    Condition::Excellent,
    Condition::Poor,
    Condition::Centered,
    Condition::Sturdy,
    Condition::Pliant,
    Condition::Malleable,
    Condition::Primed,
    Condition::GoodOmen,
];

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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
}

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConditionSummary {
    scenarios: u64,
    strict_wins: u64,
    ties: u64,
    rejected_candidates: u64,
    solve_failures: u64,
    completion_regressions: u64,
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
    fixtures: Vec<FixtureSummary>,
    totals: ConditionSummary,
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
        if action.increases_step_count() {
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

fn baseline_boundaries(settings: &Settings, actions: &[Action]) -> Vec<SimulationState> {
    let mut states = Vec::with_capacity(actions.len());
    let mut state = SimulationState::new(settings);
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
) {
    summary.scenarios += 1;
    summary.solve_micros += solve_duration.as_micros();
    let Ok(candidate) = candidate else {
        summary.solve_failures += 1;
        return;
    };
    if candidate.strictly_better_than(incumbent) {
        summary.strict_wins += 1;
        let selected = candidate;
        assert!(selected.strictly_better_than(incumbent));
        assert!(!incumbent.completes || selected.completes);
        summary.total_quality_gain += u64::from(selected.quality.saturating_sub(incumbent.quality));
        summary.max_quality_gain = summary
            .max_quality_gain
            .max(selected.quality.saturating_sub(incumbent.quality));
        summary.total_action_reduction +=
            i64::from(incumbent.actions) - i64::from(selected.actions);
        summary.total_duration_reduction +=
            i64::from(incumbent.duration) - i64::from(selected.duration);
    } else if candidate == incumbent {
        summary.ties += 1;
    } else {
        summary.rejected_candidates += 1;
    }
    if incumbent.completes && candidate.strictly_better_than(incumbent) && !candidate.completes {
        summary.completion_regressions += 1;
    }
}

fn merge(target: &mut ConditionSummary, source: &ConditionSummary) {
    target.scenarios += source.scenarios;
    target.strict_wins += source.strict_wins;
    target.ties += source.ties;
    target.rejected_candidates += source.rejected_candidates;
    target.solve_failures += source.solve_failures;
    target.completion_regressions += source.completion_regressions;
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
) -> Option<FixtureSummary> {
    let started = Instant::now();
    eprintln!("running fixture: {name}");
    let mut raphael = solver(baseline_settings);
    let Ok(baseline) = raphael.solve() else {
        eprintln!("skipping unsolved baseline fixture: {name}");
        return None;
    };
    let baseline_score = evaluate(
        &baseline_settings,
        SimulationState::new(&baseline_settings),
        Condition::Normal,
        &baseline,
    );
    assert!(
        baseline_score.completes,
        "baseline Raphael plan did not complete"
    );
    let boundaries = baseline_boundaries(&baseline_settings, &baseline);
    let mut donatello = solver(adaptive_settings);
    let mut conditions = BTreeMap::new();

    for condition in CONDITIONS {
        let mut summary = ConditionSummary::default();
        for (index, &root) in boundaries.iter().enumerate() {
            let incumbent = evaluate(&adaptive_settings, root, condition, &baseline[index..]);
            let solve_started = Instant::now();
            let solved = donatello.solve_from_state_with_condition(root, condition);
            let candidate = solved
                .map(|actions| evaluate(&adaptive_settings, root, condition, &actions))
                .map_err(|error| {
                    eprintln!(
                        "solve failure: fixture={}, condition={condition:?}, boundary={index}, root={root:?}, incumbent={incumbent:?}, incumbent_actions={:?}, error={error:?}",
                        name,
                        &baseline[index..],
                    );
                });
            record_scenario(&mut summary, incumbent, candidate, solve_started.elapsed());
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

fn run_fixture(fixture: Fixture, mode: Mode) -> FixtureSummary {
    run_case(
        fixture.name.to_owned(),
        None,
        None,
        settings(fixture, mode, false),
        settings(fixture, mode, true),
    )
    .expect("synthetic baseline Raphael solve failed")
}

fn real_fixtures(mode: Mode) -> Vec<(String, u8, bool, Settings, Settings)> {
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
        let crafter = CrafterStats {
            craftsmanship,
            control,
            cp,
            level: job_level,
            manipulation: job_level >= 65,
            heart_and_soul: false,
            quick_innovation: false,
        };
        for expert in [false, true] {
            let mut candidates = RECIPES
                .values()
                .filter(|recipe| {
                    recipe.max_level_scaling == 0
                        && recipe.is_expert == expert
                        && RLVLS[recipe.recipe_level as usize].job_level == job_level
                        && recipe.req_craftsmanship <= crafter.craftsmanship
                        && recipe.req_control <= crafter.control
                })
                .map(|recipe| {
                    let mut settings = get_game_settings(*recipe, None, crafter, None, None);
                    settings.allowed_actions = action_mask(mode, false);
                    (*recipe, settings)
                })
                .collect::<Vec<_>>();
            candidates.sort_by_key(|(_, settings)| {
                std::cmp::Reverse((settings.max_progress as u32) * (settings.max_quality as u32))
            });
            let mut seen = Vec::new();
            for (recipe, baseline) in candidates {
                let signature = (
                    baseline.max_progress,
                    baseline.max_quality,
                    baseline.max_durability,
                    baseline.base_progress,
                    baseline.base_quality,
                );
                if seen.contains(&signature) {
                    continue;
                }
                seen.push(signature);
                let mut adaptive = baseline;
                adaptive.allowed_actions = action_mask(mode, true);
                result.push((
                    format!(
                        "L{job_level}-{}-item{}-rlvl{}",
                        if expert { "expert" } else { "regular" },
                        recipe.item_id,
                        recipe.recipe_level,
                    ),
                    job_level,
                    expert,
                    baseline,
                    adaptive,
                ));
                if seen.len() == 10 {
                    break;
                }
            }
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
    let mut summary_only = false;
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--quick" => mode = Mode::Quick,
            "--full" => mode = Mode::Full,
            "--json" => json = true,
            "--real" => real = true,
            "--summary" => summary_only = true,
            "--help" | "-h" => {
                println!("Usage: donatello-bench [--quick|--full] [--real] [--summary] [--json]");
                return;
            }
            _ => panic!("unknown argument {argument}"),
        }
    }

    let mut report = Report {
        mode: match (mode, real) {
            (Mode::Quick, false) => "quick",
            (Mode::Full, false) => "full",
            (Mode::Quick, true) => "real-quick",
            (Mode::Full, true) => "real-full",
        },
        fixtures: if real {
            real_fixtures(mode)
                .into_iter()
                .filter_map(|(name, level, expert, baseline, adaptive)| {
                    run_case(name, Some(level), Some(expert), baseline, adaptive)
                })
                .collect()
        } else {
            fixtures(mode)
                .iter()
                .copied()
                .map(|fixture| run_fixture(fixture, mode))
                .collect()
        },
        totals: ConditionSummary::default(),
    };
    for fixture in &report.fixtures {
        for summary in fixture.conditions.values() {
            merge(&mut report.totals, summary);
        }
    }
    assert_eq!(report.totals.completion_regressions, 0);

    if json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!(
            "Donatello benchmark ({}) — {} scenarios, {} strict wins, {} ties, {} rejected, {} solve failures, {} completion regressions",
            report.mode,
            report.totals.scenarios,
            report.totals.strict_wins,
            report.totals.ties,
            report.totals.rejected_candidates,
            report.totals.solve_failures,
            report.totals.completion_regressions,
        );
        if real {
            let mut cohorts: BTreeMap<(u8, bool), ConditionSummary> = BTreeMap::new();
            for fixture in &report.fixtures {
                let key = (fixture.job_level.unwrap(), fixture.expert.unwrap());
                let cohort = cohorts.entry(key).or_default();
                for summary in fixture.conditions.values() {
                    merge(cohort, summary);
                }
            }
            println!("\nStrict-win rate by real-recipe cohort:");
            for ((level, expert), cohort) in cohorts {
                let rate = 100.0 * cohort.strict_wins as f64 / cohort.scenarios as f64;
                let bar = "█".repeat((rate / 2.0).round() as usize);
                println!(
                    "  L{level:>3} {:>7}: {:>6.2}% ({}/{}) {bar}",
                    if expert { "expert" } else { "regular" },
                    rate,
                    cohort.strict_wins,
                    cohort.scenarios,
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
                    "    {condition}: {}/{} strict wins, {} ties, {} rejected, {} failures",
                    summary.strict_wins,
                    summary.scenarios,
                    summary.ties,
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
            quality: 0,
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
}
