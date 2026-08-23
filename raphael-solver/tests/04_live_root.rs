use raphael_sim::{Action, ActionMask, Condition, Settings, SimulationState};
use raphael_solver::{AtomicFlag, MacroSolver, SolverSettings};

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

fn settings(allowed_actions: ActionMask) -> Settings {
    Settings {
        max_cp: 50,
        max_durability: 50,
        max_progress: 400,
        max_quality: 0,
        base_progress: 100,
        base_quality: 100,
        job_level: 100,
        allowed_actions,
        adversarial: false,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    }
}

#[test]
fn pristine_normal_live_root_matches_ordinary_solver() {
    let settings = settings(ActionMask::none().add(Action::BasicSynthesis));
    let mut ordinary = solver(settings);
    let mut live = solver(settings);
    assert_eq!(
        ordinary.solve().unwrap(),
        live.solve_from_state(SimulationState::new(&settings))
            .unwrap()
    );
}

#[test]
fn good_replan_uses_naturally_legal_intensive_synthesis() {
    let settings = settings(
        ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::IntensiveSynthesis),
    );
    let root = SimulationState::new(&settings);
    let normal = solver(settings).solve_from_state(root).unwrap();
    let good = solver(settings)
        .solve_from_state_with_condition(root, Condition::Good)
        .unwrap();
    assert_eq!(good, [Action::IntensiveSynthesis]);
    assert!(good.len() < normal.len());
}

#[test]
fn arbitrary_root_preserves_resources_and_progress() {
    let settings = settings(ActionMask::none().add(Action::BasicSynthesis));
    let root = SimulationState {
        cp: 7,
        durability: 30,
        progress: 280,
        ..SimulationState::new(&settings)
    };
    let actions = solver(settings).solve_from_state(root).unwrap();
    let final_state = actions.into_iter().fold(root, |state, action| {
        state
            .use_action(action, Condition::Normal, &settings)
            .unwrap()
    });
    assert_eq!(final_state.cp, 7);
    assert_eq!(final_state.durability, 20);
    assert!(final_state.progress >= settings.max_progress);
}

#[test]
fn completing_incumbent_preserves_the_optimal_live_result() {
    let settings = settings(
        ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::CarefulSynthesis),
    );
    let root = SimulationState::new(&settings);
    let mut solver = solver(settings);
    let optimum = solver
        .solve_from_state_with_condition(root, Condition::Good)
        .unwrap();
    let seeded = solver
        .solve_from_state_with_condition_and_incumbent(
            root,
            Condition::Good,
            &[
                Action::BasicSynthesis,
                Action::BasicSynthesis,
                Action::BasicSynthesis,
                Action::BasicSynthesis,
            ],
        )
        .unwrap();
    assert_eq!(seeded, optimum);
}

#[test]
fn invalid_incumbent_is_ignored() {
    let settings = settings(ActionMask::none().add(Action::BasicSynthesis));
    let root = SimulationState::new(&settings);
    let result = solver(settings)
        .solve_from_state_with_condition_and_incumbent(root, Condition::Good, &[Action::BasicTouch])
        .unwrap();
    assert_eq!(result, [Action::BasicSynthesis; 4]);
}

#[test]
fn quality_only_replan_returns_a_max_quality_incumbent_without_search() {
    let mut settings = settings(
        ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::BasicTouch),
    );
    settings.max_progress = 200;
    settings.max_quality = 100;
    let root = SimulationState::new(&settings);
    let incumbent = [
        Action::BasicTouch,
        Action::BasicSynthesis,
        Action::BasicSynthesis,
    ];
    let mut solver = solver(settings);
    let result = solver
        .solve_from_state_with_condition_and_incumbent_objective(
            root,
            Condition::Normal,
            &incumbent,
            false,
        )
        .unwrap();
    assert_eq!(result, incumbent);
    assert_eq!(solver.runtime_stats().search_queue_stats.processed_nodes, 0);
}

#[test]
fn expert_preference_moves_quality_before_progress_on_an_exact_tie() {
    let settings = Settings {
        max_cp: 18,
        max_durability: 30,
        max_progress: 200,
        max_quality: 100,
        base_progress: 100,
        base_quality: 100,
        job_level: 5,
        allowed_actions: ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::BasicTouch),
        adversarial: false,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let root = SimulationState::new(&settings);
    let progress_first = [
        Action::BasicSynthesis,
        Action::BasicTouch,
        Action::BasicSynthesis,
    ];
    let outcome = solver(settings)
        .solve_from_state_with_condition_and_incumbent_anytime_preference(
            root,
            Condition::Centered,
            &progress_first,
            false,
            true,
        )
        .unwrap();

    assert!(outcome.optimal);
    assert_eq!(
        outcome.actions,
        [
            Action::BasicTouch,
            Action::BasicSynthesis,
            Action::BasicSynthesis,
        ],
        "{:?}",
        outcome.stats,
    );
}

#[test]
fn every_supported_prefix_rejoins_normal_search_without_bound_failure() {
    let mut settings = settings(
        ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::BasicTouch)
            .add(Action::Innovation),
    );
    settings.max_progress = 200;
    settings.max_quality = 100;
    for condition in [
        Condition::Good,
        Condition::Excellent,
        Condition::Poor,
        Condition::Centered,
        Condition::Sturdy,
        Condition::Pliant,
        Condition::Malleable,
        Condition::Primed,
        Condition::GoodOmen,
    ] {
        let root = SimulationState::new(&settings);
        let actions = solver(settings)
            .solve_from_state_with_condition(root, condition)
            .unwrap_or_else(|error| panic!("{condition:?} prefix failed: {error:?}"));
        assert!(
            !actions.is_empty(),
            "{condition:?} prefix returned no actions"
        );
    }
}

#[test]
fn primed_derived_effect_duration_degrades_bounds_safely() {
    let mut settings = settings(
        ActionMask::none()
            .add(Action::BasicSynthesis)
            .add(Action::BasicTouch),
    );
    settings.max_progress = 200;
    settings.max_quality = 100;
    let root = SimulationState {
        effects: SimulationState::new(&settings).effects.with_innovation(6),
        ..SimulationState::new(&settings)
    };
    assert!(!solver(settings).solve_from_state(root).unwrap().is_empty());
}

#[test]
fn sturdy_prefix_preserves_a_known_completing_suffix() {
    let settings = Settings {
        max_cp: 220,
        max_durability: 35,
        max_progress: 520,
        max_quality: 1100,
        base_progress: 125,
        base_quality: 105,
        job_level: 100,
        allowed_actions: ActionMask::none()
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
            .add(Action::FinalAppraisal),
        adversarial: false,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let root = SimulationState {
        cp: 60,
        durability: 35,
        progress: 0,
        quality: 656,
        unreliable_quality: 0,
        effects: SimulationState::new(&settings)
            .effects
            .with_inner_quiet(3)
            .with_combo(raphael_sim::Combo::None)
            .with_trained_perfection_available(false),
    };
    let actions = solver(settings)
        .solve_from_state_with_condition(root, Condition::Sturdy)
        .expect("known completing Sturdy suffix must survive pruning");
    let mut state = root;
    let mut condition = Condition::Sturdy;
    for action in actions {
        state = state.use_action(action, condition, &settings).unwrap();
        if action.advances_condition() {
            condition = condition.deterministic_successor();
        }
    }
    assert!(state.is_final(&settings));
}

#[test]
fn late_sturdy_root_with_no_bound_continuation_preserves_incumbent() {
    let settings = Settings {
        max_cp: 738,
        max_durability: 20,
        max_progress: 4700,
        max_quality: 14900,
        base_progress: 290,
        base_quality: 301,
        job_level: 100,
        allowed_actions: ActionMask::all()
            .remove(Action::TrainedEye)
            .remove(Action::HeartAndSoul)
            .remove(Action::QuickInnovation)
            .remove(Action::RapidSynthesis)
            .remove(Action::HastyTouch)
            .remove(Action::DaringTouch),
        adversarial: false,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let root = SimulationState {
        cp: 67,
        durability: 15,
        progress: 4219,
        quality: 8100,
        effects: SimulationState::new(&settings)
            .effects
            .with_inner_quiet(10)
            .with_innovation(2),
        ..SimulationState::new(&settings)
    };
    let incumbent = [
        Action::GreatStrides,
        Action::ByregotsBlessing,
        Action::CarefulSynthesis,
    ];
    let result = solver(settings)
        .solve_from_state_with_condition_and_incumbent(root, Condition::Sturdy, &incumbent)
        .expect("a terminal quality-bound branch must not become an internal solver error");

    let mut state = root;
    let mut condition = Condition::Sturdy;
    for action in result {
        state = state.use_action(action, condition, &settings).unwrap();
        if action.advances_condition() {
            condition = condition.deterministic_successor();
        }
    }
    assert!(state.is_final(&settings));
    assert!(state.quality >= 12615);
}

#[test]
fn finish_oracle_cannot_discard_a_valid_specialist_incumbent() {
    let settings = Settings {
        max_cp: 630,
        max_durability: 60,
        max_progress: 8500,
        max_quality: 19500,
        base_progress: 290,
        base_quality: 301,
        job_level: 100,
        allowed_actions: ActionMask::all()
            .remove(Action::TrainedEye)
            .remove(Action::RapidSynthesis)
            .remove(Action::HastyTouch)
            .remove(Action::DaringTouch),
        adversarial: false,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let root = SimulationState {
        cp: 9,
        durability: 5,
        progress: 6785,
        quality: 13405,
        effects: SimulationState::new(&settings)
            .effects
            .with_veneration(1)
            .with_heart_and_soul_active(true),
        ..SimulationState::new(&settings)
    };
    let incumbent = [Action::IntensiveSynthesis];
    let result = solver(settings)
        .solve_from_state_with_condition_and_incumbent(root, Condition::Normal, &incumbent)
        .expect("a validated incumbent must outrank an optimistic finish-oracle rejection");

    assert_eq!(result, incumbent);
    let final_state = root
        .use_action(result[0], Condition::Normal, &settings)
        .unwrap();
    assert!(final_state.is_final(&settings));
}
