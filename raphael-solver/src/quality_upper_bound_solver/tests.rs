use raphael_sim::*;

use crate::finish_solver::Finishability;
use crate::{
    AtomicFlag, FinishSolver, SolverSettings,
    actions::{FULL_SEARCH_ACTIONS, use_action_combo},
    test_utils::*,
};

use super::QualityUbSolver;

/// Test that the QualityUbSolver is consistent and admissible.
/// It is consistent if the step-lb of a parent state is never greater than the step-lb of a child state.
/// It is admissible if the quality-ub of a state is never less than the quality of a reachable final state.
fn check_consistency(solver_settings: SolverSettings) {
    let mut solver = QualityUbSolver::new(solver_settings, AtomicFlag::default());
    solver.precompute().unwrap();
    let mut solver_shard = solver.create_shard();
    for state in generate_random_states(solver_settings, 1_000_000)
        .filter(|state| state.effects.combo() == Combo::None)
    {
        let state_upper_bound = solver_shard.quality_upper_bound(state).unwrap();
        for action in FULL_SEARCH_ACTIONS {
            let child_upper_bound = match use_action_combo(&solver_settings, state, action) {
                Ok(child) => match child.is_final(&solver_settings.simulator_settings) {
                    false => solver_shard.quality_upper_bound(child).unwrap(),
                    true if child.progress >= solver_settings.max_progress() => {
                        std::cmp::min(solver_settings.max_quality(), child.quality)
                    }
                    true => 0,
                },
                Err(_) => 0,
            };
            if state_upper_bound < child_upper_bound {
                dbg!(state, action, state_upper_bound, child_upper_bound);
                panic!("Parent's upper bound is less than child's upper bound");
            }
        }
    }
}

#[test_case::test_matrix(
    [20, 60, 80],
    [REGULAR_ACTIONS, NO_MANIPULATION, WITH_SPECIALIST_ACTIONS]
)]
fn consistency(max_durability: u16, allowed_actions: ActionMask) {
    let simulator_settings = Settings {
        max_progress: 2000,
        max_quality: 2000,
        max_durability,
        max_cp: 1000,
        base_progress: 100,
        base_quality: 100,
        job_level: 100,
        allowed_actions,
        adversarial: false,
        backload_progress: false,
        stellar_steady_hand_charges: 1,
    };
    let solver_settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: true,
    };
    check_consistency(solver_settings);
}

#[test]
fn stellar_resource_root_uses_the_admissible_max_quality_relaxation() {
    let settings = SolverSettings {
        simulator_settings: Settings {
            max_cp: 500,
            max_durability: 40,
            max_progress: 500,
            max_quality: 1000,
            base_progress: 100,
            base_quality: 100,
            job_level: 100,
            allowed_actions: REGULAR_ACTIONS,
            adversarial: false,
            backload_progress: false,
            stellar_steady_hand_charges: 1,
        },
        allow_non_max_quality_solutions: true,
    };
    let solver = QualityUbSolver::new(settings, AtomicFlag::default());
    let mut shard = solver.create_shard();
    let mut root = SimulationState::new(&settings.simulator_settings);
    assert_eq!(
        shard.quality_upper_bound(root).unwrap(),
        settings.max_quality()
    );
    root.effects.set_stellar_steady_hand_charges(0);
    root.effects.set_stellar_steady_hand(2);
    assert_eq!(
        shard.quality_upper_bound(root).unwrap(),
        settings.max_quality()
    );
}

#[test]
fn precompute_reuses_proven_maximal_slot_after_tricks_restores_cp() {
    let simulator_settings = Settings {
        max_cp: 756,
        max_durability: 40,
        max_progress: 10,
        max_quality: 96,
        base_progress: 1070,
        base_quality: 1630,
        job_level: 100,
        allowed_actions: ActionMask::all()
            .remove(Action::Manipulation)
            .remove(Action::TrainedEye)
            .remove(Action::StellarSteadyHand)
            .remove(Action::RapidSynthesis)
            .remove(Action::HastyTouch)
            .remove(Action::DaringTouch),
        adversarial: false,
        backload_progress: true,
        stellar_steady_hand_charges: 0,
    };
    let solver_settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: true,
    };

    let mut solver = QualityUbSolver::new(solver_settings, AtomicFlag::default());
    solver.precompute().unwrap();
}

#[test]
fn progress_query_matches_full_pareto_bound() {
    let simulator_settings = Settings {
        max_progress: 900,
        max_quality: 1200,
        max_durability: 40,
        max_cp: 300,
        base_progress: 100,
        base_quality: 100,
        job_level: 100,
        allowed_actions: REGULAR_ACTIONS,
        adversarial: false,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: true,
    };
    let mut solver = QualityUbSolver::new(settings, AtomicFlag::default());
    solver.precompute().unwrap();
    let mut pareto = solver.create_shard();
    let mut query = solver.create_shard();
    for state in generate_random_states(settings, 10_000).take(2_000) {
        assert_eq!(
            query.quality_upper_bound_for_progress(state).unwrap(),
            pareto.quality_upper_bound(state).unwrap(),
            "query-specific quality bound diverged for {state:?}",
        );
    }
}

#[test]
fn canonical_specialist_roots_are_covered_by_precompute() {
    let simulator_settings = Settings {
        max_progress: 500,
        max_quality: 1000,
        max_durability: 40,
        max_cp: 300,
        base_progress: 100,
        base_quality: 100,
        job_level: 100,
        allowed_actions: WITH_SPECIALIST_ACTIONS,
        adversarial: false,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: true,
    };
    let mut solver = QualityUbSolver::new(settings, AtomicFlag::default());
    solver.precompute().unwrap();
    let mut shard = solver.create_shard();
    for delineations in 0..=2 {
        let mut root = SimulationState::new(&simulator_settings);
        root.effects.set_crafter_delineations(delineations);
        root.effects = root.effects.canonicalize_specialist_resources();
        assert!(shard.quality_upper_bound_for_progress(root).is_ok());
    }
}

#[test]
fn daring_touch_interrupted_combo_bound_is_admissible() {
    let simulator_settings = Settings {
        max_cp: 0,
        max_durability: 30,
        max_progress: 100,
        max_quality: 347,
        base_progress: 100,
        base_quality: 100,
        job_level: 100,
        allowed_actions: ActionMask::regular().add(Action::QuickInnovation),
        adversarial: false,
        backload_progress: false,
        stellar_steady_hand_charges: 1,
    };
    let settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: true,
    };
    let mut solver = QualityUbSolver::new(settings, AtomicFlag::default());
    solver.precompute().unwrap();
    let mut shard = solver.create_shard();
    let mut finish = FinishSolver::new(settings);
    finish.precompute().unwrap();
    let mut state = SimulationState::new(&simulator_settings);
    for action in [
        Action::StellarSteadyHand,
        Action::HastyTouch,
        Action::QuickInnovation,
        Action::DaringTouch,
    ] {
        assert_eq!(
            shard.quality_upper_bound(state).unwrap(),
            347,
            "bound before {action:?} at {state:?}",
        );
        assert_ne!(
            finish.can_finish(&state).unwrap(),
            Finishability::Impossible
        );
        state = state
            .use_action(action, Condition::Normal, &simulator_settings)
            .unwrap();
    }
    assert_eq!(shard.quality_upper_bound(state).unwrap(), 347);
    assert_ne!(
        finish.can_finish(&state).unwrap(),
        Finishability::Impossible
    );
    state = state
        .use_action(
            Action::BasicSynthesis,
            Condition::Normal,
            &simulator_settings,
        )
        .unwrap();
    assert!(state.progress >= simulator_settings.max_progress);
}
