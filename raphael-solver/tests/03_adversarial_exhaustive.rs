use expect_test::expect;
use raphael_sim::*;
use raphael_solver::{AtomicFlag, MacroSolver, SolverSettings};

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct SolutionScore {
    pub capped_quality: u16,
    pub steps: u8,
    pub duration: u8,
    pub overflow_quality: u16,
}

fn is_progress_backloaded(settings: &SolverSettings, actions: &[Action]) -> bool {
    let mut state = SimulationState::new(&settings.simulator_settings);
    let mut quality_lock = None;
    for action in actions {
        state = state
            .use_action(*action, Condition::Normal, &settings.simulator_settings)
            .unwrap();
        if state.progress != 0 && quality_lock.is_none() {
            quality_lock = Some(state.quality);
        }
    }
    quality_lock.is_none_or(|quality| state.quality == quality)
}

fn test_with_settings(
    settings: SolverSettings,
    expected_score: expect_test::Expect,
    expected_runtime_stats: expect_test::Expect,
) {
    let mut solver = MacroSolver::new(
        settings,
        Box::new(|_| {}),
        Box::new(|_| {}),
        AtomicFlag::new(),
    );
    let result = solver.solve();
    let score = result.map(|actions| {
        let final_state =
            SimulationState::from_macro(&settings.simulator_settings, &actions).unwrap();
        assert!(final_state.progress >= settings.max_progress());
        if settings.simulator_settings.backload_progress {
            assert!(is_progress_backloaded(&settings, &actions));
        }
        SolutionScore {
            capped_quality: std::cmp::min(final_state.quality, settings.max_quality()),
            steps: actions.len() as u8,
            duration: actions.iter().map(|action| action.time_cost()).sum(),
            overflow_quality: final_state.quality.saturating_sub(settings.max_quality()),
        }
    });
    expected_score.assert_debug_eq(&score);
    expected_runtime_stats.assert_debug_eq(&solver.runtime_stats());
}

const SETTINGS: Settings = Settings {
    max_cp: 370,
    max_durability: 60,
    max_progress: 2000,
    max_quality: 40000,
    base_progress: 100,
    base_quality: 100,
    job_level: 100,
    allowed_actions: ActionMask::all()
        .remove(Action::TrainedEye)
        .remove(Action::HeartAndSoul)
        .remove(Action::QuickInnovation),
    adversarial: true,
    backload_progress: false,
    stellar_steady_hand_charges: 0,
};

#[test]
fn stuffed_peppers() {
    // lv99 Rarefied Stuffed Peppers
    // 4785 CMS, 4758 Ctrl, 646 CP
    let simulator_settings = Settings {
        max_cp: 646,
        max_durability: 80,
        max_progress: 6300,
        max_quality: 11400,
        base_progress: 289,
        base_quality: 360,
        ..SETTINGS
    };
    let solver_settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: true,
    };
    let expected_score = expect![[r#"
        Ok(
            SolutionScore {
                capped_quality: 11400,
                steps: 16,
                duration: 45,
                overflow_quality: 282,
            },
        )
    "#]];
    let expected_runtime_stats = expect![[r#"
        MacroSolverStats {
            search_queue_stats: SearchQueueStats {
                inserted_nodes: 1235413,
                processed_nodes: 56079,
            },
            finish_solver_stats: FinishSolverStats {
                states: 15891,
                values: 210965,
            },
            quality_ub_stats: QualityUbSolverStats {
                states_on_main: 2271577,
                states_on_shards: 14,
                values: 39200084,
            },
            step_lb_stats: StepLbSolverStats {
                states_on_main: 366835,
                states_on_shards: 33059353,
                values: 44443261,
            },
        }
    "#]];
    test_with_settings(solver_settings, expected_score, expected_runtime_stats);
}

#[test]
fn test_rare_tacos_2() {
    // lv100 Rarefied Tacos de Carne Asada
    // 4785 CMS, 4758 Ctrl, 646 CP
    let simulator_settings = Settings {
        max_cp: 646,
        max_durability: 80,
        max_progress: 6600,
        max_quality: 12000,
        base_progress: 256,
        base_quality: 265,
        job_level: 100,
        allowed_actions: ActionMask::regular(),
        adversarial: true,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let solver_settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: false,
    };
    let expected_score = expect![[r#"
        Ok(
            SolutionScore {
                capped_quality: 12000,
                steps: 32,
                duration: 91,
                overflow_quality: 127,
            },
        )
    "#]];
    let expected_runtime_stats = expect![[r#"
        MacroSolverStats {
            search_queue_stats: SearchQueueStats {
                inserted_nodes: 8532353,
                processed_nodes: 3572309,
            },
            finish_solver_stats: FinishSolverStats {
                states: 15891,
                values: 308710,
            },
            quality_ub_stats: QualityUbSolverStats {
                states_on_main: 2490500,
                states_on_shards: 77965,
                values: 70123242,
            },
            step_lb_stats: StepLbSolverStats {
                states_on_main: 878246,
                states_on_shards: 32291671,
                values: 59354698,
            },
        }
    "#]];
    test_with_settings(solver_settings, expected_score, expected_runtime_stats);
}

#[test]
fn test_mountain_chromite_ingot_no_manipulation() {
    // Mountain Chromite Ingot
    // 3076 Craftsmanship, 3106 Control, Level 90, HQ Tsai Tou Vonou
    let simulator_settings = Settings {
        max_cp: 616,
        max_durability: 40,
        max_progress: 2000,
        max_quality: 8200,
        base_progress: 217,
        base_quality: 293,
        job_level: 90,
        allowed_actions: ActionMask::all()
            .remove(Action::Manipulation)
            .remove(Action::TrainedEye)
            .remove(Action::HeartAndSoul)
            .remove(Action::QuickInnovation),
        adversarial: true,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let solver_settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: true,
    };
    let expected_score = expect![[r#"
        Ok(
            SolutionScore {
                capped_quality: 8200,
                steps: 14,
                duration: 38,
                overflow_quality: 32,
            },
        )
    "#]];
    let expected_runtime_stats = expect![[r#"
        MacroSolverStats {
            search_queue_stats: SearchQueueStats {
                inserted_nodes: 350315,
                processed_nodes: 21875,
            },
            finish_solver_stats: FinishSolverStats {
                states: 798,
                values: 3551,
            },
            quality_ub_stats: QualityUbSolverStats {
                states_on_main: 1800421,
                states_on_shards: 39134,
                values: 16726546,
            },
            step_lb_stats: StepLbSolverStats {
                states_on_main: 28922,
                states_on_shards: 1130203,
                values: 1421347,
            },
        }
    "#]];
    test_with_settings(solver_settings, expected_score, expected_runtime_stats);
}

#[test]
fn test_indagator_3858_4057() {
    let simulator_settings = Settings {
        max_cp: 687,
        max_durability: 70,
        max_progress: 5720,
        max_quality: 12900,
        base_progress: 239,
        base_quality: 271,
        job_level: 90,
        allowed_actions: ActionMask::regular(),
        adversarial: true,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let solver_settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: true,
    };
    let expected_score = expect![[r#"
        Ok(
            SolutionScore {
                capped_quality: 10686,
                steps: 26,
                duration: 71,
                overflow_quality: 0,
            },
        )
    "#]];
    let expected_runtime_stats = expect![[r#"
        MacroSolverStats {
            search_queue_stats: SearchQueueStats {
                inserted_nodes: 233283,
                processed_nodes: 20992,
            },
            finish_solver_stats: FinishSolverStats {
                states: 5308,
                values: 102529,
            },
            quality_ub_stats: QualityUbSolverStats {
                states_on_main: 2515254,
                states_on_shards: 140896,
                values: 64645870,
            },
            step_lb_stats: StepLbSolverStats {
                states_on_main: 0,
                states_on_shards: 0,
                values: 0,
            },
        }
    "#]];
    test_with_settings(solver_settings, expected_score, expected_runtime_stats);
}

#[test]
fn test_rare_tacos_4628_4410() {
    let simulator_settings = Settings {
        max_cp: 675,
        max_durability: 80,
        max_progress: 6600,
        max_quality: 12000,
        base_progress: 246,
        base_quality: 246,
        job_level: 100,
        allowed_actions: ActionMask::all()
            .remove(Action::Manipulation)
            .remove(Action::TrainedEye)
            .remove(Action::HeartAndSoul)
            .remove(Action::QuickInnovation),
        adversarial: true,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let solver_settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: true,
    };
    let expected_score = expect![[r#"
        Ok(
            SolutionScore {
                capped_quality: 11748,
                steps: 31,
                duration: 88,
                overflow_quality: 0,
            },
        )
    "#]];
    let expected_runtime_stats = expect![[r#"
        MacroSolverStats {
            search_queue_stats: SearchQueueStats {
                inserted_nodes: 16891597,
                processed_nodes: 3391553,
            },
            finish_solver_stats: FinishSolverStats {
                states: 2966,
                values: 65772,
            },
            quality_ub_stats: QualityUbSolverStats {
                states_on_main: 2623346,
                states_on_shards: 89941,
                values: 78759558,
            },
            step_lb_stats: StepLbSolverStats {
                states_on_main: 213458,
                states_on_shards: 4355645,
                values: 9879020,
            },
        }
    "#]];
    test_with_settings(solver_settings, expected_score, expected_runtime_stats);
}

#[test]
fn issue_113() {
    // https://github.com/KonaeAkira/raphael-rs/issues/113
    // Ceremonial Gunblade
    // 5428/5236/645 + HQ Ceviche + HQ Cunning Tisane
    let simulator_settings = Settings {
        max_cp: 768,
        max_durability: 70,
        max_progress: 9000,
        max_quality: 18700,
        base_progress: 297,
        base_quality: 288,
        job_level: 100,
        allowed_actions: ActionMask::regular(),
        adversarial: true,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let solver_settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: true,
    };
    let expected_score = expect![[r#"
        Ok(
            SolutionScore {
                capped_quality: 14070,
                steps: 33,
                duration: 93,
                overflow_quality: 0,
            },
        )
    "#]];
    let expected_runtime_stats = expect![[r#"
        MacroSolverStats {
            search_queue_stats: SearchQueueStats {
                inserted_nodes: 24816993,
                processed_nodes: 1551677,
            },
            finish_solver_stats: FinishSolverStats {
                states: 13977,
                values: 437006,
            },
            quality_ub_stats: QualityUbSolverStats {
                states_on_main: 3043236,
                states_on_shards: 80576,
                values: 120614822,
            },
            step_lb_stats: StepLbSolverStats {
                states_on_main: 0,
                states_on_shards: 0,
                values: 0,
            },
        }
    "#]];
    test_with_settings(solver_settings, expected_score, expected_runtime_stats);
}

#[test]
fn issue_118() {
    // https://github.com/KonaeAkira/raphael-rs/issues/118
    let simulator_settings = Settings {
        max_cp: 614,
        max_durability: 20,
        max_progress: 2310,
        max_quality: 8400,
        base_progress: 205,
        base_quality: 240,
        job_level: 100,
        allowed_actions: ActionMask::regular(),
        adversarial: true,
        backload_progress: false,
        stellar_steady_hand_charges: 0,
    };
    let solver_settings = SolverSettings {
        simulator_settings,
        allow_non_max_quality_solutions: true,
    };
    let expected_score = expect![[r#"
        Ok(
            SolutionScore {
                capped_quality: 8400,
                steps: 19,
                duration: 52,
                overflow_quality: 84,
            },
        )
    "#]];
    let expected_runtime_stats = expect![[r#"
        MacroSolverStats {
            search_queue_stats: SearchQueueStats {
                inserted_nodes: 18370324,
                processed_nodes: 1330458,
            },
            finish_solver_stats: FinishSolverStats {
                states: 3619,
                values: 22096,
            },
            quality_ub_stats: QualityUbSolverStats {
                states_on_main: 1930860,
                states_on_shards: 61569,
                values: 25587911,
            },
            step_lb_stats: StepLbSolverStats {
                states_on_main: 100860,
                states_on_shards: 5977571,
                values: 7693957,
            },
        }
    "#]];
    test_with_settings(solver_settings, expected_score, expected_runtime_stats);
}
