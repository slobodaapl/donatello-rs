use raphael_sim::{Action, ActionMask, Condition, Settings, SimulationState};

const SETTINGS: Settings = Settings {
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
};

#[test]
fn expert_conditions_apply_exact_resource_and_value_modifiers() {
    let root = SimulationState::new(&SETTINGS);
    assert_eq!(
        Action::RapidSynthesis.success_rate(&root, Condition::Normal),
        50
    );
    assert_eq!(
        Action::RapidSynthesis.success_rate(&root, Condition::Centered),
        75
    );
    assert_eq!(
        Action::HastyTouch.success_rate(&root, Condition::Centered),
        85
    );
    let pliant = root
        .use_action(Action::Innovation, Condition::Pliant, &SETTINGS)
        .unwrap();
    assert_eq!(pliant.cp, 491);

    let sturdy = root
        .use_action(Action::BasicTouch, Condition::Sturdy, &SETTINGS)
        .unwrap();
    assert_eq!(sturdy.durability, 75);

    let waste_not_root = SimulationState {
        effects: root.effects.with_waste_not(2),
        ..root
    };
    let sturdy_waste_not = waste_not_root
        .use_action(Action::BasicTouch, Condition::Sturdy, &SETTINGS)
        .unwrap();
    assert_eq!(sturdy_waste_not.durability, 77);

    let normal = root
        .use_action(Action::BasicSynthesis, Condition::Normal, &SETTINGS)
        .unwrap();
    let malleable = root
        .use_action(Action::BasicSynthesis, Condition::Malleable, &SETTINGS)
        .unwrap();
    assert_eq!(malleable.progress, normal.progress * 3 / 2);

    let primed = root
        .use_action(Action::Innovation, Condition::Primed, &SETTINGS)
        .unwrap();
    assert_eq!(primed.effects.innovation(), 6);
    let primed_appraisal = root
        .use_action(Action::FinalAppraisal, Condition::Primed, &SETTINGS)
        .unwrap();
    assert_eq!(primed_appraisal.effects.final_appraisal(), 7);
}

#[test]
fn observed_condition_successors_are_deterministic() {
    assert_eq!(
        Condition::Excellent.deterministic_successor(),
        Condition::Poor
    );
    assert_eq!(
        Condition::GoodOmen.deterministic_successor(),
        Condition::Good
    );
    assert_eq!(Condition::Good.deterministic_successor(), Condition::Normal);
}

#[test]
fn zero_step_action_metadata_preserves_known_condition() {
    assert!(!Action::HeartAndSoul.increases_step_count());
    assert!(!Action::QuickInnovation.increases_step_count());
    assert!(Action::BasicTouch.increases_step_count());
}

#[test]
fn good_and_excellent_allow_condition_actions_without_heart_and_soul() {
    let root = SimulationState::new(&SETTINGS);
    assert!(
        root.use_action(Action::PreciseTouch, Condition::Good, &SETTINGS)
            .is_ok()
    );
    assert!(
        root.use_action(Action::IntensiveSynthesis, Condition::Excellent, &SETTINGS)
            .is_ok()
    );
    assert!(
        root.use_action(Action::TricksOfTheTrade, Condition::Good, &SETTINGS)
            .is_ok()
    );
    assert!(!root.effects.heart_and_soul_active());
}

#[test]
fn final_appraisal_is_zero_step_and_prevents_exactly_one_completion() {
    let root = SimulationState {
        progress: 4900,
        ..SimulationState::new(&SETTINGS)
    };
    let appraisal = root
        .use_action(Action::FinalAppraisal, Condition::Excellent, &SETTINGS)
        .unwrap();
    assert_eq!(appraisal.cp, root.cp - 1);
    assert_eq!(appraisal.effects.final_appraisal(), 5);
    assert!(!Action::FinalAppraisal.increases_step_count());

    let prevented = appraisal
        .use_action(Action::BasicSynthesis, Condition::Excellent, &SETTINGS)
        .unwrap();
    assert_eq!(prevented.progress, SETTINGS.max_progress - 1);
    assert_eq!(prevented.effects.final_appraisal(), 0);
    assert!(!prevented.is_final(&SETTINGS));

    let completed = prevented
        .use_action(Action::BasicSynthesis, Condition::Poor, &SETTINGS)
        .unwrap();
    assert!(completed.progress >= SETTINGS.max_progress);
}

#[test]
fn careful_observation_charges_are_persistent_state() {
    let root = SimulationState {
        effects: SimulationState::new(&SETTINGS)
            .effects
            .with_careful_observation_charges(2),
        ..SimulationState::new(&SETTINGS)
    };
    let next = root
        .use_action(Action::BasicSynthesis, Condition::Normal, &SETTINGS)
        .unwrap();
    assert_eq!(next.effects.careful_observation_charges(), 2);
}

#[test]
fn donatello_action_does_not_change_raphael_regular_defaults() {
    assert!(!ActionMask::regular().has(Action::FinalAppraisal));
}
