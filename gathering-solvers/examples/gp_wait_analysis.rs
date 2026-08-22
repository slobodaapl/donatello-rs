use gathering_solvers::{
    ActionModel, GatheringAction, LegacyOptions, Mechanics, RewardTier, SolveRequest, SolverMode,
    State, WeightedGain, solve,
};

fn request(gp: u16, rewards: Vec<RewardTier>) -> SolveRequest {
    SolveRequest {
        mode: SolverMode::ExpectedScrip,
        state: State {
            integrity: 6,
            max_integrity: 6,
            gp,
            max_gp: 1_000,
            remaining: 6,
            ..State::default()
        },
        rewards,
        actions: ActionModel {
            scour_gain: 200,
            meticulous_gain: 150,
            brazen_gains: (0..=10)
                .map(|step| WeightedGain {
                    gain: 100 + step * 20,
                    weight: 1,
                })
                .collect(),
            scrutiny_cost: 200,
            focus_cost: 100,
            priming_cost: 100,
            solid_reason_cost: 300,
            scour: true,
            brazen: false,
            meticulous: true,
            scrutiny: true,
            collectors_focus: false,
            priming_touch: false,
            solid_reason: true,
            wise_to_the_world: true,
        },
        mechanics: Mechanics {
            gather_success_bp: 10_000,
            intuition_bp: 4_000,
            focus_intuition_bp: 10_000,
            intuition_gain: 100,
            standard_proc_bp: 2_000,
            high_standard_upgrade_bp: 2_000,
            meticulous_preserve_bp: 2_500,
            high_standard_preserve_bonus_bp: 4_000,
            priming_preserve_multiplier: 2,
            solid_reason_eureka_bp: 5_000,
            revisit_bp: 0,
            collect_gp_regen: 6,
            scrutiny_gain_multiplier_bp: 20_000,
            max_states: 500_000,
        },
        legacy: LegacyOptions {
            target_score: 1_000,
            minimum_score: 600,
            use_full_rotation: true,
            always_use_solid_reason: true,
            abandon_when_complete: false,
        },
        unsupported_reason: None,
        plan_starting_gp: false,
    }
}

fn main() {
    let profiles = [
        (
            "purple20",
            vec![
                RewardTier {
                    threshold: 400,
                    scrip: 3,
                },
                RewardTier {
                    threshold: 700,
                    scrip: 7,
                },
                RewardTier {
                    threshold: 1_000,
                    scrip: 20,
                },
            ],
        ),
        (
            "orange38",
            vec![
                RewardTier {
                    threshold: 600,
                    scrip: 16,
                },
                RewardTier {
                    threshold: 800,
                    scrip: 23,
                },
                RewardTier {
                    threshold: 1_000,
                    scrip: 38,
                },
            ],
        ),
    ];

    println!("profile,gp,expected_reward,expected_terminal_gp,first_action,fallback");
    for (profile, rewards) in profiles {
        for gp in (0..=1_000).step_by(8) {
            let decision =
                solve(&request(gp, rewards.clone())).expect("representative solve failed");
            println!(
                "{profile},{gp},{:.6},{:.6},{:?},{}",
                decision.expected_reward,
                decision.expected_terminal_gp,
                decision.action,
                decision.fallback_reason.as_deref().unwrap_or("")
            );
        }
    }

    let _ = GatheringAction::Collect;
}
