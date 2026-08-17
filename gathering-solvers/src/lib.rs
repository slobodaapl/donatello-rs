use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SolverMode {
    ExpectedScrip,
    Legacy,
    MaximizeCollectability,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum GatheringAction {
    Scour,
    Brazen,
    Meticulous,
    Scrutiny,
    CollectorsFocus,
    PrimingTouch,
    SolidReason,
    WiseToTheWorld,
    Collect,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct State {
    pub collectability: u16,
    pub integrity: u8,
    pub max_integrity: u8,
    pub gp: u16,
    pub max_gp: u16,
    pub remaining: u16,
    #[serde(default)]
    pub scrutiny: bool,
    #[serde(default)]
    pub collectors_focus: bool,
    #[serde(default)]
    pub priming_touch: bool,
    #[serde(default)]
    pub standard: u8,
    #[serde(default)]
    pub eureka: bool,
    #[serde(default)]
    pub revisit_used: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RewardTier {
    pub threshold: u16,
    pub scrip: u16,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WeightedGain {
    pub gain: u16,
    pub weight: u16,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionModel {
    pub scour_gain: u16,
    pub meticulous_gain: u16,
    pub brazen_gains: Vec<WeightedGain>,
    pub scrutiny_cost: u16,
    pub focus_cost: u16,
    pub priming_cost: u16,
    pub solid_reason_cost: u16,
    #[serde(default = "yes")]
    pub scour: bool,
    #[serde(default = "yes")]
    pub brazen: bool,
    #[serde(default = "yes")]
    pub meticulous: bool,
    #[serde(default = "yes")]
    pub scrutiny: bool,
    #[serde(default)]
    pub collectors_focus: bool,
    #[serde(default)]
    pub priming_touch: bool,
    #[serde(default = "yes")]
    pub solid_reason: bool,
    #[serde(default = "yes")]
    pub wise_to_the_world: bool,
}

const fn yes() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Mechanics {
    pub gather_success_bp: u16,
    pub intuition_bp: u16,
    pub focus_intuition_bp: u16,
    pub intuition_gain: u16,
    pub standard_proc_bp: u16,
    pub high_standard_upgrade_bp: u16,
    pub meticulous_preserve_bp: u16,
    pub high_standard_preserve_bonus_bp: u16,
    pub priming_preserve_multiplier: u8,
    pub solid_reason_eureka_bp: u16,
    #[serde(default)]
    pub revisit_bp: u16,
    #[serde(default = "default_gp_regen")]
    pub collect_gp_regen: u16,
    #[serde(default = "default_scrutiny_multiplier")]
    pub scrutiny_gain_multiplier_bp: u16,
    #[serde(default = "default_max_states")]
    pub max_states: u32,
}

const fn default_gp_regen() -> u16 {
    6
}

const fn default_scrutiny_multiplier() -> u16 {
    20_000
}

const fn default_max_states() -> u32 {
    500_000
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LegacyOptions {
    pub target_score: u16,
    pub minimum_score: u16,
    pub use_full_rotation: bool,
    pub always_use_solid_reason: bool,
    pub abandon_when_complete: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SolveRequest {
    pub mode: SolverMode,
    pub state: State,
    pub rewards: Vec<RewardTier>,
    pub actions: ActionModel,
    pub mechanics: Mechanics,
    pub legacy: LegacyOptions,
    #[serde(default)]
    pub unsupported_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Decision {
    pub action: GatheringAction,
    pub solver_used: SolverMode,
    pub expected_reward: f64,
    pub expected_perfect_collects: f64,
    pub expected_terminal_gp: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Value {
    perfect_collects: f64,
    reward: f64,
    terminal_gp: f64,
    actions: f64,
}

impl Value {
    fn after(
        self,
        probability: f64,
        immediate_reward: f64,
        immediate_perfect_collects: f64,
    ) -> Self {
        Self {
            perfect_collects: probability * (self.perfect_collects + immediate_perfect_collects),
            reward: probability * (self.reward + immediate_reward),
            terminal_gp: probability * self.terminal_gp,
            actions: probability * self.actions,
        }
    }

    fn add(&mut self, other: Self) {
        self.perfect_collects += other.perfect_collects;
        self.reward += other.reward;
        self.terminal_gp += other.terminal_gp;
        self.actions += other.actions;
    }

    fn better_than(self, other: Self) -> bool {
        const EPSILON: f64 = 1e-9;
        self.perfect_collects > other.perfect_collects + EPSILON
            || ((self.perfect_collects - other.perfect_collects).abs() <= EPSILON
                && (self.reward > other.reward + EPSILON
                    || ((self.reward - other.reward).abs() <= EPSILON
                        && (self.terminal_gp > other.terminal_gp + EPSILON
                            || ((self.terminal_gp - other.terminal_gp).abs() <= EPSILON
                                && self.actions < other.actions - EPSILON)))))
    }
}

pub fn solve(request: &SolveRequest) -> Result<Decision, String> {
    validate(request)?;
    if request.mode == SolverMode::Legacy {
        return legacy_decision(request, None);
    }
    if request.mode == SolverMode::ExpectedScrip
        && let Some(reason) = &request.unsupported_reason
    {
        return legacy_decision(request, Some(reason.clone()));
    }

    match ExpectedSolver::new(request).solve() {
        Ok((action, value)) => Ok(Decision {
            action,
            solver_used: request.mode,
            expected_reward: value.reward,
            expected_perfect_collects: value.perfect_collects,
            expected_terminal_gp: value.terminal_gp,
            fallback_reason: None,
        }),
        Err(reason) => legacy_decision(request, Some(reason)),
    }
}

fn validate(request: &SolveRequest) -> Result<(), String> {
    let state = request.state;
    if state.integrity == 0 || state.max_integrity == 0 || state.integrity > state.max_integrity {
        return Err(String::from("invalid integrity"));
    }
    if state.gp > state.max_gp || state.standard > 2 {
        return Err(String::from("invalid state"));
    }
    if request.rewards.is_empty() || request.rewards.iter().any(|tier| tier.threshold == 0) {
        return Err(String::from("missing reward tiers"));
    }
    if request
        .rewards
        .windows(2)
        .any(|tiers| tiers[0].threshold >= tiers[1].threshold || tiers[0].scrip > tiers[1].scrip)
    {
        return Err(String::from("reward tiers are not strictly ordered"));
    }
    let mechanics = request.mechanics;
    for probability in [
        mechanics.gather_success_bp,
        mechanics.intuition_bp,
        mechanics.focus_intuition_bp,
        mechanics.standard_proc_bp,
        mechanics.high_standard_upgrade_bp,
        mechanics.meticulous_preserve_bp,
        mechanics.high_standard_preserve_bonus_bp,
        mechanics.solid_reason_eureka_bp,
        mechanics.revisit_bp,
    ] {
        if probability > 10_000 {
            return Err(String::from("probability exceeds 100%"));
        }
    }
    if request.actions.brazen
        && (request.actions.brazen_gains.is_empty()
            || request
                .actions
                .brazen_gains
                .iter()
                .all(|gain| gain.weight == 0))
    {
        return Err(String::from("missing Brazen Appraisal distribution"));
    }
    Ok(())
}

fn legacy_decision(
    request: &SolveRequest,
    fallback_reason: Option<String>,
) -> Result<Decision, String> {
    let state = request.state;
    let actions = &request.actions;
    let legacy = &request.legacy;
    if state.remaining == 0 && legacy.abandon_when_complete {
        return Err(String::from("requested quantity already satisfied"));
    }
    if state.eureka && state.integrity < state.max_integrity && actions.wise_to_the_world {
        return legacy_result(GatheringAction::WiseToTheWorld, fallback_reason);
    }
    if state.collectability >= legacy.target_score {
        if (legacy.use_full_rotation || legacy.always_use_solid_reason)
            && state.integrity <= 2.min(state.max_integrity.saturating_sub(1))
            && state.remaining > u16::from(state.integrity)
            && actions.solid_reason
            && state.gp >= actions.solid_reason_cost
        {
            return legacy_result(GatheringAction::SolidReason, fallback_reason);
        }
        return legacy_result(GatheringAction::Collect, fallback_reason);
    }
    if state.integrity == 1 && state.collectability >= legacy.minimum_score {
        return legacy_result(GatheringAction::Collect, fallback_reason);
    }

    let scrutiny_available =
        actions.scrutiny && !state.scrutiny && state.gp >= actions.scrutiny_cost;
    let scour_available = actions.scour;
    let meticulous_available = actions.meticulous;
    let brazen_available = actions.brazen;
    let target = legacy.target_score;
    let reaches = |gain: u16| state.collectability.saturating_add(gain) >= target;
    let brazen_max = actions
        .brazen_gains
        .iter()
        .map(|gain| gain.gain)
        .max()
        .unwrap_or(0);
    if legacy.use_full_rotation
        && scrutiny_available
        && !(scour_available && reaches(actions.scour_gain)
            || meticulous_available && reaches(actions.meticulous_gain)
            || brazen_available && reaches(brazen_max))
    {
        return legacy_result(GatheringAction::Scrutiny, fallback_reason);
    }
    if meticulous_available && reaches(actions.meticulous_gain) {
        return legacy_result(GatheringAction::Meticulous, fallback_reason);
    }
    if state.standard == 2 && brazen_available {
        return legacy_result(GatheringAction::Brazen, fallback_reason);
    }
    if scour_available && reaches(actions.scour_gain) {
        return legacy_result(GatheringAction::Scour, fallback_reason);
    }
    if meticulous_available {
        return legacy_result(GatheringAction::Meticulous, fallback_reason);
    }
    if state.standard == 1 && brazen_available {
        return legacy_result(GatheringAction::Brazen, fallback_reason);
    }
    if scour_available {
        return legacy_result(GatheringAction::Scour, fallback_reason);
    }
    if brazen_available {
        return legacy_result(GatheringAction::Brazen, fallback_reason);
    }
    Err(String::from("no legal collectable action"))
}

fn legacy_result(
    action: GatheringAction,
    fallback_reason: Option<String>,
) -> Result<Decision, String> {
    Ok(Decision {
        action,
        solver_used: SolverMode::Legacy,
        expected_reward: 0.0,
        expected_perfect_collects: 0.0,
        expected_terminal_gp: 0.0,
        fallback_reason,
    })
}

struct ExpectedSolver<'a> {
    request: &'a SolveRequest,
    memo: HashMap<State, Value>,
    visiting: HashSet<State>,
}

impl<'a> ExpectedSolver<'a> {
    fn new(request: &'a SolveRequest) -> Self {
        Self {
            request,
            memo: HashMap::new(),
            visiting: HashSet::new(),
        }
    }

    fn solve(mut self) -> Result<(GatheringAction, Value), String> {
        if self.request.state.remaining == 0 {
            return Err(String::from("requested quantity already satisfied"));
        }
        let candidates = self.candidates(self.request.state)?;
        candidates
            .into_iter()
            .max_by(|left, right| compare_candidates(*left, *right))
            .ok_or_else(|| String::from("no legal collectable action"))
    }

    fn value(&mut self, state: State) -> Result<Value, String> {
        if state.remaining == 0 || state.integrity == 0 {
            return Ok(Value {
                terminal_gp: f64::from(state.gp),
                ..Value::default()
            });
        }
        if let Some(value) = self.memo.get(&state) {
            return Ok(*value);
        }
        if self.memo.len() as u32 >= self.request.mechanics.max_states {
            return Err(String::from("gathering solver search budget exceeded"));
        }
        if !self.visiting.insert(state) {
            return Err(String::from("cyclic transition kernel"));
        }
        let result = self
            .candidates(state)?
            .into_iter()
            .max_by(|left, right| compare_candidates(*left, *right))
            .map(|(_, value)| value)
            .unwrap_or(Value {
                terminal_gp: f64::from(state.gp),
                ..Value::default()
            });
        self.visiting.remove(&state);
        self.memo.insert(state, result);
        Ok(result)
    }

    fn candidates(&mut self, state: State) -> Result<Vec<(GatheringAction, Value)>, String> {
        let mut result = Vec::new();
        for action in ACTION_ORDER {
            if let Some(outcomes) = self.outcomes(state, action) {
                let mut value = Value::default();
                for (probability, successor, reward, perfect_collects) in outcomes {
                    let successor_value = self.value(successor)?;
                    value.add(successor_value.after(probability, reward, perfect_collects));
                }
                value.actions += 1.0;
                result.push((action, value));
            }
        }
        Ok(result)
    }

    fn outcomes(
        &self,
        state: State,
        action: GatheringAction,
    ) -> Option<Vec<(f64, State, f64, f64)>> {
        match action {
            GatheringAction::Collect
                if self.request.mode == SolverMode::MaximizeCollectability
                    || self.reward(state.collectability) > 0 =>
            {
                Some(self.collect_outcomes(state))
            }
            GatheringAction::Scour
                if self.request.actions.scour && self.appraisal_useful(state) =>
            {
                Some(self.appraisal_outcomes(
                    state,
                    [(self.request.actions.scour_gain, 1.0)],
                    false,
                ))
            }
            GatheringAction::Brazen
                if self.request.actions.brazen && self.appraisal_useful(state) =>
            {
                let gains = if state.standard == 2 {
                    let maximum = self
                        .request
                        .actions
                        .brazen_gains
                        .iter()
                        .map(|gain| gain.gain)
                        .max()?;
                    vec![(maximum, 1.0)]
                } else {
                    let total: u32 = self
                        .request
                        .actions
                        .brazen_gains
                        .iter()
                        .map(|gain| u32::from(gain.weight))
                        .sum();
                    self.request
                        .actions
                        .brazen_gains
                        .iter()
                        .filter(|gain| gain.weight > 0)
                        .map(|gain| {
                            let effective_gain = if state.standard == 1 {
                                gain.gain.max(self.request.actions.scour_gain)
                            } else {
                                gain.gain
                            };
                            (effective_gain, f64::from(gain.weight) / f64::from(total))
                        })
                        .collect()
                };
                Some(self.appraisal_outcomes(state, gains, false))
            }
            GatheringAction::Meticulous
                if self.request.actions.meticulous && self.appraisal_useful(state) =>
            {
                let gain = if state.standard > 0 {
                    self.request
                        .actions
                        .meticulous_gain
                        .max(self.request.actions.scour_gain)
                } else {
                    self.request.actions.meticulous_gain
                };
                Some(self.appraisal_outcomes(state, [(gain, 1.0)], true))
            }
            GatheringAction::Scrutiny
                if self.request.actions.scrutiny
                    && !state.scrutiny
                    && state.gp >= self.request.actions.scrutiny_cost =>
            {
                let mut next = state;
                next.gp -= self.request.actions.scrutiny_cost;
                next.scrutiny = true;
                Some(vec![(1.0, next, 0.0, 0.0)])
            }
            GatheringAction::CollectorsFocus
                if self.request.actions.collectors_focus
                    && !state.collectors_focus
                    && state.gp >= self.request.actions.focus_cost =>
            {
                let mut next = state;
                next.gp -= self.request.actions.focus_cost;
                next.collectors_focus = true;
                Some(vec![(1.0, next, 0.0, 0.0)])
            }
            GatheringAction::PrimingTouch
                if self.request.actions.priming_touch
                    && !state.priming_touch
                    && state.gp >= self.request.actions.priming_cost =>
            {
                let mut next = state;
                next.gp -= self.request.actions.priming_cost;
                next.priming_touch = true;
                Some(vec![(1.0, next, 0.0, 0.0)])
            }
            GatheringAction::SolidReason
                if self.request.actions.solid_reason
                    && state.integrity < state.max_integrity
                    && state.gp >= self.request.actions.solid_reason_cost =>
            {
                let mut success = state;
                success.gp -= self.request.actions.solid_reason_cost;
                success.integrity += 1;
                success.eureka = true;
                let mut failure = success;
                failure.eureka = false;
                Some(binary_outcomes(
                    self.request.mechanics.solid_reason_eureka_bp,
                    success,
                    failure,
                ))
            }
            GatheringAction::WiseToTheWorld
                if self.request.actions.wise_to_the_world
                    && state.eureka
                    && state.integrity < state.max_integrity =>
            {
                let mut next = state;
                next.eureka = false;
                next.integrity += 1;
                Some(vec![(1.0, next, 0.0, 0.0)])
            }
            _ => None,
        }
    }

    fn appraisal_outcomes(
        &self,
        state: State,
        gains: impl IntoIterator<Item = (u16, f64)>,
        meticulous: bool,
    ) -> Vec<(f64, State, f64, f64)> {
        let mechanics = self.request.mechanics;
        let intuition_bp = if state.collectors_focus {
            mechanics.focus_intuition_bp
        } else {
            mechanics.intuition_bp
        };
        let standard_bonus = if state.standard == 2 {
            mechanics.high_standard_preserve_bonus_bp
        } else {
            0
        };
        let multiplier = if state.priming_touch {
            u32::from(mechanics.priming_preserve_multiplier.max(1))
        } else {
            1
        };
        let preserve_bp = if meticulous {
            ((u32::from(mechanics.meticulous_preserve_bp) * multiplier) + u32::from(standard_bonus))
                .min(10_000) as u16
        } else {
            0
        };
        let mut outcomes = Vec::new();
        for (base_gain, gain_probability) in gains {
            let gain = if state.scrutiny {
                (u32::from(base_gain) * u32::from(mechanics.scrutiny_gain_multiplier_bp) / 10_000)
                    .min(u32::from(u16::MAX)) as u16
            } else {
                base_gain
            };
            for (preserve_probability, preserved) in probability_branches(preserve_bp) {
                for (intuition_probability, intuition) in probability_branches(intuition_bp) {
                    for (standard_probability, standard_proc) in
                        probability_branches(mechanics.standard_proc_bp)
                    {
                        let high_branches = if standard_proc {
                            probability_branches(mechanics.high_standard_upgrade_bp)
                        } else {
                            vec![(1.0, false)]
                        };
                        for (high_probability, high_standard) in high_branches {
                            let mut next = state;
                            let intuition_gain = if intuition {
                                mechanics.intuition_gain
                            } else {
                                0
                            };
                            next.collectability = next
                                .collectability
                                .saturating_add(gain)
                                .saturating_add(intuition_gain)
                                .min(1000);
                            if !preserved {
                                next.integrity -= 1;
                            }
                            next.scrutiny = false;
                            next.collectors_focus = false;
                            next.priming_touch = false;
                            next.standard = if high_standard {
                                2
                            } else if standard_proc {
                                1
                            } else {
                                0
                            };
                            outcomes.push((
                                gain_probability
                                    * preserve_probability
                                    * intuition_probability
                                    * standard_probability
                                    * high_probability,
                                next,
                                0.0,
                                0.0,
                            ));
                        }
                    }
                }
            }
        }
        merge_outcomes(outcomes)
    }

    fn collect_outcomes(&self, state: State) -> Vec<(f64, State, f64, f64)> {
        let success_probability = probability(self.request.mechanics.gather_success_bp);
        let reward = f64::from(self.reward(state.collectability));
        let perfect_collects = if self.request.mode == SolverMode::MaximizeCollectability
            && state.collectability == 1000
        {
            1.0
        } else {
            0.0
        };
        let mut outcomes = Vec::new();
        for (gather_probability, success) in [
            (success_probability, true),
            (1.0 - success_probability, false),
        ] {
            if gather_probability == 0.0 {
                continue;
            }
            let mut next = state;
            next.integrity -= 1;
            next.gp = next
                .gp
                .saturating_add(self.request.mechanics.collect_gp_regen)
                .min(next.max_gp);
            if success {
                next.remaining = next.remaining.saturating_sub(1);
                next.collectability = 0;
                next.scrutiny = false;
                next.collectors_focus = false;
                next.priming_touch = false;
                next.standard = 0;
                next.eureka = false;
            }
            if next.integrity == 0 && next.remaining > 0 && !next.revisit_used {
                let revisit_probability = probability(self.request.mechanics.revisit_bp);
                if revisit_probability > 0.0 {
                    let mut revisited = next;
                    revisited.integrity = revisited.max_integrity;
                    revisited.gp = revisited.max_gp;
                    revisited.revisit_used = true;
                    revisited.collectability = 0;
                    revisited.scrutiny = false;
                    revisited.collectors_focus = false;
                    revisited.priming_touch = false;
                    revisited.standard = 0;
                    revisited.eureka = false;
                    outcomes.push((
                        gather_probability * revisit_probability,
                        revisited,
                        if success { reward } else { 0.0 },
                        if success { perfect_collects } else { 0.0 },
                    ));
                    outcomes.push((
                        gather_probability * (1.0 - revisit_probability),
                        next,
                        if success { reward } else { 0.0 },
                        if success { perfect_collects } else { 0.0 },
                    ));
                    continue;
                }
            }
            outcomes.push((
                gather_probability,
                next,
                if success { reward } else { 0.0 },
                if success { perfect_collects } else { 0.0 },
            ));
        }
        merge_outcomes(outcomes)
    }

    fn reward(&self, collectability: u16) -> u16 {
        if self.request.mode == SolverMode::MaximizeCollectability {
            return collectability;
        }

        self.request
            .rewards
            .iter()
            .rev()
            .find(|tier| collectability >= tier.threshold)
            .map_or(0, |tier| tier.scrip)
    }

    fn appraisal_useful(&self, state: State) -> bool {
        if self.request.mode == SolverMode::MaximizeCollectability {
            return state.collectability < 1000;
        }

        self.request
            .rewards
            .last()
            .is_some_and(|tier| state.collectability < tier.threshold)
    }
}

const ACTION_ORDER: [GatheringAction; 9] = [
    GatheringAction::Collect,
    GatheringAction::Meticulous,
    GatheringAction::Scour,
    GatheringAction::Brazen,
    GatheringAction::Scrutiny,
    GatheringAction::CollectorsFocus,
    GatheringAction::PrimingTouch,
    GatheringAction::WiseToTheWorld,
    GatheringAction::SolidReason,
];

fn compare_candidates(
    left: (GatheringAction, Value),
    right: (GatheringAction, Value),
) -> std::cmp::Ordering {
    if left.1.better_than(right.1) {
        std::cmp::Ordering::Greater
    } else if right.1.better_than(left.1) {
        std::cmp::Ordering::Less
    } else {
        action_rank(right.0).cmp(&action_rank(left.0))
    }
}

fn action_rank(action: GatheringAction) -> usize {
    ACTION_ORDER
        .iter()
        .position(|candidate| *candidate == action)
        .unwrap()
}

fn probability(bp: u16) -> f64 {
    f64::from(bp) / 10_000.0
}

fn probability_branches(bp: u16) -> Vec<(f64, bool)> {
    let success = probability(bp);
    match bp {
        0 => vec![(1.0, false)],
        10_000 => vec![(1.0, true)],
        _ => vec![(success, true), (1.0 - success, false)],
    }
}

fn binary_outcomes(bp: u16, success: State, failure: State) -> Vec<(f64, State, f64, f64)> {
    probability_branches(bp)
        .into_iter()
        .map(|(probability, branch)| {
            (
                probability,
                if branch { success } else { failure },
                0.0,
                0.0,
            )
        })
        .collect()
}

fn merge_outcomes(outcomes: Vec<(f64, State, f64, f64)>) -> Vec<(f64, State, f64, f64)> {
    let mut merged: HashMap<(State, u64, u64), f64> = HashMap::new();
    for (probability, state, reward, perfect_collects) in outcomes {
        *merged
            .entry((state, reward.to_bits(), perfect_collects.to_bits()))
            .or_default() += probability;
    }
    merged
        .into_iter()
        .map(|((state, reward, perfect_collects), probability)| {
            (
                probability,
                state,
                f64::from_bits(reward),
                f64::from_bits(perfect_collects),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(rewards: Vec<RewardTier>) -> SolveRequest {
        SolveRequest {
            mode: SolverMode::ExpectedScrip,
            state: State {
                integrity: 4,
                max_integrity: 4,
                gp: 0,
                max_gp: 1000,
                remaining: 2,
                ..State::default()
            },
            rewards,
            actions: ActionModel {
                scour_gain: 500,
                meticulous_gain: 100,
                brazen_gains: vec![WeightedGain {
                    gain: 100,
                    weight: 1,
                }],
                scrutiny_cost: 200,
                focus_cost: 100,
                priming_cost: 400,
                solid_reason_cost: 300,
                scour: true,
                brazen: false,
                meticulous: false,
                scrutiny: false,
                collectors_focus: false,
                priming_touch: false,
                solid_reason: false,
                wise_to_the_world: false,
            },
            mechanics: Mechanics {
                gather_success_bp: 10_000,
                intuition_bp: 0,
                focus_intuition_bp: 10_000,
                intuition_gain: 100,
                standard_proc_bp: 0,
                high_standard_upgrade_bp: 0,
                meticulous_preserve_bp: 0,
                high_standard_preserve_bonus_bp: 0,
                priming_preserve_multiplier: 2,
                solid_reason_eureka_bp: 5_000,
                revisit_bp: 0,
                collect_gp_regen: 6,
                scrutiny_gain_multiplier_bp: 20_000,
                max_states: 10_000,
            },
            legacy: LegacyOptions {
                target_score: 500,
                minimum_score: 500,
                use_full_rotation: true,
                always_use_solid_reason: true,
                abandon_when_complete: true,
            },
            unsupported_reason: None,
        }
    }

    #[test]
    fn two_lower_tier_collects_beat_one_high_tier_collect() {
        let decision = solve(&request(vec![
            RewardTier {
                threshold: 500,
                scrip: 60,
            },
            RewardTier {
                threshold: 1000,
                scrip: 100,
            },
        ]))
        .unwrap();
        assert_eq!(decision.action, GatheringAction::Scour);
        assert!((decision.expected_reward - 120.0).abs() < 1e-9);
    }

    #[test]
    fn high_payout_can_justify_consuming_both_integrity() {
        let decision = solve(&request(vec![
            RewardTier {
                threshold: 500,
                scrip: 40,
            },
            RewardTier {
                threshold: 1000,
                scrip: 100,
            },
        ]))
        .unwrap();
        assert_eq!(decision.action, GatheringAction::Scour);
        assert!((decision.expected_reward - 100.0).abs() < 1e-9);
    }

    #[test]
    fn exact_preservation_probability_changes_expected_yield() {
        let mut input = request(vec![RewardTier {
            threshold: 100,
            scrip: 10,
        }]);
        input.actions.scour = false;
        input.actions.meticulous = true;
        input.mechanics.meticulous_preserve_bp = 5_000;
        input.state.remaining = 3;
        let decision = solve(&input).unwrap();
        input.mechanics.meticulous_preserve_bp = 0;
        let without_preservation = solve(&input).unwrap();
        assert_eq!(decision.action, GatheringAction::Meticulous);
        assert!(
            decision.expected_reward > without_preservation.expected_reward,
            "with={} without={}",
            decision.expected_reward,
            without_preservation.expected_reward
        );
    }

    #[test]
    fn maximize_collectability_prefers_a_perfect_collect_over_two_lower_collects() {
        let mut input = request(vec![RewardTier {
            threshold: 500,
            scrip: 1,
        }]);
        input.mode = SolverMode::MaximizeCollectability;
        input.state.collectability = 500;
        input.state.integrity = 3;
        input.state.max_integrity = 3;
        input.state.remaining = 2;
        let decision = solve(&input).unwrap();
        assert_eq!(decision.action, GatheringAction::Scour);
        assert!((decision.expected_perfect_collects - 1.0).abs() < 1e-9);
        assert!((decision.expected_reward - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn maximize_collectability_collects_the_best_available_subcap_result() {
        let mut input = request(vec![RewardTier {
            threshold: 500,
            scrip: 1,
        }]);
        input.mode = SolverMode::MaximizeCollectability;
        input.state.collectability = 700;
        input.state.integrity = 1;
        input.state.max_integrity = 1;
        input.state.remaining = 1;
        input.unsupported_reason = Some(String::from("aetherial reduction collectable"));
        let decision = solve(&input).unwrap();
        assert_eq!(decision.solver_used, SolverMode::MaximizeCollectability);
        assert_eq!(decision.action, GatheringAction::Collect);
        assert_eq!(decision.expected_perfect_collects, 0.0);
        assert!((decision.expected_reward - 700.0).abs() < 1e-9);
    }

    #[test]
    fn unsupported_reward_model_falls_back_to_rust_legacy() {
        let mut input = request(vec![RewardTier {
            threshold: 500,
            scrip: 60,
        }]);
        input.unsupported_reason = Some(String::from("custom delivery"));
        let decision = solve(&input).unwrap();
        assert_eq!(decision.solver_used, SolverMode::Legacy);
        assert_eq!(decision.fallback_reason.as_deref(), Some("custom delivery"));
    }
}
