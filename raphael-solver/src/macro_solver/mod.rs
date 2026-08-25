mod pareto_front;
mod search_queue;
mod solver;

pub use solver::{MacroSolveOutcome, MacroSolver, MacroSolverStats, SolveProgressSignal};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum FirstActionPreference {
    Progress,
    #[default]
    Other,
    Quality,
}

impl FirstActionPreference {
    fn strictly_preferred_to(self, other: Self) -> bool {
        matches!((self, other), (Self::Quality, Self::Progress))
    }
}
