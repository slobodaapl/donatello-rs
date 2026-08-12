#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Condition {
    Normal,
    Good,
    Excellent,
    Poor,
    Centered,
    Sturdy,
    Pliant,
    Malleable,
    Primed,
    GoodOmen,
    Robust,
}

impl Condition {
    #[must_use]
    pub const fn deterministic_successor(self) -> Self {
        match self {
            Self::Excellent => Self::Poor,
            Self::GoodOmen => Self::Good,
            Self::Robust => Self::Sturdy,
            _ => Self::Normal,
        }
    }
}
