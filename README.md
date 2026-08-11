# Donatello

Donatello is the crafting solver used by
[GatherBuddy Ascended](https://github.com/slobodaapl/GatherBuddyAscended). It builds on Raphael's
global optimizer and adds live-state replanning for in-game crafting.

It can restart the solve from the craft's current CP, durability, progress, quality, effects,
combo, specialist charges, and observed condition. This lets GatherBuddy Ascended react to expert
craft conditions and unexpected state changes while optimizing the complete remaining craft.

Donatello ships with GatherBuddy Ascended. Installation, releases, documentation, and issues are
handled in the [GatherBuddy Ascended repository](https://github.com/slobodaapl/GatherBuddyAscended).

## Workspace

- `donatello-ffi` — in-process native interface used by GatherBuddy Ascended.
- `raphael-solver` / `raphael-sim` — optimizer and simulator.
- `raphael-cli` — compatibility and development tooling.
- `donatello-bench` — reproducible Raphael comparison benchmark.

Plans are ranked by completed quality, then action count, then duration. GatherBuddy Ascended
simulates replans with Vulcan before switching the active craft.

## Building

Rust 1.92.0 or newer is required.

```text
cargo build --locked --release --package donatello-ffi --package raphael-cli
cargo test --locked --workspace
```

GatherBuddy Ascended pins Donatello as a Git submodule and packages the matching
`donatello_ffi.dll` automatically.

## Credits

Donatello is based on [Raphael XIV](https://github.com/KonaeAkira/raphael-rs) by KonaeAkira and its
contributors. Upstream attribution and licenses remain in this repository.
