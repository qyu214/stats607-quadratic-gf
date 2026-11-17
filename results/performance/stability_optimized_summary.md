# Optimized stability check

Low-rank Project 3 solver: RK45, rtol=1e-6, atol=1e-8.
Tested 11 trajectories covering 11 parameter conditions (1 initialization(s) per condition).

- Solver/finite-value failures: **0**
- Cases requiring review (warnings or risk diagnostics): **0**
- Captured Python warning messages: **0**
- Largest positive change between adjacent saved risks: **0**
- Smallest observed risk minus theoretical floor: **2.00272e-05**

| Study | d | Width | beta | Rep | Status | Warnings | Risk increases* | Floor violations* | Tight risk diff |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| main | 32 | 8 | 0.1 | 0 | PASS | 0 | 0 | 0 |  |
| main | 32 | 8 | 0.35 | 0 | PASS | 0 | 0 | 0 |  |
| main | 32 | 8 | 0.6 | 0 | PASS | 0 | 0 | 0 |  |
| main | 32 | 16 | 0.1 | 0 | PASS | 0 | 0 | 0 |  |
| main | 32 | 16 | 0.35 | 0 | PASS | 0 | 0 | 0 | 1.48e-06 |
| main | 32 | 16 | 0.6 | 0 | PASS | 0 | 0 | 0 |  |
| fixed_alpha | 32 | 16 | 0.25 | 0 | PASS | 0 | 0 | 0 |  |
| fixed_alpha | 32 | 16 | 0.55 | 0 | PASS | 0 | 0 | 0 |  |
| width256 | 256 | 32 | 0.35 | 0 | PASS | 0 | 0 | 0 |  |
| width256 | 256 | 64 | 0.35 | 0 | PASS | 0 | 0 | 0 | 6.64e-06 |
| width256 | 256 | 128 | 0.35 | 0 | PASS | 0 | 0 | 0 |  |

*For the two risk diagnostics, a step is flagged only when its size exceeds 1e-8 + 2e-6 * max(1, |initial risk|). The raw changes are in the CSV.

The script checks finite risk and centering-mismatch values at saved time points. The optimized solver raises an exception on unsuccessful integration; this script does not inspect every intermediate internal solver state.

Warning counts reflect captured Python warnings for these runs, not all possible numerical conditions. A small difference from the tighter-tolerance solution is a sensitivity diagnostic, not a proof of solution accuracy.

Tight tolerance (rtol=1e-8, atol=1e-10) was tested for one d=32/width=16 and one d=256/width=64 trajectory.
