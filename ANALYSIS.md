# Project 2 — Critical Analysis

## Design choices

I used a quadratic teacher–student model with Gaussian features and aligned diagonal covariance. I chose this setting because the population risk and the optimal rank-constrained risk can be computed explicitly, which gives me a reference for checking the simulation results.

In the main experiment, I fixed $\alpha+\beta=0.7$ and varied $\beta$ and student width. This keeps the theoretical rank floor unchanged across $\beta$ for a given width, making it easier to compare the learning dynamics. I used $d=32$ and ten random initializations per condition to keep the simulation manageable. I also ran two follow-up experiments: a width comparison at $d=256$ with five initializations per condition, and a fixed-$\alpha$ experiment to check how the results depend on the original parameter constraint. The larger-dimensional runs were more computationally expensive, so I used fewer repetitions.

## Fairness of the comparisons

I used the same risk definition, solver settings, and bias-update timescale across the experiments. When comparing different $\beta$ values at a fixed dimension and width, I reused the same random initializations to reduce unnecessary variation. I also compared the risk with the theoretical rank floor, since different widths can have different approximation limits.

However, the comparisons are not completely controlled. Increasing width also changes the number of parameters, while fixing $\alpha+\beta$ means that $\alpha$ changes with $\beta$. Even when $\alpha$ is fixed, the normalized teacher still changes with $\beta$. These differences need to be considered when interpreting the results.

## Results and interpretation

In the main experiment, the risk-to-floor gap at $t=80$ increased with $\beta$ for both student widths, even though the rank floor was unchanged. In the $d=256$ experiment, increasing width from 32 to 128 reduced the mean final risk from 0.1243 to 0.0377, but the gap above the corresponding rank floor increased from 0.0010 to 0.0071.

The fixed-$\alpha$ experiment showed a similar pattern: increasing $\beta$ reduced the rank floor while slightly increasing the finite-time gap. However, these differences were relatively small compared with Monte Carlo uncertainty.

These results suggest that lower population risk does not necessarily mean a network is closer to its best rank-constrained approximation. This distinction is useful when comparing networks with different capacities. The simulations are also consistent with the idea that spectral structure affects learning dynamics, although they do not establish a general scaling law.

## Limitations and missing scenarios

The model is quite idealized. I assumed Gaussian features, aligned diagonal teacher and covariance matrices, and no observation noise. I also simulated population gradient flow rather than finite-sample SGD. Non-aligned matrices could introduce interactions between spectral directions, while finite-sample training and observation noise could change the learning trajectories.

The experiments used only five or ten random initializations per condition. The reported Monte Carlo uncertainty reflects variation in initialization but does not account for numerical integration error. The results are therefore limited to the specific settings I considered.

## Implementation and next steps

The main implementation challenges were matching the risk normalization and gradient-flow time scale, handling the trainable bias, and using the correct rank floor in the fixed-$\alpha$ experiment. I checked the gradients numerically and included tests for reproducibility and the theoretical floor. These checks give me some confidence in the implementation, although they cannot rule out all numerical errors.

If I had more time, I would first check sensitivity to the ODE solver tolerances and use more initializations. I would then consider non-aligned covariance structures and finite-sample SGD to see whether the same qualitative behavior persists.