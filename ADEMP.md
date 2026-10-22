# STATS 607 Project 2 — Simulation Design (ADEMP)

This is an original simulation study based on the quadratic teacher–student model from my research. I use population gradient flow rather than the online SGD studied in the motivating paper. The goal is to understand how the covariance spectrum and student width affect learning over a finite time interval.

## A — Aims

I want to distinguish two reasons that neural networks may have different population risks: they may have different approximation limits, or they may be different distances from those limits after the same amount of training time. I use the theoretical rank-constrained risk floor to separate these effects.

The main experiment changes the covariance exponent $\beta$ and student width while keeping $\alpha+\beta=0.7$. I then add two exploratory comparisons: increasing width at $d=256$, and varying $\beta$ while keeping $\alpha=0.35$ fixed.

## D — Data-generating mechanism and conditions

Let $x\sim N(0,\Sigma)$, where

$$
\Sigma=\mathrm{diag}(i^{-\beta})_{i=1}^d,\qquad
T=\mathrm{diag}(i^{-\alpha})_{i=1}^d,\qquad
B=\frac{T}{\|T\Sigma\|_F}.
$$

The centered teacher response is

$$
y=x^\top Bx-\mathrm{tr}(\Sigma B).
$$

I do not add observation noise. Rather than sampling a finite training dataset, I evaluate the Gaussian population expectations analytically. The student model is

$$
\hat y=x^\top\frac{WW^\top}{r_s}x-b,\qquad W\in\mathbb R^{d\times r_s}.
$$

I draw $W_{0,ij}\sim N(0,1/d)$ and $b_0\sim N(0,1)$. The master seed is `20261019`, with random streams based on `SeedSequence([20261019, d, r_s, replicate])`. At a fixed dimension, width, and replicate ID, the same initialization is reused across different values of $\beta$.

### Simulation design

| Experiment | $d$ | $\alpha$ | $\beta$ | Student width $r_s$ | Replicates | End time |
|---|---:|---:|---:|---:|---:|---:|
| Main | 32 | 0.60 | 0.10 | 8 | 10 | 80 |
| Main | 32 | 0.35 | 0.35 | 8 | 10 | 80 |
| Main | 32 | 0.10 | 0.60 | 8 | 10 | 80 |
| Main | 32 | 0.60 | 0.10 | 16 | 10 | 80 |
| Main | 32 | 0.35 | 0.35 | 16 | 10 | 80 |
| Main | 32 | 0.10 | 0.60 | 16 | 10 | 80 |
| Width follow-up | 256 | 0.35 | 0.35 | 32 | 5 | 160 |
| Width follow-up | 256 | 0.35 | 0.35 | 64 | 5 | 160 |
| Width follow-up | 256 | 0.35 | 0.35 | 128 | 5 | 160 |
| Fixed-$\alpha$ follow-up | 32 | 0.35 | 0.25 | 16 | 10 | 80 |
| Fixed-$\alpha$ follow-up | 32 | 0.35 | 0.35 | 16 | 10 | 80 |
| Fixed-$\alpha$ follow-up | 32 | 0.35 | 0.55 | 16 | 10 | 80 |

This gives 60 main runs, 15 width runs, and 30 fixed-$\alpha$ runs. The $(d,\alpha,\beta,r_s)=(32,0.35,0.35,16)$ condition uses the same ten initializations in the main and fixed-$\alpha$ experiments, so the 105 executed trajectories include ten repeated baseline trajectories. The main and fixed-$\alpha$ studies save 75 time points up to $t=80$; the width follow-up saves 101 time points up to $t=160$. Exact settings are stored in `config.json` and the files under `configs/`.

## E — Estimands

With $M=WW^\top/r_s$, the population excess risk is

$$
R(W,b)=\|\Sigma^{1/2}(M-B)\Sigma^{1/2}\|_F^2
+\frac12\bigl(b-\mathrm{tr}(\Sigma M)\bigr)^2.
$$

For each condition and time point, I estimate the mean risk over randomized initializations. I also calculate the best possible rank-constrained risk in this aligned model:

$$
R_{\mathrm{floor}}(d,r_s;\alpha,\beta)
=\frac{\sum_{i=r_s+1}^{d}i^{-2(\alpha+\beta)}}
{\sum_{i=1}^{d}i^{-2(\alpha+\beta)}}.
$$

The finite-time gap is $G(t)=R(W_t,b_t)-R_{\mathrm{floor}}$. In the main experiment the floor is constant across $\beta$ for a given width. In the fixed-$\alpha$ experiment it varies with $\beta$.

## M — Methods

I numerically solve the full-matrix, two-timescale population gradient flow:

$$
\dot W=-r_s\nabla_W R(W,b),\qquad
\dot b=-\delta^{-1}\partial_b R(W,b),\qquad \delta=0.05.
$$

I use SciPy `solve_ivp` (RK45), with `rtol=1e-6` and `atol=1e-8`. The implementation separates model setup, the flow equations, simulation, summaries, and plotting. Raw trajectories are saved before computing averages. Tests check the gradients numerically, reproducibility, and the rank-floor calculations.

## P — Performance measures

I plot mean population excess risk over time, with theoretical floor lines, and compare the risk-to-floor gaps at the selected time points. Where appropriate, plots show approximate pointwise 95% Monte Carlo intervals over initializations. I also inspect individual trajectories and the bias-centering mismatch as diagnostics.

The intervals measure variation across starting points, not ODE integration error. These experiments use relatively few initializations and an idealized Gaussian, aligned, noiseless population model. The two follow-ups are exploratory, and none of the results alone establishes an asymptotic law for online SGD.
