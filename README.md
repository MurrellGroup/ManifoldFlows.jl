# <img width="25%" src="./assets/logo.svg" align="right" /> ManifoldFlows.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/ManifoldFlows.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/ManifoldFlows.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/ManifoldFlows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/ManifoldFlows.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/ManifoldFlows.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/ManifoldFlows.jl)

ManifoldFlows.jl allows you to build [Flow Matching](https://arxiv.org/pdf/2302.00482) models, but where the objects you're modeling [exist on manifolds](https://arxiv.org/abs/2302.03660). ManifoldFlows.jl does not try to be a faithful replication of anything specific, but instead cobbles together a collection of tricks that have (sometimes) worked for us, in our use cases. We currently support Euclidean space, rotations, and the probability simplex, piggybacking off [Manifolds.jl](https://github.com/JuliaManifolds/Manifolds.jl) much of the time.

The basic idea is that you can train a model (typically a deep neural network) to interpolate between a simple distribution (that you can sample from) and a complex distribution (that you only have training examples from).

This is a visualization of a [toy example](./examples/spiral.jl) showing: i) the initial samples ($X_t$, blue) and how they change from their base distribution (a Gaussian) at $t=0$, to the spiral target distribution at $t=1$. Also shown is the model's estimate of the end state ($X_1 | X_t$, red).

<img src="./assets/spiral_animation.gif" width="350">

The bahaviour differs depending on whether the target and base samples are paired randomly during training, or paired via optimal transport:

<img src="./assets/spiral_animation_OT.gif" width="350">