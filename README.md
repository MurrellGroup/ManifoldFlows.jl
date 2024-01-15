# <img width="25%" src="./assets/logo.svg" align="right" /> ManifoldFlows.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/ManifoldFlows.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/ManifoldFlows.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/ManifoldFlows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/ManifoldFlows.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/ManifoldFlows.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/ManifoldFlows.jl)

ManifoldFlows.jl allows you to build [Flow Matching](https://arxiv.org/pdf/2302.00482) models, but where the objects you're modeling [exist on manifolds](https://arxiv.org/abs/2302.03660). ManifoldFlows.jl does not try to be a faithful replication of anything specific, but instead cobbles together a collection of tricks that have (sometimes) worked for us, in our use cases. We currently support Euclidean space, rotations, and the probability simplex, piggybacking off [Manifolds.jl](https://github.com/JuliaManifolds/Manifolds.jl) much of the time.

<img src="./assets/spiral_animation.gif" width="350">