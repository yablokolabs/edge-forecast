# edge-forecast

A compact Rust forecasting engine for edge and streaming time-series.

## What it is
`edge-forecast` is a practical forecasting crate focused on:
- short-horizon time-series prediction
- small deployable models
- low-latency inference
- streaming / edge-friendly operation
- mathematically principled behavior with a Lean specification companion

## What it includes
- autoregressive baseline forecaster
- compact reservoir forecaster
- spin-inspired temporal forecaster
- CSV-based train/eval flow
- forecasting metrics
- Lean spec for core mathematical principles

## What it is not
- a reproduction of quantum hardware experiments
- a claim of quantum advantage
- full formal verification of floating-point Rust code

## Initial direction
This repo takes inspiration from compact high-memory temporal dynamics and turns that idea into a useful Rust-first forecasting engine.

## Practical usefulness
`edge-forecast` is intended to be useful when you need forecasting that is:
- small and deployable
- fast enough for low-latency environments
- easier to operate than a large deep-learning stack
- good for streaming or edge-style time-series workloads

### Who this can help
- platform / infra teams forecasting load, latency, or saturation
- IoT and telemetry teams forecasting sensor streams
- industrial monitoring teams watching process signals
- developers building lightweight anomaly detection from forecast residuals
- researchers and engineers who want compact temporal models in Rust without heavy ML infrastructure

### Practical scenarios
- short-horizon CPU / memory / queue forecasting
- sensor drift or anomaly detection through prediction error
- forecasting operational metrics on resource-constrained systems
- embedding compact predictors into Rust services, agents, or edge pipelines

## Planned CLI
- `edge-forecast train`
- `edge-forecast forecast`
- `edge-forecast eval`

## Lean
The Lean side is intended to specify and prove core mathematical properties such as:
- nonnegativity of error metrics
- monotonicity of aggregate loss under smaller residuals
- simple state/update semantics at the specification layer
