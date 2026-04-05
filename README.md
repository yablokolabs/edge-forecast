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
- multivariate input support via multiple CSV columns
- model metadata + saved config/state
- lightweight HTTP forecast service mode
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

## CLI
- `edge-forecast train`
- `edge-forecast forecast`
- `edge-forecast eval`
- `edge-forecast score`

## Usage examples
### Train a spin-inspired model
```bash
cargo run --bin edge-forecast -- train \
  --input examples/sample.csv \
  --columns 0 \
  --model spin \
  --output /tmp/edge-forecast-model.json
```

### Forecast the next 3 points from a saved model
```bash
cargo run --bin edge-forecast -- forecast \
  --input examples/sample.csv \
  --model-file /tmp/edge-forecast-model.json \
  --horizon 3
```

### Evaluate a compact reservoir model
```bash
cargo run --bin edge-forecast -- eval \
  --input examples/sample.csv \
  --columns 0 \
  --model reservoir
```

### Score likely anomalies from forecast residuals
```bash
cargo run --bin edge-forecast -- score \
  --input examples/with_anomaly.csv \
  --columns 0 \
  --model spin \
  --top-k 3
```

### Train with multiple input columns
```bash
cargo run --bin edge-forecast -- train \
  --input examples/multivariate.csv \
  --columns 0,1 \
  --model reservoir \
  --output /tmp/multivariate-model.json
```

### Serve forecasts over HTTP
```bash
cargo run --bin edge-forecast -- serve \
  --model-file /tmp/multivariate-model.json \
  --bind 127.0.0.1:8080
```

Then POST:
```bash
curl -X POST http://127.0.0.1:8080/forecast \
  -H 'content-type: application/json' \
  -d '{"context":[0.75,0.80,0.84,0.88,0.91,0.95,0.99,1.02,1.04,1.08],"horizon":3}'
```

## Why this is practical
You can use `edge-forecast` when you want:
- a small forecasting engine embedded into a Rust service
- lightweight telemetry forecasting without a heavy ML stack
- anomaly scoring from prediction residuals
- an edge-friendly baseline before jumping to larger sequence models

## Lean
The Lean side is intended to specify and prove core mathematical properties such as:
- nonnegativity of error metrics
- monotonicity of aggregate loss under smaller residuals
- simple state/update semantics at the specification layer
