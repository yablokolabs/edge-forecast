# Rust ↔ Lean Alignment

This document explains what the Lean side is intended to formalize for `edge-forecast`.

## Rust concepts
- `ForecastWindow`
- `ForecastResult`
- `Forecaster`
- `ModelState`
- forecasting metrics (`mae`, `mse`, `rmse`)
- compact temporal update behavior

## Lean concepts
- signals and residuals
- absolute and squared error
- aggregate loss over horizons
- simple state-update semantics
- monotonicity of aggregate loss under smaller pointwise residuals

## Honest scope
Lean currently supports the mathematical principles/story.
It does **not** prove exact floating-point equivalence with the Rust implementation.
