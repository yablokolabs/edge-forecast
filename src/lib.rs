pub mod core;
#[cfg(feature = "cli")]
pub mod data;
pub mod metrics;
pub mod model_io;
pub mod models;
pub mod online;

pub use core::{ForecastResult, ForecastWindow, Forecaster, ModelState};
pub use models::{AutoregressiveForecaster, ReservoirForecaster, SpinForecaster};
