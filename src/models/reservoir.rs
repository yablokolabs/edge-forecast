use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::core::{ForecastWindow, Forecaster, ModelState};

/// Compact nonlinear reservoir forecaster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReservoirForecaster {
    pub input_scale: f64,
    pub recurrence: f64,
    pub readout_scale: f64,
    pub bias: f64,
}

impl Default for ReservoirForecaster {
    fn default() -> Self {
        Self {
            input_scale: 0.8,
            recurrence: 0.6,
            readout_scale: 1.0,
            bias: 0.0,
        }
    }
}

impl ReservoirForecaster {
    fn encode(&self, series: &[f64]) -> f64 {
        let mut state = 0.0;
        for &x in series {
            state = (self.input_scale * x + self.recurrence * state).tanh();
        }
        state
    }
}

impl Forecaster for ReservoirForecaster {
    fn fit(&mut self, series: &[f64]) -> Result<()> {
        if series.len() < 2 {
            return Err(anyhow!("series must contain at least 2 values"));
        }
        let residual_mean =
            series.windows(2).map(|w| w[1] - w[0]).sum::<f64>() / (series.len() - 1) as f64;
        self.bias = residual_mean;
        Ok(())
    }

    fn predict_next(&self, context: &ForecastWindow) -> Result<f64> {
        if context.is_empty() {
            return Err(anyhow!("forecast context is empty"));
        }
        let state = self.encode(&context.values);
        let last = *context.values.last().expect("checked non-empty context");
        Ok(last + self.readout_scale * state + self.bias)
    }

    fn model_state(&self) -> ModelState {
        ModelState::Reservoir {
            input_scale: self.input_scale,
            recurrence: self.recurrence,
            readout_scale: self.readout_scale,
            bias: self.bias,
        }
    }
}
