use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::core::{ForecastWindow, Forecaster, ModelState};

/// Spin-inspired temporal forecaster with compact coupled internal dynamics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpinForecaster {
    pub coupling: f64,
    pub memory: f64,
    pub nonlinearity: f64,
    pub readout_scale: f64,
    pub bias: f64,
}

impl Default for SpinForecaster {
    fn default() -> Self {
        Self {
            coupling: 0.7,
            memory: 0.5,
            nonlinearity: 1.2,
            readout_scale: 0.8,
            bias: 0.0,
        }
    }
}

impl SpinForecaster {
    fn encode(&self, series: &[f64]) -> f64 {
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        let mut s3 = 0.0;
        for &x in series {
            let n1 = (self.nonlinearity * x + self.memory * s1 + self.coupling * s2).tanh();
            let n2 = (self.nonlinearity * x + self.memory * s2 + self.coupling * s3).tanh();
            let n3 = (self.nonlinearity * x + self.memory * s3 + self.coupling * s1).tanh();
            s1 = n1;
            s2 = n2;
            s3 = n3;
        }
        (s1 + s2 + s3) / 3.0
    }
}

impl Forecaster for SpinForecaster {
    fn fit(&mut self, series: &[f64]) -> Result<()> {
        if series.len() < 2 {
            return Err(anyhow!("series must contain at least 2 values"));
        }
        self.bias = series.windows(2).map(|w| w[1] - w[0]).sum::<f64>() / (series.len() - 1) as f64;
        Ok(())
    }

    fn predict_next(&self, context: &ForecastWindow) -> Result<f64> {
        if context.is_empty() {
            return Err(anyhow!("forecast context is empty"));
        }
        let encoded = self.encode(&context.values);
        let last = *context.values.last().expect("checked non-empty context");
        Ok(last + self.readout_scale * encoded + self.bias)
    }

    fn model_state(&self) -> ModelState {
        ModelState::Spin {
            coupling: self.coupling,
            memory: self.memory,
            nonlinearity: self.nonlinearity,
            readout_scale: self.readout_scale,
            bias: self.bias,
        }
    }
}
