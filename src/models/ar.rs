use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::core::{ForecastWindow, Forecaster, ModelState};

/// Simple AR(1)-style baseline forecaster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoregressiveForecaster {
    pub coefficient: f64,
    pub bias: f64,
}

impl Default for AutoregressiveForecaster {
    fn default() -> Self {
        Self {
            coefficient: 1.0,
            bias: 0.0,
        }
    }
}

impl Forecaster for AutoregressiveForecaster {
    fn fit(&mut self, series: &[f64]) -> Result<()> {
        if series.len() < 2 {
            return Err(anyhow!("series must contain at least 2 values"));
        }
        let xs = &series[..series.len() - 1];
        let ys = &series[1..];
        let x_mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
        let numerator: f64 = xs
            .iter()
            .zip(ys)
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();
        let denominator: f64 = xs.iter().map(|x| (x - x_mean).powi(2)).sum();
        self.coefficient = if denominator.abs() < f64::EPSILON {
            1.0
        } else {
            numerator / denominator
        };
        self.bias = y_mean - self.coefficient * x_mean;
        Ok(())
    }

    fn predict_next(&self, context: &ForecastWindow) -> Result<f64> {
        let last = context
            .values
            .last()
            .copied()
            .ok_or_else(|| anyhow!("forecast context is empty"))?;
        Ok(self.coefficient * last + self.bias)
    }

    fn model_state(&self) -> ModelState {
        ModelState::Autoregressive {
            coefficient: self.coefficient,
            bias: self.bias,
        }
    }
}
