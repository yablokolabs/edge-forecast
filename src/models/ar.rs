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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ar_fit_on_linear_series() {
        let mut ar = AutoregressiveForecaster::default();
        // y[t+1] = y[t] + 1, so coefficient ≈ 1.0, bias ≈ 1.0
        let series: Vec<f64> = (0..20).map(|i| i as f64).collect();
        ar.fit(&series).unwrap();
        assert_relative_eq!(ar.coefficient, 1.0, epsilon = 1e-10);
        assert_relative_eq!(ar.bias, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn ar_predict_next_returns_linear_extrapolation() {
        let mut ar = AutoregressiveForecaster::default();
        let series: Vec<f64> = (0..20).map(|i| i as f64).collect();
        ar.fit(&series).unwrap();
        let ctx = ForecastWindow::new(vec![18.0, 19.0]);
        let next = ar.predict_next(&ctx).unwrap();
        assert_relative_eq!(next, 20.0, epsilon = 1e-6);
    }

    #[test]
    fn ar_fit_rejects_short_series() {
        let mut ar = AutoregressiveForecaster::default();
        assert!(ar.fit(&[1.0]).is_err());
        assert!(ar.fit(&[]).is_err());
    }

    #[test]
    fn ar_predict_on_empty_context_fails() {
        let ar = AutoregressiveForecaster::default();
        let ctx = ForecastWindow::new(vec![]);
        assert!(ar.predict_next(&ctx).is_err());
    }

    #[test]
    fn ar_constant_series_uses_fallback() {
        let mut ar = AutoregressiveForecaster::default();
        ar.fit(&[5.0, 5.0, 5.0, 5.0]).unwrap();
        // denominator is zero, coefficient falls back to 1.0
        assert_relative_eq!(ar.coefficient, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn ar_model_state_roundtrip() {
        let ar = AutoregressiveForecaster {
            coefficient: 0.95,
            bias: 0.3,
        };
        match ar.model_state() {
            ModelState::Autoregressive { coefficient, bias } => {
                assert_relative_eq!(coefficient, 0.95);
                assert_relative_eq!(bias, 0.3);
            }
            _ => panic!("wrong variant"),
        }
    }
}
