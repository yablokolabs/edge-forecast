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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_fit_and_predict() {
        let mut s = SpinForecaster::default();
        let series: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        s.fit(&series).unwrap();
        let ctx = ForecastWindow::new(series[15..].to_vec());
        let next = s.predict_next(&ctx).unwrap();
        assert!(next.is_finite());
    }

    #[test]
    fn spin_fit_rejects_short_series() {
        let mut s = SpinForecaster::default();
        assert!(s.fit(&[1.0]).is_err());
    }

    #[test]
    fn spin_predict_on_empty_context_fails() {
        let s = SpinForecaster::default();
        assert!(s.predict_next(&ForecastWindow::new(vec![])).is_err());
    }

    #[test]
    fn spin_encode_bounded() {
        let s = SpinForecaster::default();
        let large: Vec<f64> = (0..100).map(|i| i as f64 * 100.0).collect();
        let encoded = s.encode(&large);
        // mean of three tanh values, bounded in [-1, 1]
        assert!(encoded.abs() <= 1.0);
    }

    #[test]
    fn spin_model_state_variant() {
        let s = SpinForecaster::default();
        assert!(matches!(s.model_state(), ModelState::Spin { .. }));
    }

    #[test]
    fn spin_forecast_multi_horizon() {
        let mut s = SpinForecaster::default();
        let series: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        s.fit(&series).unwrap();
        let ctx = ForecastWindow::new(series[15..].to_vec());
        let result = s.forecast(&ctx, 5).unwrap();
        assert_eq!(result.predictions.len(), 5);
        for p in &result.predictions {
            assert!(p.is_finite());
        }
    }
}
