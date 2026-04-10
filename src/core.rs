use serde::{Deserialize, Serialize};

/// Input context for forecasting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastWindow {
    pub values: Vec<f64>,
}

impl ForecastWindow {
    #[must_use]
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Forecast output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastResult {
    pub predictions: Vec<f64>,
}

impl ForecastResult {
    #[must_use]
    pub fn next(&self) -> Option<f64> {
        self.predictions.first().copied()
    }
}

/// Serializable model state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelState {
    Autoregressive {
        coefficient: f64,
        bias: f64,
    },
    Reservoir {
        input_scale: f64,
        recurrence: f64,
        readout_scale: f64,
        bias: f64,
    },
    Spin {
        coupling: f64,
        memory: f64,
        nonlinearity: f64,
        readout_scale: f64,
        bias: f64,
    },
}

/// Common forecasting interface.
pub trait Forecaster {
    fn fit(&mut self, series: &[f64]) -> anyhow::Result<()>;
    fn predict_next(&self, context: &ForecastWindow) -> anyhow::Result<f64>;
    fn model_state(&self) -> ModelState;

    fn forecast(&self, context: &ForecastWindow, horizon: usize) -> anyhow::Result<ForecastResult> {
        let mut window = context.clone();
        let mut predictions = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let next = self.predict_next(&window)?;
            predictions.push(next);
            window.values.push(next);
        }
        Ok(ForecastResult { predictions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecast_window_len_and_empty() {
        let empty = ForecastWindow::new(vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let non_empty = ForecastWindow::new(vec![1.0, 2.0]);
        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.len(), 2);
    }

    #[test]
    fn forecast_result_next() {
        let empty = ForecastResult {
            predictions: vec![],
        };
        assert_eq!(empty.next(), None);

        let result = ForecastResult {
            predictions: vec![3.15, 2.72],
        };
        assert_eq!(result.next(), Some(3.15));
    }

    #[test]
    fn forecast_window_preserves_values() {
        let values = vec![1.0, 2.0, 3.0];
        let window = ForecastWindow::new(values.clone());
        assert_eq!(window.values, values);
    }
}
