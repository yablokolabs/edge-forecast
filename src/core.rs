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
        let mut history = context.values.clone();
        let mut predictions = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let next = self.predict_next(&ForecastWindow::new(history.clone()))?;
            predictions.push(next);
            history.push(next);
        }
        Ok(ForecastResult { predictions })
    }
}
