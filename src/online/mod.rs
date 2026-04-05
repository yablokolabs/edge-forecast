use serde::{Deserialize, Serialize};

use crate::core::ForecastWindow;

/// Rolling stream state for online inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnlineState {
    pub window_size: usize,
    pub values: Vec<f64>,
}

impl OnlineState {
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: Vec::with_capacity(window_size),
        }
    }

    pub fn push(&mut self, value: f64) {
        self.values.push(value);
        if self.values.len() > self.window_size {
            self.values.remove(0);
        }
    }

    #[must_use]
    pub fn window(&self) -> ForecastWindow {
        ForecastWindow::new(self.values.clone())
    }
}
