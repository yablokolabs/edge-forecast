use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::core::ForecastWindow;

/// Rolling stream state for online inference.
///
/// Maintains a fixed-size sliding window using a `VecDeque` for O(1)
/// push and eviction — suitable for high-throughput streaming workloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnlineState {
    pub window_size: usize,
    pub values: VecDeque<f64>,
}

impl OnlineState {
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: VecDeque::with_capacity(window_size),
        }
    }

    /// Push a new value, evicting the oldest if the window is full.
    pub fn push(&mut self, value: f64) {
        if self.values.len() >= self.window_size {
            self.values.pop_front();
        }
        self.values.push_back(value);
    }

    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.values.len() >= self.window_size
    }

    #[must_use]
    pub fn window(&self) -> ForecastWindow {
        ForecastWindow::new(self.values.iter().copied().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_within_capacity() {
        let mut state = OnlineState::new(5);
        state.push(1.0);
        state.push(2.0);
        assert_eq!(state.values.len(), 2);
        assert!(!state.is_ready());
    }

    #[test]
    fn push_evicts_oldest_at_capacity() {
        let mut state = OnlineState::new(3);
        state.push(1.0);
        state.push(2.0);
        state.push(3.0);
        assert!(state.is_ready());
        state.push(4.0);
        let w = state.window();
        assert_eq!(w.values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn window_returns_correct_snapshot() {
        let mut state = OnlineState::new(3);
        for v in [10.0, 20.0, 30.0] {
            state.push(v);
        }
        let w = state.window();
        assert_eq!(w.values, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn is_ready_becomes_true_at_capacity() {
        let mut state = OnlineState::new(2);
        assert!(!state.is_ready());
        state.push(1.0);
        assert!(!state.is_ready());
        state.push(2.0);
        assert!(state.is_ready());
    }
}
