use std::fs;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::core::ModelState;
use crate::models::{AutoregressiveForecaster, ReservoirForecaster, SpinForecaster};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedModel {
    pub model_name: String,
    pub version: String,
    pub window_size: usize,
    pub columns: Vec<usize>,
    pub state: ModelState,
}

pub fn save_model(path: &str, model: &SavedModel) -> Result<()> {
    let json = serde_json::to_string_pretty(model)?;
    fs::write(path, json)?;
    Ok(())
}

pub fn load_model(path: &str) -> Result<SavedModel> {
    let json = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json)?)
}

pub fn build_forecaster_from_state(state: &ModelState) -> Box<dyn crate::core::Forecaster> {
    match state {
        ModelState::Autoregressive { coefficient, bias } => Box::new(AutoregressiveForecaster {
            coefficient: *coefficient,
            bias: *bias,
        }),
        ModelState::Reservoir {
            input_scale,
            recurrence,
            readout_scale,
            bias,
        } => Box::new(ReservoirForecaster {
            input_scale: *input_scale,
            recurrence: *recurrence,
            readout_scale: *readout_scale,
            bias: *bias,
        }),
        ModelState::Spin {
            coupling,
            memory,
            nonlinearity,
            readout_scale,
            bias,
        } => Box::new(SpinForecaster {
            coupling: *coupling,
            memory: *memory,
            nonlinearity: *nonlinearity,
            readout_scale: *readout_scale,
            bias: *bias,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ForecastWindow;

    fn roundtrip(state: ModelState) {
        let saved = SavedModel {
            model_name: "test".to_string(),
            version: "0.1.0".to_string(),
            window_size: 10,
            columns: vec![0],
            state,
        };
        let json = serde_json::to_string(&saved).unwrap();
        let loaded: SavedModel = serde_json::from_str(&json).unwrap();
        assert_eq!(saved, loaded);
    }

    #[test]
    fn roundtrip_ar() {
        roundtrip(ModelState::Autoregressive {
            coefficient: 0.95,
            bias: 0.1,
        });
    }

    #[test]
    fn roundtrip_reservoir() {
        roundtrip(ModelState::Reservoir {
            input_scale: 0.8,
            recurrence: 0.6,
            readout_scale: 1.0,
            bias: 0.05,
        });
    }

    #[test]
    fn roundtrip_spin() {
        roundtrip(ModelState::Spin {
            coupling: 0.7,
            memory: 0.5,
            nonlinearity: 1.2,
            readout_scale: 0.8,
            bias: 0.02,
        });
    }

    #[test]
    fn build_forecaster_from_ar_state() {
        let state = ModelState::Autoregressive {
            coefficient: 1.0,
            bias: 0.0,
        };
        let f = build_forecaster_from_state(&state);
        let ctx = ForecastWindow::new(vec![1.0, 2.0, 3.0]);
        let next = f.predict_next(&ctx).unwrap();
        assert!(next.is_finite());
    }

    #[test]
    fn save_and_load_file() {
        let saved = SavedModel {
            model_name: "ar".to_string(),
            version: "0.1.0".to_string(),
            window_size: 10,
            columns: vec![0],
            state: ModelState::Autoregressive {
                coefficient: 0.95,
                bias: 0.1,
            },
        };
        let path = "/tmp/edge-forecast-test-model.json";
        save_model(path, &saved).unwrap();
        let loaded = load_model(path).unwrap();
        assert_eq!(saved, loaded);
        std::fs::remove_file(path).ok();
    }
}
