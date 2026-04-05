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
