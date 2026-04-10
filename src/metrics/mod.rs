/// Mean absolute error. Panics if slices have different lengths.
#[must_use]
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(
        actual.len(),
        predicted.len(),
        "mae: actual and predicted must have the same length"
    );
    if actual.is_empty() {
        return 0.0;
    }
    actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / actual.len() as f64
}

/// Mean squared error. Panics if slices have different lengths.
#[must_use]
pub fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(
        actual.len(),
        predicted.len(),
        "mse: actual and predicted must have the same length"
    );
    if actual.is_empty() {
        return 0.0;
    }
    actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / actual.len() as f64
}

/// Root mean squared error. Panics if slices have different lengths.
#[must_use]
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    mse(actual, predicted).sqrt()
}

/// Per-element residuals (actual - predicted). Panics if slices have different lengths.
#[must_use]
pub fn residuals(actual: &[f64], predicted: &[f64]) -> Vec<f64> {
    assert_eq!(
        actual.len(),
        predicted.len(),
        "residuals: actual and predicted must have the same length"
    );
    actual.iter().zip(predicted).map(|(a, p)| a - p).collect()
}

/// Per-element absolute residuals. Panics if slices have different lengths.
#[must_use]
pub fn anomaly_scores(actual: &[f64], predicted: &[f64]) -> Vec<f64> {
    residuals(actual, predicted)
        .into_iter()
        .map(f64::abs)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn error_metrics_are_nonnegative() {
        let actual = [1.0, 2.0, 3.0];
        let predicted = [1.5, 2.5, 2.0];
        assert!(mae(&actual, &predicted) >= 0.0);
        assert!(mse(&actual, &predicted) >= 0.0);
        assert!(rmse(&actual, &predicted) >= 0.0);
    }

    #[test]
    fn perfect_prediction_has_zero_error() {
        let actual = [1.0, 2.0, 3.0];
        assert_relative_eq!(mae(&actual, &actual), 0.0);
        assert_relative_eq!(mse(&actual, &actual), 0.0);
        assert_relative_eq!(rmse(&actual, &actual), 0.0);
    }

    #[test]
    fn anomaly_scores_match_absolute_residuals() {
        let actual = [2.0, 4.0, 6.0];
        let predicted = [1.0, 5.0, 5.5];
        assert_eq!(anomaly_scores(&actual, &predicted), vec![1.0, 1.0, 0.5]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn mae_panics_on_mismatched_lengths() {
        let _ = mae(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn mse_panics_on_mismatched_lengths() {
        let _ = mse(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn residuals_panics_on_mismatched_lengths() {
        let _ = residuals(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn empty_slices_return_zero() {
        assert_relative_eq!(mae(&[], &[]), 0.0);
        assert_relative_eq!(mse(&[], &[]), 0.0);
    }

    #[test]
    fn residuals_correct_values() {
        let r = residuals(&[5.0, 3.0], &[4.0, 4.0]);
        assert_eq!(r, vec![1.0, -1.0]);
    }
}
