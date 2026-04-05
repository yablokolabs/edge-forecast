pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / actual.len().max(1) as f64
}

pub fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / actual.len().max(1) as f64
}

pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    mse(actual, predicted).sqrt()
}

pub fn residuals(actual: &[f64], predicted: &[f64]) -> Vec<f64> {
    actual.iter().zip(predicted).map(|(a, p)| a - p).collect()
}

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
}
