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
}
