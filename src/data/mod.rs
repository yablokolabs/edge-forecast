use anyhow::{Result, anyhow};
use csv::StringRecord;

pub fn load_single_column_csv(path: &str, column_index: usize) -> Result<Vec<f64>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut values = Vec::new();
    for record in rdr.records() {
        let record: StringRecord = record?;
        let raw = record
            .get(column_index)
            .ok_or_else(|| anyhow!("missing column index {column_index}"))?;
        values.push(raw.parse::<f64>()?);
    }
    Ok(values)
}

pub fn load_multi_column_csv(path: &str, column_indexes: &[usize]) -> Result<Vec<Vec<f64>>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut columns = vec![Vec::new(); column_indexes.len()];
    for record in rdr.records() {
        let record: StringRecord = record?;
        for (out_idx, column_idx) in column_indexes.iter().copied().enumerate() {
            let raw = record
                .get(column_idx)
                .ok_or_else(|| anyhow!("missing column index {column_idx}"))?;
            columns[out_idx].push(raw.parse::<f64>()?);
        }
    }
    Ok(columns)
}

pub fn mean_series(columns: &[Vec<f64>]) -> Result<Vec<f64>> {
    if columns.is_empty() {
        return Err(anyhow!("at least one column is required"));
    }
    let len = columns[0].len();
    if columns.iter().any(|c| c.len() != len) {
        return Err(anyhow!("all columns must have the same length"));
    }
    let mut out = vec![0.0; len];
    for column in columns {
        for (idx, value) in column.iter().copied().enumerate() {
            out[idx] += value;
        }
    }
    let denom = columns.len() as f64;
    for value in &mut out {
        *value /= denom;
    }
    Ok(out)
}

pub fn load_series(path: &str, column_indexes: &[usize]) -> Result<Vec<f64>> {
    match column_indexes {
        [] => Err(anyhow!("at least one column index must be provided")),
        [single] => load_single_column_csv(path, *single),
        many => {
            let columns = load_multi_column_csv(path, many)?;
            mean_series(&columns)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_csv(name: &str, content: &str) -> String {
        let path = format!(
            "/tmp/edge-forecast-test-{}-{}.csv",
            name,
            std::process::id()
        );
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn load_single_column() {
        let path = write_csv("single", "value\n1.0\n2.0\n3.0\n");
        let vals = load_single_column_csv(&path, 0).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_multi_column() {
        let path = write_csv("multi", "a,b\n1.0,4.0\n2.0,5.0\n3.0,6.0\n");
        let cols = load_multi_column_csv(&path, &[0, 1]).unwrap();
        assert_eq!(cols[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(cols[1], vec![4.0, 5.0, 6.0]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mean_series_averages_columns() {
        let cols = vec![vec![2.0, 4.0], vec![6.0, 8.0]];
        let m = mean_series(&cols).unwrap();
        assert_eq!(m, vec![4.0, 6.0]);
    }

    #[test]
    fn mean_series_rejects_empty() {
        assert!(mean_series(&[]).is_err());
    }

    #[test]
    fn load_series_rejects_empty_columns() {
        assert!(load_series("/dev/null", &[]).is_err());
    }

    #[test]
    fn load_single_column_missing_index() {
        let path = write_csv("missing", "value\n1.0\n");
        let result = load_single_column_csv(&path, 5);
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }
}
