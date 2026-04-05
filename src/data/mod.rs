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
