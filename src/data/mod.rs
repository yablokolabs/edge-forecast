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
