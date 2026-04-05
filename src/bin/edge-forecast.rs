use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand, ValueEnum};
use edge_forecast::core::{ForecastWindow, Forecaster};
use edge_forecast::data::load_single_column_csv;
use edge_forecast::metrics::{mae, rmse};
use edge_forecast::model_io::{build_forecaster_from_state, load_model, save_model};
use edge_forecast::models::{AutoregressiveForecaster, ReservoirForecaster, SpinForecaster};

#[derive(Debug, Parser)]
#[command(name = "edge-forecast")]
#[command(about = "Compact forecasting CLI for edge time-series")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Train {
        #[arg(long)]
        input: String,
        #[arg(long, default_value_t = 0)]
        column: usize,
        #[arg(long, value_enum)]
        model: ModelKind,
        #[arg(long)]
        output: String,
    },
    Forecast {
        #[arg(long)]
        input: String,
        #[arg(long, default_value_t = 0)]
        column: usize,
        #[arg(long)]
        model_file: String,
        #[arg(long, default_value_t = 5)]
        horizon: usize,
    },
    Eval {
        #[arg(long)]
        input: String,
        #[arg(long, default_value_t = 0)]
        column: usize,
        #[arg(long, value_enum)]
        model: ModelKind,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelKind {
    Ar,
    Reservoir,
    Spin,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train {
            input,
            column,
            model,
            output,
        } => train_cmd(&input, column, model, &output),
        Command::Forecast {
            input,
            column,
            model_file,
            horizon,
        } => forecast_cmd(&input, column, &model_file, horizon),
        Command::Eval {
            input,
            column,
            model,
        } => eval_cmd(&input, column, model),
    }
}

fn build_model(model: ModelKind) -> Box<dyn Forecaster> {
    match model {
        ModelKind::Ar => Box::new(AutoregressiveForecaster::default()),
        ModelKind::Reservoir => Box::new(ReservoirForecaster::default()),
        ModelKind::Spin => Box::new(SpinForecaster::default()),
    }
}

fn train_cmd(input: &str, column: usize, model: ModelKind, output: &str) -> Result<()> {
    let series = load_single_column_csv(input, column)?;
    if series.len() < 10 {
        return Err(anyhow!("need at least 10 observations for training"));
    }
    let mut forecaster = build_model(model);
    forecaster.fit(&series)?;
    save_model(output, &forecaster.model_state())?;
    println!("saved model to {output}");
    Ok(())
}

fn forecast_cmd(input: &str, column: usize, model_file: &str, horizon: usize) -> Result<()> {
    let series = load_single_column_csv(input, column)?;
    if series.len() < 10 {
        return Err(anyhow!("need at least 10 observations for forecasting"));
    }
    let state = load_model(model_file)?;
    let forecaster = build_forecaster_from_state(&state);
    let context = ForecastWindow::new(series[series.len() - 10..].to_vec());
    let result = forecaster.forecast(&context, horizon)?;
    println!(
        "predictions: {}",
        serde_json::to_string_pretty(&result.predictions)?
    );
    Ok(())
}

fn eval_cmd(input: &str, column: usize, model: ModelKind) -> Result<()> {
    let series = load_single_column_csv(input, column)?;
    if series.len() < 20 {
        return Err(anyhow!("need at least 20 observations for evaluation"));
    }
    let split = series.len() * 4 / 5;
    let train = &series[..split];
    let test = &series[split..];
    let mut forecaster = build_model(model);
    forecaster.fit(train)?;
    let context = ForecastWindow::new(train[train.len() - 10..].to_vec());
    let result = forecaster.forecast(&context, test.len())?;
    println!("mae={:.6}", mae(test, &result.predictions));
    println!("rmse={:.6}", rmse(test, &result.predictions));
    Ok(())
}
