use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand, ValueEnum};
use edge_forecast::core::{ForecastWindow, Forecaster};
use edge_forecast::data::load_series;
use edge_forecast::metrics::{mae, rmse};
use edge_forecast::model_io::{SavedModel, build_forecaster_from_state, load_model, save_model};
use edge_forecast::models::{AutoregressiveForecaster, ReservoirForecaster, SpinForecaster};
use tiny_http::{Method, Response, Server};

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
        #[arg(long, value_delimiter = ',', default_value = "0")]
        columns: Vec<usize>,
        #[arg(long, value_enum)]
        model: ModelKind,
        #[arg(long)]
        output: String,
        #[arg(long, default_value_t = 10)]
        window_size: usize,
    },
    Forecast {
        #[arg(long)]
        input: String,
        #[arg(long)]
        model_file: String,
        #[arg(long, default_value_t = 5)]
        horizon: usize,
    },
    Eval {
        #[arg(long)]
        input: String,
        #[arg(long, value_delimiter = ',', default_value = "0")]
        columns: Vec<usize>,
        #[arg(long, value_enum)]
        model: ModelKind,
        #[arg(long, default_value_t = 10)]
        window_size: usize,
    },
    Score {
        #[arg(long)]
        input: String,
        #[arg(long, value_delimiter = ',', default_value = "0")]
        columns: Vec<usize>,
        #[arg(long, value_enum)]
        model: ModelKind,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long, default_value_t = 10)]
        window_size: usize,
    },
    Serve {
        #[arg(long)]
        model_file: String,
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,
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
            columns,
            model,
            output,
            window_size,
        } => train_cmd(&input, &columns, model, &output, window_size),
        Command::Forecast {
            input,
            model_file,
            horizon,
        } => forecast_cmd(&input, &model_file, horizon),
        Command::Eval {
            input,
            columns,
            model,
            window_size,
        } => eval_cmd(&input, &columns, model, window_size),
        Command::Score {
            input,
            columns,
            model,
            top_k,
            window_size,
        } => score_cmd(&input, &columns, model, top_k, window_size),
        Command::Serve { model_file, bind } => serve_cmd(&model_file, &bind),
    }
}

fn build_model(model: ModelKind) -> Box<dyn Forecaster> {
    match model {
        ModelKind::Ar => Box::new(AutoregressiveForecaster::default()),
        ModelKind::Reservoir => Box::new(ReservoirForecaster::default()),
        ModelKind::Spin => Box::new(SpinForecaster::default()),
    }
}

fn model_name(model: ModelKind) -> &'static str {
    match model {
        ModelKind::Ar => "ar",
        ModelKind::Reservoir => "reservoir",
        ModelKind::Spin => "spin",
    }
}

fn train_cmd(
    input: &str,
    columns: &[usize],
    model: ModelKind,
    output: &str,
    window_size: usize,
) -> Result<()> {
    let series = load_series(input, columns)?;
    if series.len() < window_size {
        return Err(anyhow!(
            "need at least {window_size} observations for training"
        ));
    }
    let mut forecaster = build_model(model);
    forecaster.fit(&series)?;
    let saved = SavedModel {
        model_name: model_name(model).to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        window_size,
        columns: columns.to_vec(),
        state: forecaster.model_state(),
    };
    save_model(output, &saved)?;
    println!("saved model to {output}");
    Ok(())
}

fn forecast_cmd(input: &str, model_file: &str, horizon: usize) -> Result<()> {
    let saved = load_model(model_file)?;
    let series = load_series(input, &saved.columns)?;
    if series.len() < saved.window_size {
        return Err(anyhow!(
            "need at least {} observations for forecasting",
            saved.window_size
        ));
    }
    let forecaster = build_forecaster_from_state(&saved.state);
    let context = ForecastWindow::new(series[series.len() - saved.window_size..].to_vec());
    let result = forecaster.forecast(&context, horizon)?;
    println!(
        "predictions: {}",
        serde_json::to_string_pretty(&result.predictions)?
    );
    Ok(())
}

fn eval_cmd(input: &str, columns: &[usize], model: ModelKind, window_size: usize) -> Result<()> {
    let series = load_series(input, columns)?;
    if series.len() < window_size * 2 {
        return Err(anyhow!(
            "need at least {} observations for evaluation",
            window_size * 2
        ));
    }
    let split = series.len() * 4 / 5;
    let train = &series[..split];
    let test = &series[split..];
    let mut forecaster = build_model(model);
    forecaster.fit(train)?;
    let context = ForecastWindow::new(train[train.len() - window_size..].to_vec());
    let result = forecaster.forecast(&context, test.len())?;
    println!("mae={:.6}", mae(test, &result.predictions));
    println!("rmse={:.6}", rmse(test, &result.predictions));
    Ok(())
}

fn score_cmd(
    input: &str,
    columns: &[usize],
    model: ModelKind,
    top_k: usize,
    window_size: usize,
) -> Result<()> {
    let series = load_series(input, columns)?;
    if series.len() < window_size * 2 {
        return Err(anyhow!(
            "need at least {} observations for scoring",
            window_size * 2
        ));
    }
    let split = series.len() * 4 / 5;
    let train = &series[..split];
    let test = &series[split..];
    let mut forecaster = build_model(model);
    forecaster.fit(train)?;
    let context = ForecastWindow::new(train[train.len() - window_size..].to_vec());
    let result = forecaster.forecast(&context, test.len())?;
    let mut scored: Vec<(usize, f64, f64, f64)> = test
        .iter()
        .copied()
        .zip(result.predictions.iter().copied())
        .enumerate()
        .map(|(idx, (actual, predicted))| (idx, actual, predicted, (actual - predicted).abs()))
        .collect();
    scored.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(core::cmp::Ordering::Equal));
    println!("top anomaly candidates:");
    for (idx, actual, predicted, score) in scored.into_iter().take(top_k) {
        println!(
            "index={} actual={:.6} predicted={:.6} score={:.6}",
            idx, actual, predicted, score
        );
    }
    Ok(())
}

fn serve_cmd(model_file: &str, bind: &str) -> Result<()> {
    let saved = load_model(model_file)?;
    let forecaster = build_forecaster_from_state(&saved.state);
    let server = Server::http(bind).map_err(|e| anyhow!(e.to_string()))?;
    println!("serving on http://{bind}");
    println!("POST /forecast with JSON: {{\"context\": [..], \"horizon\": 3}}");
    for mut request in server.incoming_requests() {
        if request.method() != &Method::Post || request.url() != "/forecast" {
            let response = Response::from_string("not found").with_status_code(404);
            let _ = request.respond(response);
            continue;
        }
        let mut body = String::new();
        request.as_reader().read_to_string(&mut body)?;
        let payload: ForecastRequest = serde_json::from_str(&body)?;
        let context = ForecastWindow::new(payload.context);
        let result = forecaster.forecast(&context, payload.horizon)?;
        let response = Response::from_string(serde_json::to_string(&result.predictions)?);
        let _ = request.respond(response);
    }
    Ok(())
}

#[derive(Debug, serde::Deserialize)]
struct ForecastRequest {
    context: Vec<f64>,
    horizon: usize,
}
