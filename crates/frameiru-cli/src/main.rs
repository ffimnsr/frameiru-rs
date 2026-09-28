//! Frameiru CLI entrypoint: argument parsing and subcommand dispatch.

mod commands;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "frameiru",
    version,
    about = "Virtual webcam with background removal",
    long_about = "Runs a virtual webcam pipeline (capture -> segmentation -> \
                  compositing -> loopback) and controls it via a Unix socket."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the pipeline in the foreground.
    Run(commands::RunArgs),
    /// Launch the pipeline as a background daemon.
    Start(commands::RunArgs),
    /// Stop a running pipeline through its control socket.
    Stop(commands::SocketArg),
    /// Print the pipeline's runtime status.
    Status(commands::SocketArg),
    /// Change the background of a running pipeline.
    #[command(name = "set-bg")]
    SetBg(commands::SetBgArgs),
    /// List V4L2 capture and loopback devices.
    Devices,
    /// Inspect a camera device's formats and resolutions.
    Inspect(commands::DeviceArg),
    /// Manage ONNX segmentation models.
    Models(commands::ModelsCmd),
    /// Run an inference/composition benchmark without a video sink.
    Benchmark(commands::BenchArgs),
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => commands::run::run(args),
        Command::Start(args) => commands::run::start(args),
        Command::Stop(args) => commands::control::stop(args.socket()),
        Command::Status(args) => commands::control::status(args.socket()),
        Command::SetBg(args) => commands::control::set_background(args),
        Command::Devices => commands::devices::list_devices(),
        Command::Inspect(args) => commands::devices::inspect(args.device),
        Command::Models(cmd) => commands::models::models(cmd),
        Command::Benchmark(args) => commands::bench::benchmark(args),
    }
}
