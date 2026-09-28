//! `stop`, `status`, and `set-bg`: talk to a running pipeline.

use std::path::PathBuf;

use anyhow::Context as _;
use frameiru_ipc::{IpcClient, IpcRequest, IpcResponse};

use super::SetBgArgs;

pub fn stop(socket: PathBuf) -> anyhow::Result<()> {
    let client = connect(&socket)?;
    match client.request(&IpcRequest::Stop)? {
        IpcResponse::Ok => {
            println!("stopped");
            Ok(())
        }
        IpcResponse::Error { message } => {
            anyhow::bail!("server reported an error: {message}")
        }
        other => anyhow::bail!("unexpected response: {other:?}"),
    }
}

pub fn status(socket: PathBuf) -> anyhow::Result<()> {
    let client = connect(&socket)?;
    match client.request(&IpcRequest::GetStatus)? {
        IpcResponse::Status { status } => {
            println!("running      : {}", status.running);
            match status.resolution {
                Some(res) => println!("resolution   : {}x{}", res.width, res.height),
                None => println!("resolution   : (no frames yet)"),
            }
            println!("background   : {:?}", status.background);
            println!("capture fps  : {:.1}", status.capture_fps);
            println!("composite fps: {:.1}", status.composite_fps);
            println!("frames       : {}", status.frames_composited);
            println!("masks        : {} computed", status.masks_computed);
            println!("latency      : {} us", status.latency_us);
            Ok(())
        }
        IpcResponse::Error { message } => {
            anyhow::bail!("server reported an error: {message}")
        }
        other => anyhow::bail!("unexpected response: {other:?}"),
    }
}

pub fn set_background(args: SetBgArgs) -> anyhow::Result<()> {
    let mode = super::parse_background(&args.background)?;
    let client = connect(&args.socket())?;
    match client.request(&frameiru_ipc::IpcRequest::SetBackground { mode: mode.clone() })? {
        IpcResponse::Ok => {
            println!("background set to {mode:?}");
            Ok(())
        }
        IpcResponse::Error { message } => {
            anyhow::bail!("server rejected the background: {message}")
        }
        other => anyhow::bail!("unexpected response: {other:?}"),
    }
}

fn connect(socket: &PathBuf) -> anyhow::Result<IpcClient> {
    IpcClient::connect(socket).with_context(|| {
        format!(
            "cannot connect to control socket {} (is a pipeline running?)",
            socket.display()
        )
    })
}
