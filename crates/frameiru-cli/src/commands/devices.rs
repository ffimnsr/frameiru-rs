//! `devices` and `inspect`: V4L2 discovery and format probing.

use std::path::PathBuf;

#[cfg(feature = "v4l2")]
use anyhow::Context as _;
#[cfg(feature = "v4l2")]
use frameiru_capture::v4l;

pub fn list_devices() -> anyhow::Result<()> {
    #[cfg(feature = "v4l2")]
    {
        for node in v4l::context::enum_devices() {
            let name = node.name().unwrap_or_else(|| "(unreadable)".into());
            println!("{} ({})", node.path().display(), name);
        }
        Ok(())
    }
    #[cfg(not(feature = "v4l2"))]
    {
        eprintln!(
            "device discovery needs the `v4l2` feature; rebuild with `cargo build --features v4l2`"
        );
        Ok(())
    }
}

pub fn inspect(device: PathBuf) -> anyhow::Result<()> {
    #[cfg(feature = "v4l2")]
    {
        use v4l::video::traits::Capture;

        let dev = v4l::Device::with_path(&device)
            .with_context(|| format!("cannot open {}", device.display()))?;
        let caps = dev.query_caps()?;
        println!("{}: {} ({})", device.display(), caps.card, caps.driver);
        println!("capabilities: {}", caps.capabilities);

        for format in dev.enum_formats()? {
            println!(
                "\n{} ({})",
                format.fourcc.str().unwrap_or("????"),
                format.description
            );
            for size in dev.enum_framesizes(format.fourcc)? {
                for discrete in size.size.to_discrete() {
                    println!("  {}x{}", discrete.width, discrete.height);
                }
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "v4l2"))]
    {
        let _ = device;
        eprintln!(
            "device inspection needs the `v4l2` feature; rebuild with `cargo build --features v4l2`"
        );
        Ok(())
    }
}
