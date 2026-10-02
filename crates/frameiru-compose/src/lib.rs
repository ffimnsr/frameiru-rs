//! Background replacement compositor.
//!
//! [`CpuCompositor`] (rayon, always available) and — with the `gpu` feature —
//! [`GpuCompositor`] (wgpu) implement the [`Compositor`] trait. Use
//! [`new_compositor`] to get a working compositor with automatic GPU fallback.

pub mod cpu;
pub mod mode;
pub mod video;

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod shaders;

pub use cpu::CpuCompositor;
pub use mode::Background;

#[cfg(feature = "gpu")]
pub use gpu::GpuCompositor;

use frameiru_core::traits::Compositor;

/// Creates a compositor, preferring the GPU pipeline and falling back to CPU.
///
/// WGPU initialization can fail for many reasons (no GPU, headless session,
/// driver issue); the compositor is never required to be fast, so a warning
/// and a CPU fallback is the correct failure mode.
pub fn new_compositor() -> Box<dyn Compositor> {
    #[cfg(feature = "gpu")]
    match GpuCompositor::try_new() {
        Ok(gpu) => {
            tracing::info!("compositor: using wgpu GPU pipeline");
            return Box::new(gpu);
        }
        Err(e) => tracing::warn!("compositor: wgpu unavailable ({e}); using CPU compositor"),
    }

    #[cfg(not(feature = "gpu"))]
    tracing::warn!("compositor: gpu feature disabled; using CPU compositor");

    Box::new(CpuCompositor::new())
}
