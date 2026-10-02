//! Standard V4L2 control access via `VIDIOC_G_CTRL` / `VIDIOC_S_CTRL`.

use std::os::unix::io::AsRawFd;

use nix::errno::Errno;
use nix::ioctl_readwrite;
use thiserror::Error;

/// Mirror of `struct v4l2_queryctrl` from `linux/videodev2.h`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct V4l2QueryCtrl {
    id: u32,
    ctype: u32,
    name: [u8; 32],
    minimum: i32,
    maximum: i32,
    step: i32,
    default_value: i32,
    flags: u32,
    reserved: [u32; 2],
}

// `VIDIOC_QUERYCTRL` is `_IOWR('V', 36, struct v4l2_queryctrl)`.
ioctl_readwrite!(vidio_queryctrl, b'V', 36, V4l2QueryCtrl);

/// Mirror of `struct v4l2_control` from `linux/videodev2.h`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct V4l2Control {
    id: u32,
    value: i32,
}

// `VIDIOC_G_CTRL` / `VIDIOC_S_CTRL` are `_IOWR('V', 27 | 28, struct v4l2_control)`.
ioctl_readwrite!(vidio_g_ctrl, b'V', 27, V4l2Control);
ioctl_readwrite!(vidio_s_ctrl, b'V', 28, V4l2Control);

#[derive(Debug, Error)]
pub enum V4l2Error {
    #[error("v4l2 {operation} control 0x{id:08x} failed: {errno}")]
    Query {
        operation: &'static str,
        id: u32,
        errno: Errno,
    },
}

/// Query the minimum/maximum range of a control (`VIDIOC_QUERYCTRL`).
pub fn range(fd: &impl AsRawFd, id: u32) -> Result<(i32, i32), V4l2Error> {
    let mut query = V4l2QueryCtrl {
        id,
        ..Default::default()
    };
    unsafe { vidio_queryctrl(fd.as_raw_fd(), &mut query) }.map_err(|errno| V4l2Error::Query {
        operation: "query",
        id,
        errno,
    })?;
    Ok((query.minimum, query.maximum))
}

/// Read a control value.
pub fn get(fd: &impl AsRawFd, id: u32) -> Result<i32, V4l2Error> {
    let mut ctrl = V4l2Control { id, value: 0 };
    unsafe { vidio_g_ctrl(fd.as_raw_fd(), &mut ctrl) }.map_err(|errno| V4l2Error::Query {
        operation: "get",
        id,
        errno,
    })?;
    Ok(ctrl.value)
}

/// Write a control value.
pub fn set(fd: &impl AsRawFd, id: u32, value: i32) -> Result<(), V4l2Error> {
    let mut ctrl = V4l2Control { id, value };
    unsafe { vidio_s_ctrl(fd.as_raw_fd(), &mut ctrl) }.map_err(|errno| V4l2Error::Query {
        operation: "set",
        id,
        errno,
    })?;
    Ok(())
}
