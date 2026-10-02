//! UVC Extension Unit (XU) control queries via the `UVCIOC_CTRL_QUERY` ioctl.
//!
//! Reverse-engineered protocol for the Anker PowerConf C200: unit `0x06`, with
//! vendor selectors queried through the V4L2 UVC driver. Queries follow the
//! C reference implementation from `erans/anker-powerconf-c200-linux-tools`:
//! every `SET_CUR` first discovers the real control length with `GET_LEN`
//! (the Linux path must use the reported length, not a fixed staging size).

use std::os::unix::io::AsRawFd;

use nix::errno::Errno;
use nix::ioctl_readwrite;
use thiserror::Error;

/// Camera extension unit id.
pub const UNIT_ID: u8 = 0x06;
/// Staging buffer size; reported `GET_LEN` may be smaller (60 for FOV).
pub const MAX_PAYLOAD_SIZE: usize = 64;

/// UVC class-specific request codes (`linux/usb/video.h`, section A.8).
pub const UVC_GET_CUR: u8 = 0x81;
pub const UVC_SET_CUR: u8 = 0x01;
pub const UVC_GET_LEN: u8 = 0x85;

/// Mirror of `struct uvc_xu_control_query` from `linux/uvcvideo.h`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct UvcXuControlQuery {
    unit: u8,
    selector: u8,
    query: u8,
    size: u16,
    data: *mut u8,
}

// `UVCIOC_CTRL_QUERY` is `_IOWR('u', 0x21, struct uvc_xu_control_query)`.
ioctl_readwrite!(uvcioc_ctrl_query, b'u', 0x21, UvcXuControlQuery);

#[derive(Debug, Error)]
pub enum VendorError {
    #[error("uvc extension unit {operation} (selector 0x{selector:02x}) failed: {errno}")]
    Query {
        operation: &'static str,
        selector: u8,
        errno: Errno,
    },
    #[error("uvc extension unit reported invalid length {size} for selector 0x{selector:02x}")]
    BadLength { selector: u8, size: u16 },
    #[error("value offset {offset} outside payload of {max} bytes")]
    BadOffset { offset: u16, max: u16 },
}

/// Zero-padded control payload plus the control's real length.
#[derive(Debug, Clone, Copy)]
pub struct Payload {
    pub data: [u8; MAX_PAYLOAD_SIZE],
    pub size: u16,
}

impl Default for Payload {
    fn default() -> Self {
        Self {
            data: [0u8; MAX_PAYLOAD_SIZE],
            size: 0,
        }
    }
}

/// Reject offsets that cannot hold a u16 within the payload (C: `offset + 1 >= 64`).
pub fn validate_offset(offset: u16) -> Result<(), VendorError> {
    if offset >= (MAX_PAYLOAD_SIZE - 1) as u16 {
        Err(VendorError::BadOffset {
            offset,
            max: MAX_PAYLOAD_SIZE as u16,
        })
    } else {
        Ok(())
    }
}

fn query(
    fd: &impl AsRawFd,
    selector: u8,
    operation: &'static str,
    code: u8,
    size: u16,
    payload: &mut [u8],
) -> Result<(), VendorError> {
    let mut ctrl = UvcXuControlQuery {
        unit: UNIT_ID,
        selector,
        query: code,
        size,
        data: payload.as_mut_ptr(),
    };
    unsafe { uvcioc_ctrl_query(fd.as_raw_fd(), &mut ctrl) }.map_err(|errno| {
        VendorError::Query {
            operation,
            selector,
            errno,
        }
    })?;
    Ok(())
}

fn get_length(fd: &impl AsRawFd, selector: u8) -> Result<u16, VendorError> {
    let mut bytes = [0u8; 2];
    query(fd, selector, "GET_LEN", UVC_GET_LEN, 2, &mut bytes)?;
    let size = u16::from_le_bytes(bytes);
    if size == 0 || size > MAX_PAYLOAD_SIZE as u16 {
        return Err(VendorError::BadLength { selector, size });
    }
    Ok(size)
}

fn get_payload(fd: &impl AsRawFd, selector: u8) -> Result<Payload, VendorError> {
    let size = get_length(fd, selector)?;
    let mut payload = Payload {
        size,
        ..Payload::default()
    };
    query(
        fd,
        selector,
        "GET_CUR",
        UVC_GET_CUR,
        size,
        &mut payload.data,
    )?;
    Ok(payload)
}

fn set_payload(fd: &impl AsRawFd, selector: u8, payload: &Payload) -> Result<(), VendorError> {
    let mut data = payload.data;
    query(
        fd,
        selector,
        "SET_CUR",
        UVC_SET_CUR,
        payload.size,
        &mut data,
    )
}

/// Build a zeroed payload of the given size with one byte set.
fn payload_with_byte(size: u16, index: usize, value: u8) -> Payload {
    let mut payload = Payload {
        size,
        ..Payload::default()
    };
    payload.data[index] = value;
    payload
}

pub fn get_bool(fd: &impl AsRawFd, selector: u8) -> Result<bool, VendorError> {
    let payload = get_payload(fd, selector)?;
    Ok(payload.data[0] == 0x01)
}

pub fn set_bool(fd: &impl AsRawFd, selector: u8, value: bool) -> Result<(), VendorError> {
    let discovered = get_payload(fd, selector)?;
    let payload = payload_with_byte(discovered.size, 0, if value { 0x01 } else { 0x00 });
    set_payload(fd, selector, &payload)
}

pub fn get_u8(fd: &impl AsRawFd, selector: u8) -> Result<u8, VendorError> {
    let payload = get_payload(fd, selector)?;
    Ok(payload.data[0])
}

pub fn set_u8(fd: &impl AsRawFd, selector: u8, value: u8) -> Result<(), VendorError> {
    let discovered = get_payload(fd, selector)?;
    let payload = payload_with_byte(discovered.size, 0, value);
    set_payload(fd, selector, &payload)
}

pub fn get_u16(fd: &impl AsRawFd, selector: u8, value_offset: u16) -> Result<u16, VendorError> {
    validate_offset(value_offset)?;
    let payload = get_payload(fd, selector)?;
    let offset = value_offset as usize;
    Ok(u16::from_le_bytes([
        payload.data[offset],
        payload.data[offset + 1],
    ]))
}

/// Set a u16 at `value_offset`; mirrors the C wire format, which also stamps
/// `0x00 0x01` at the front of the payload for vendor selectors.
pub fn set_u16(
    fd: &impl AsRawFd,
    selector: u8,
    value: u16,
    value_offset: u16,
) -> Result<(), VendorError> {
    validate_offset(value_offset)?;
    let discovered = get_payload(fd, selector)?;
    let mut payload = Payload {
        size: discovered.size,
        ..Payload::default()
    };
    let offset = value_offset as usize;
    payload.data[0] = 0x00;
    payload.data[1] = 0x01;
    payload.data[offset] = (value & 0xff) as u8;
    payload.data[offset + 1] = (value >> 8) as u8;
    set_payload(fd, selector, &payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_validation_accepts_in_range() {
        assert!(validate_offset(0).is_ok());
        assert!(validate_offset(62).is_ok());
    }

    #[test]
    fn offset_validation_rejects_out_of_range() {
        assert!(matches!(
            validate_offset(63),
            Err(VendorError::BadOffset { .. })
        ));
        assert!(matches!(
            validate_offset(u16::MAX),
            Err(VendorError::BadOffset { .. })
        ));
    }
}
