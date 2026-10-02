//! Wire protocol: newline-delimited JSON messages over the control socket.

use frameiru_core::error::FrameiruError;
use frameiru_core::format::Resolution;
use frameiru_core::BackgroundMode;
use serde::{Deserialize, Serialize};

/// Client -> server control messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IpcRequest {
    /// Change the running pipeline's background.
    SetBackground { mode: BackgroundMode },
    /// Ask for a runtime snapshot.
    GetStatus,
    /// Stop the pipeline and close the server.
    Stop,
}

/// Server -> client replies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IpcResponse {
    /// Command acknowledged (SetBackground, Stop).
    Ok,
    /// Reply to GetStatus.
    Status { status: StatusInfo },
    /// Request could not be fulfilled.
    Error { message: String },
}

/// Runtime snapshot returned by GetStatus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusInfo {
    pub running: bool,
    /// Resolution of the last composited frame.
    pub resolution: Option<Resolution>,
    pub capture_fps: f64,
    pub composite_fps: f64,
    pub frames_composited: u64,
    /// Idle frames written by reusing the previous composite (composite skipped).
    pub composites_skipped: u64,
    /// Successful async mask computations (inference rate).
    pub masks_computed: u64,
    /// Capture-to-composite latency of the last frame, microseconds.
    pub latency_us: u64,
    pub background: BackgroundMode,
}

/// Serializes a message as one JSON line (trailing `\n`).
pub fn encode(message: &impl Serialize) -> Result<Vec<u8>, FrameiruError> {
    let mut bytes = serde_json::to_vec(message)
        .map_err(|e| FrameiruError::Internal(format!("ipc serialization failed: {e}")))?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Parses one JSON line (without the trailing newline).
pub fn decode<T: serde::de::DeserializeOwned>(line: &[u8]) -> Result<T, FrameiruError> {
    serde_json::from_slice(line)
        .map_err(|e| FrameiruError::InvalidArgument(format!("malformed ipc message: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_roundtrip_all_variants() {
        let cases = [
            IpcRequest::SetBackground {
                mode: BackgroundMode::Blur { radius: 12.0 },
            },
            IpcRequest::SetBackground {
                mode: BackgroundMode::Image {
                    path: "/tmp/bg.png".into(),
                },
            },
            IpcRequest::SetBackground {
                mode: BackgroundMode::Video {
                    path: "/tmp/bg.mp4".into(),
                },
            },
            IpcRequest::GetStatus,
            IpcRequest::Stop,
        ];
        for case in cases {
            let bytes = encode(&case).unwrap();
            assert!(bytes.ends_with(b"\n"));
            let decoded: IpcRequest = decode(&bytes[..bytes.len() - 1]).unwrap();
            assert_eq!(decoded, case);
        }
    }

    #[test]
    fn response_roundtrip() {
        let status = StatusInfo {
            running: true,
            resolution: Some(Resolution {
                width: 640,
                height: 480,
            }),
            capture_fps: 29.5,
            composite_fps: 30.0,
            frames_composited: 1234,
            composites_skipped: 42,
            masks_computed: 60,
            latency_us: 42,
            background: BackgroundMode::Color { r: 1, g: 2, b: 3 },
        };
        for case in [
            IpcResponse::Ok,
            IpcResponse::Status {
                status: status.clone(),
            },
            IpcResponse::Error {
                message: "boom".into(),
            },
        ] {
            let bytes = encode(&case).unwrap();
            let decoded: IpcResponse = decode(&bytes[..bytes.len() - 1]).unwrap();
            assert_eq!(decoded, case);
        }
    }

    #[test]
    fn rejects_malformed_and_unknown_messages() {
        assert!(decode::<IpcRequest>(b"not json").is_err());
        // Unknown discriminant.
        assert!(
            decode::<IpcRequest>(b"{\"type\":\"explode\"}").is_err(),
            "unknown request type must fail"
        );
        // Known type but wrong payload.
        assert!(decode::<IpcRequest>(b"{\"type\":\"set_background\"}").is_err());
    }

    #[test]
    fn status_info_uses_snake_case_tag() {
        let bytes = encode(&IpcRequest::GetStatus).unwrap();
        assert_eq!(&bytes[..], b"{\"type\":\"get_status\"}\n");
    }
}
