//! Vendor FOV preset control (extension unit `0x06`, selector `0x10`).
//!
//! Wire format (from the C reference implementation):
//! - SET_CUR payload: `[0x00, 0x01, value_le16, 0x00, ...zeros]`
//! - value lives at payload offset 2

use std::os::unix::io::AsRawFd;

use crate::vendor;
use crate::Error;

pub const UNIT_ID: u8 = 0x06;
pub const SELECTOR: u8 = 0x10;
pub const PAYLOAD_SIZE: usize = 64;

/// Known FOV presets: raw value, preset name, display string.
pub const FOV_PRESETS: [(u16, &str, &str); 3] = [
    (65, "narrow", "65 (narrow)"),
    (78, "medium", "78 (medium)"),
    (95, "wide", "95 (wide)"),
];

pub fn get(fd: &impl AsRawFd) -> Result<u16, Error> {
    Ok(vendor::get_u16(fd, SELECTOR, 2)?)
}

pub fn set(fd: &impl AsRawFd, value: u16) -> Result<(), Error> {
    Ok(vendor::set_u16(fd, SELECTOR, value, 2)?)
}

/// Encode the SET_CUR payload for a FOV value (64 bytes, zero-filled).
pub fn encode_set_payload(value: u16) -> [u8; PAYLOAD_SIZE] {
    let mut payload = [0u8; PAYLOAD_SIZE];
    payload[0] = 0x00;
    payload[1] = 0x01;
    payload[2] = (value & 0xff) as u8;
    payload[3] = (value >> 8) as u8;
    payload
}

/// Decode the FOV value from a GET_CUR payload (offset 2, little-endian).
pub fn decode_payload(payload: &[u8]) -> u16 {
    u16::from_le_bytes([payload[2], payload[3]])
}

/// Accepts preset names (`narrow`/`medium`/`wide`) or a raw u16 (mirrors C).
pub fn parse_value(text: &str) -> Option<u16> {
    match text.to_ascii_lowercase().as_str() {
        "narrow" => Some(65),
        "medium" => Some(78),
        "wide" => Some(95),
        _ => text.parse::<u16>().ok(),
    }
}

/// Render a FOV value with its preset name when known.
pub fn describe_value(value: u16) -> String {
    match FOV_PRESETS.iter().find(|(raw, _, _)| *raw == value) {
        Some((_, _, label)) => (*label).to_string(),
        None => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_set_payload_matches_c_reference() {
        let payload = encode_set_payload(95);
        assert_eq!(payload[0], 0x00);
        assert_eq!(payload[1], 0x01);
        assert_eq!(payload[2], 0x5f);
        assert_eq!(payload[3], 0x00);
        assert_eq!(payload[4], 0x00);
        assert_eq!(payload[63], 0x00);
        assert_eq!(payload[5..63].iter().sum::<u8>(), 0);
    }

    #[test]
    fn encode_set_payload_all_presets() {
        for (raw, _, _) in FOV_PRESETS {
            let payload = encode_set_payload(raw);
            assert_eq!(decode_payload(&payload), raw);
        }
    }

    #[test]
    fn decode_payload_reads_le_u16_at_offset_2() {
        let mut payload = [0u8; PAYLOAD_SIZE];
        payload[2] = 78;
        payload[3] = 0;
        assert_eq!(decode_payload(&payload), 78);
    }

    #[test]
    fn parse_value_accepts_presets_and_raw() {
        assert_eq!(parse_value("65"), Some(65));
        assert_eq!(parse_value("narrow"), Some(65));
        assert_eq!(parse_value("NARROW"), Some(65));
        assert_eq!(parse_value("medium"), Some(78));
        assert_eq!(parse_value("wide"), Some(95));
        assert_eq!(parse_value("100"), Some(100));
        assert_eq!(parse_value("0"), Some(0));
        assert_eq!(parse_value("65535"), Some(u16::MAX));
    }

    #[test]
    fn parse_value_rejects_invalid_input() {
        assert_eq!(parse_value("banana"), None);
        assert_eq!(parse_value("70000"), None);
        assert_eq!(parse_value("-1"), None);
        assert_eq!(parse_value("10.5"), None);
        assert_eq!(parse_value(""), None);
    }

    #[test]
    fn describe_value_annotates_presets() {
        assert_eq!(describe_value(65), "65 (narrow)");
        assert_eq!(describe_value(78), "78 (medium)");
        assert_eq!(describe_value(95), "95 (wide)");
        assert_eq!(describe_value(123), "123");
    }
}
