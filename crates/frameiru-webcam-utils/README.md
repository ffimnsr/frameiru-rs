# frameiru-webcam-utils

Rust CLI for the Anker PowerConf C200 webcam: read and set vendor-only camera
controls (FOV preset, HDR, horizontal flip, vertical screen, anti-flicker)
plus standard V4L2/UVC controls.

Port of the C reference tool
[`erans/anker-powerconf-c200-linux-tools`](https://github.com/erans/anker-powerconf-c200-linux-tools)
(MIT), which reverse engineered the vendor extension-unit protocol used by the
official Anker app.

## Usage

```sh
frameiru-webcam-utils list-controls
frameiru-webcam-utils get fov
frameiru-webcam-utils set fov wide
frameiru-webcam-utils set hdr on
frameiru-webcam-utils set horizontal_flip off
frameiru-webcam-utils set brightness 60
```

Compatibility aliases:

```sh
frameiru-webcam-utils get-fov
frameiru-webcam-utils set-fov narrow
```

Use a different device node:

```sh
frameiru-webcam-utils --device /dev/video2 get fov
```

## Protocol notes (reverse engineered)

Confirmed on a C200 with USB VID:PID `291a:3369`.

- Vendor controls use UVC Extension Unit `0x06`.
- Linux control length is discovered with `GET_LEN` and reused for `SET_CUR`
  (FOV reports 60 bytes on Linux; the macOS staging buffer is 64).
- FOV selector `0x10`, HDR `0x13`, horizontal flip `0x11`,
  vertical screen `0x0a`, anti-flicker `0x12`.
- FOV payload: `[0x00, 0x01, value_le16, 0x00, ...]`; presets `65` (narrow),
  `78` (medium), `95` (wide).
- `zoom_absolute` is the standard UVC zoom control, separate from the vendor
  FOV preset.

Works only on Linux (V4L2 ioctls). Needs read/write access to the video node;
you may need to be in the `video` group.