//! Picture-in-picture mini preview: an always-on-top window shown at the
//! bottom-right of the desktop while the main window is minimized and the
//! engine is running.
//!
//! The winit backend mirrors the OS minimize state into
//! [`slint::Window::is_minimized`], so a UI-thread [`slint::Timer`] polls it
//! at a low cadence and shows/hides the PiP window. Placement uses the X11
//! EWMH `_NET_WORKAREA` of the current desktop; on Wayland and other window
//! managers that ignore `set_position` the window manager picks the spot.

use std::cell::Cell;
use std::rc::Rc;
use std::time::Duration;

use slint::{ComponentHandle, PhysicalPosition, Weak};

use crate::gen::{MainWindow, MiniPreview};

/// Margin between the PiP window and the work-area edges.
const EDGE_MARGIN: u32 = 16;
/// Poll cadence for the minimize state (UI-thread timer).
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Work area of the current desktop, in physical pixels (EWMH
/// `_NET_WORKAREA`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkArea {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Bottom-right placement for a `width` x `height` (physical px) window
/// inside `work`, inset by `margin`. Clamped so the window never leaves the
/// work area horizontally. Pure, so it is unit-testable without a display.
pub fn pip_position(work: WorkArea, width: u32, height: u32, margin: u32) -> (i32, i32) {
    let x = work.x + work.width as i32 - width as i32 - margin as i32;
    let y = work.y + work.height as i32 - height as i32 - margin as i32;
    (x.max(work.x), y.max(work.y))
}

/// Queries the current desktop's work area over X11 EWMH. Returns `None`
/// when X11 is unavailable (e.g. Wayland) or the property is missing; the
/// caller then leaves placement to the window manager.
#[cfg(target_os = "linux")]
fn x11_work_area() -> Option<WorkArea> {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _};

    let (conn, screen_num) = x11rb::connect(None).ok()?;
    let screen = &conn.setup().roots[screen_num];
    let workarea_atom = conn
        .intern_atom(false, b"_NET_WORKAREA")
        .ok()?
        .reply()
        .ok()?
        .atom;
    let desktop_atom = conn
        .intern_atom(false, b"_NET_CURRENT_DESKTOP")
        .ok()?
        .reply()
        .ok()?
        .atom;

    // _NET_WORKAREA is `x, y, w, h` per desktop, each a single CARDINAL.
    let values: Vec<u32> = conn
        .get_property(false, screen.root, workarea_atom, AtomEnum::CARDINAL, 0, 32)
        .ok()?
        .reply()
        .ok()?
        .value32()?
        .collect();
    let desktop = conn
        .get_property(false, screen.root, desktop_atom, AtomEnum::CARDINAL, 0, 32)
        .ok()?
        .reply()
        .ok()?
        .value32()?
        .next()? as usize;
    let area = values.get(desktop * 4..desktop * 4 + 4)?;
    Some(WorkArea {
        x: area[0] as i32,
        y: area[1] as i32,
        width: area[2],
        height: area[3],
    })
}

/// Owns the PiP window and drives its visibility from the main window's
/// minimize state. UI-thread only; the poll timer and all handles stay on
/// the thread that created the controller.
pub struct MiniPreviewController {
    main: Weak<MainWindow>,
    mini: MiniPreview,
    /// Set when the user closes the PiP window; it stays hidden for the
    /// current minimize session, then re-arms on restore.
    user_closed: Rc<Cell<bool>>,
    /// Poll timer; started/stopped by `set_enabled`.
    poll: slint::Timer,
}

impl MiniPreviewController {
    pub fn new(main: Weak<MainWindow>) -> anyhow::Result<Self> {
        let mini = MiniPreview::new()?;
        let user_closed = Rc::new(Cell::new(false));
        let flag = Rc::clone(&user_closed);
        mini.on_close_clicked(move || flag.set(true));
        Ok(Self {
            main,
            mini,
            user_closed,
            poll: slint::Timer::default(),
        })
    }

    /// A cheap handle for the preview pump (sets the image on the UI thread).
    pub fn weak(&self) -> Weak<MiniPreview> {
        self.mini.as_weak()
    }

    /// Enables or disables the minimized -> PiP behavior. Disabling hides
    /// the PiP window immediately.
    pub fn set_enabled(&self, enabled: bool) {
        if enabled {
            if !self.poll.running() {
                let main = self.main.clone();
                let mini = self.mini.as_weak();
                let user_closed = Rc::clone(&self.user_closed);
                self.poll
                    .start(slint::TimerMode::Repeated, POLL_INTERVAL, move || {
                        tick(&main, &mini, &user_closed);
                    });
            }
        } else {
            self.poll.stop();
            self.user_closed.set(false);
            let _ = self.mini.hide();
        }
    }
}

/// One poll tick: show the PiP over the bottom-right corner while the main
/// window is minimized and the engine runs; hide it on restore/stop.
fn tick(main: &Weak<MainWindow>, mini: &Weak<MiniPreview>, user_closed: &Rc<Cell<bool>>) {
    let Some(main) = main.upgrade() else {
        return;
    };
    let Some(mini) = mini.upgrade() else {
        return;
    };
    let show_pip = main.get_is_running() && main.window().is_minimized();
    if show_pip {
        if !user_closed.get() && !mini.window().is_visible() {
            position_bottom_right(&mini);
            let _ = mini.show();
        }
    } else {
        // Restored or engine stopped: PiP goes away and the next minimize
        // session re-arms it.
        user_closed.set(false);
        if mini.window().is_visible() {
            let _ = mini.hide();
        }
    }
}

/// Moves the PiP window to the bottom-right of the current desktop work
/// area. No-op when the geometry cannot be queried (non-X11).
fn position_bottom_right(mini: &MiniPreview) {
    #[cfg(target_os = "linux")]
    {
        let Some(work) = x11_work_area() else {
            return;
        };
        let size = mini.window().size();
        let (x, y) = pip_position(work, size.width, size.height, EDGE_MARGIN);
        mini.window().set_position(PhysicalPosition::new(x, y));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_in_bottom_right_of_work_area() {
        let work = WorkArea {
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
        };
        assert_eq!(pip_position(work, 300, 225, 16), (1604, 839));
    }

    #[test]
    fn respects_nonzero_work_area_origin() {
        let work = WorkArea {
            x: 1920,
            y: 0,
            width: 1920,
            height: 1080,
        };
        assert_eq!(pip_position(work, 300, 225, 16), (3524, 839));
    }

    #[test]
    fn clamps_when_work_area_is_smaller_than_window() {
        let work = WorkArea {
            x: 10,
            y: 10,
            width: 200,
            height: 100,
        };
        assert_eq!(pip_position(work, 300, 225, 16), (10, 10));
    }

    #[test]
    fn zero_margin_touches_edge() {
        let work = WorkArea {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
        };
        assert_eq!(pip_position(work, 320, 240, 0), (480, 360));
    }
}
