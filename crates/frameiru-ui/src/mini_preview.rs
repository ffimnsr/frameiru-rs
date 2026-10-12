//! Picture-in-picture mini preview: an always-on-top window shown at the
//! bottom-right of the desktop while the main window is minimized and the
//! engine is running.
//!
//! Minimize detection:
//! - [`slint::Window::is_minimized`] mirrors the winit backend's state sync
//!   (resize/occlusion driven; unreliable on X11 and absent on Wayland).
//! - X11 additionally polls the EWMH `_NET_WM_STATE` of our own native
//!   window for `_NET_WM_STATE_HIDDEN`, because winit has no unmap handler
//!   for X11 and some window managers never trigger the state sync.
//! - KDE Plasma (KDE Plasma Wayland) reports minimize via a KWin script
//!   over the session D-Bus ([`crate::kwin_watch`]); other Wayland
//!   compositors expose no minimize signal at all.
//!
//! Placement uses the X11 EWMH `_NET_WORKAREA` of the current desktop; on
//! non-X11 the window manager picks the spot.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use slint::{ComponentHandle, PhysicalPosition, Weak};

use crate::gen::{MainWindow, MiniPreview};
use crate::kwin_watch::KWinWatch;

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

/// Interns `name` once and caches the atom in `cache`.
fn cached_atom(
    conn: Option<&x11rb::rust_connection::RustConnection>,
    name: &[u8],
    cache: &mut Option<u32>,
) -> Option<u32> {
    if let Some(atom) = *cache {
        return Some(atom);
    }
    use x11rb::protocol::xproto::ConnectionExt as _;
    let atom = conn?.intern_atom(false, name).ok()?.reply().ok()?.atom;
    *cache = Some(atom);
    Some(atom)
}

/// Cached X11 EWMH state watch. Connects lazily on first use; every poll is
/// a couple of cheap local-socket roundtrips. UI thread only.
#[derive(Default)]
struct X11Watch {
    conn: Option<(x11rb::rust_connection::RustConnection, usize)>,
    workarea_atom: Option<u32>,
    desktop_atom: Option<u32>,
    state_atom: Option<u32>,
    hidden_atom: Option<u32>,
}

impl X11Watch {
    fn ensure_conn(&mut self) {
        if self.conn.is_none() {
            self.conn = x11rb::connect(None).ok();
        }
    }

    /// Work area of the current desktop; `None` outside X11.
    fn work_area(&mut self) -> Option<WorkArea> {
        self.ensure_conn();
        use x11rb::connection::Connection;
        use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _};
        let (conn, screen_num) = self.conn.as_ref()?;
        let root = conn.setup().roots[*screen_num].root;
        let wa = cached_atom(Some(conn), b"_NET_WORKAREA", &mut self.workarea_atom)?;
        let da = cached_atom(Some(conn), b"_NET_CURRENT_DESKTOP", &mut self.desktop_atom)?;
        // _NET_WORKAREA is `x, y, w, h` per desktop, each a single CARDINAL.
        let values: Vec<u32> = conn
            .get_property(false, root, wa, AtomEnum::CARDINAL, 0, 32)
            .ok()?
            .reply()
            .ok()?
            .value32()?
            .collect();
        let desktop = conn
            .get_property(false, root, da, AtomEnum::CARDINAL, 0, 32)
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

    /// True when the window manager reports the window minimized
    /// (`_NET_WM_STATE_HIDDEN`, EWMH). False outside X11.
    fn ewmh_minimized(&mut self, window: &slint::Window) -> bool {
        self.ensure_conn();
        use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _};
        let Some(id) = native_x11_window_id(window) else {
            return false;
        };
        let Some(state) = cached_atom(
            self.conn.as_ref().map(|(conn, _)| conn),
            b"_NET_WM_STATE",
            &mut self.state_atom,
        ) else {
            return false;
        };
        let Some(hidden) = cached_atom(
            self.conn.as_ref().map(|(conn, _)| conn),
            b"_NET_WM_STATE_HIDDEN",
            &mut self.hidden_atom,
        ) else {
            return false;
        };
        let Some(reply) = self
            .conn
            .as_ref()
            .and_then(|(conn, _)| {
                conn.get_property(false, id, state, AtomEnum::ATOM, 0, 32)
                    .ok()
            })
            .and_then(|cookie| cookie.reply().ok())
        else {
            return false;
        };
        reply
            .value32()
            .is_some_and(|mut atoms| atoms.any(|atom| atom == hidden))
    }
}

/// The native X11 window id behind a Slint window, when running under X11.
fn native_x11_window_id(window: &slint::Window) -> Option<u32> {
    use raw_window_handle::{HasWindowHandle as _, RawWindowHandle};
    let handle = window.window_handle();
    let raw = handle.window_handle().ok()?;
    match raw.as_raw() {
        RawWindowHandle::Xlib(h) => Some(h.window as usize as u32),
        RawWindowHandle::Xcb(h) => Some(h.window.get()),
        _ => None,
    }
}

/// Shared minimized/visibility state. Held by the controller and captured
/// (cloned) by the poll timer; it never references the timer back, so the
/// timer closure cannot keep a cycle alive.
struct PipShared {
    main: Weak<MainWindow>,
    mini: Weak<MiniPreview>,
    /// Close button pressed: stays hidden for this minimize session.
    user_closed: Cell<bool>,
    x11: RefCell<X11Watch>,
    /// KWin (KDE Plasma Wayland) reported minimize state.
    kwin_minimized: Arc<AtomicBool>,
}

impl PipShared {
    /// Show the PiP while the engine runs and the main window is minimized,
    /// hide it on restore/stop; first show re-arms the user close flag.
    fn tick(&self) {
        let Some(main) = self.main.upgrade() else {
            return;
        };
        let Some(mini) = self.mini.upgrade() else {
            return;
        };
        let minimized = main.window().is_minimized()
            || self.x11.borrow_mut().ewmh_minimized(main.window())
            || self.kwin_minimized.load(Ordering::Relaxed);
        let show_pip = main.get_is_running() && minimized;
        if show_pip {
            if !self.user_closed.get() && !mini.window().is_visible() {
                // Show first so the native window exists, then place it;
                // both happen within this tick, so no intermediate frame is
                // painted at the window manager's default spot.
                let _ = mini.show();
                self.position_bottom_right(&mini);
            }
        } else {
            self.user_closed.set(false);
            if mini.window().is_visible() {
                let _ = mini.hide();
            }
        }
    }

    /// Moves the PiP window to the bottom-right of the current desktop work
    /// area. No-op when the geometry cannot be queried (non-X11).
    fn position_bottom_right(&self, mini: &MiniPreview) {
        let Some(work) = self.x11.borrow_mut().work_area() else {
            return;
        };
        let size = mini.window().size();
        let (x, y) = pip_position(work, size.width, size.height, EDGE_MARGIN);
        mini.window().set_position(PhysicalPosition::new(x, y));
    }
}

/// Owns the PiP window and drives its visibility from the main window's
/// minimize state. UI-thread only; the poll timer and all handles stay on
/// the thread that created the controller.
pub struct MiniPreviewController {
    mini: MiniPreview,
    shared: Rc<PipShared>,
    /// Poll timer; started/stopped by `set_enabled`.
    poll: slint::Timer,
    /// KDE Plasma (Wayland) minimize watch via KWin scripting.
    kwin: KWinWatch,
}

impl MiniPreviewController {
    pub fn new(main: Weak<MainWindow>) -> anyhow::Result<Self> {
        let mini = MiniPreview::new()?;
        let kwin_minimized = Arc::new(AtomicBool::new(false));
        let shared = Rc::new(PipShared {
            main,
            mini: mini.as_weak(),
            user_closed: Cell::new(false),
            x11: RefCell::new(X11Watch::default()),
            kwin_minimized: Arc::clone(&kwin_minimized),
        });
        let flag = Rc::clone(&shared);
        mini.on_close_clicked(move || flag.user_closed.set(true));
        Ok(Self {
            mini,
            shared,
            poll: slint::Timer::default(),
            kwin: KWinWatch::new(std::process::id(), kwin_minimized),
        })
    }

    /// A cheap handle for the preview pump (sets the image on the UI thread).
    pub fn weak(&self) -> Weak<MiniPreview> {
        self.mini.as_weak()
    }

    /// Enables or disables the minimized -> PiP behavior. Disabling hides
    /// the PiP window immediately.
    pub fn set_enabled(&self, enabled: bool) {
        self.kwin.set_enabled(enabled);
        if enabled {
            if !self.poll.running() {
                let shared = Rc::clone(&self.shared);
                self.poll
                    .start(slint::TimerMode::Repeated, POLL_INTERVAL, move || {
                        shared.tick();
                    });
            }
        } else {
            self.poll.stop();
            self.shared.user_closed.set(false);
            let _ = self.mini.hide();
        }
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
