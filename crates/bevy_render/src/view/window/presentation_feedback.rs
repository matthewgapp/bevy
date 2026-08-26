//! One-shot terminal presentation feedback for one exact window frame.
//!
//! The main world moves a request component onto one window and retains its unique receiver. The
//! render world privately duplicates only the shared request identity during extraction, then owns
//! the exact wgpu feedback future until it reaches a terminal result. The public request is not
//! cloneable, so callers cannot fan one request out across multiple windows or presentations.

use alloc::sync::Arc;
use bevy_ecs::component::Component;
use bevy_platform::sync::Mutex;
use core::{
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum RequestState {
    Requested,
    Presenting,
    Complete(wgpu::PresentationFeedbackResult),
    Taken,
}

/// A per-window request for terminal feedback about one exact presented frame.
///
/// Construct this with [`WindowPresentationFeedbackRequest::new`] and attach it to the target
/// window entity before the corresponding scene state is extracted. The paired receiver is the
/// only way to consume the result. After completion the component is inert and ordinary untracked
/// frames may continue; replace it with a fresh request to track another exact presentation.
///
/// A request is affine and cannot be duplicated across windows:
/// ```compile_fail
/// use bevy_render::view::window::WindowPresentationFeedbackRequest;
/// let (request, _receiver) = WindowPresentationFeedbackRequest::new(1);
/// let _duplicate = request.clone();
/// ```
#[derive(Component, Debug)]
pub struct WindowPresentationFeedbackRequest {
    serial: u64,
    state: Arc<Mutex<RequestState>>,
}

impl WindowPresentationFeedbackRequest {
    /// Create one request and its unique result receiver.
    pub fn new(serial: u64) -> (Self, WindowPresentationFeedbackReceiver) {
        let state = Arc::new(Mutex::new(RequestState::Requested));
        (
            Self {
                serial,
                state: Arc::clone(&state),
            },
            WindowPresentationFeedbackReceiver { serial, state },
        )
    }

    /// The caller-provided serial bound to this request.
    pub fn serial(&self) -> u64 {
        self.serial
    }

    pub(super) fn is_presenting(&self) -> bool {
        matches!(*self.state.lock().unwrap(), RequestState::Presenting)
    }

    pub(super) fn is_terminal(&self) -> bool {
        matches!(
            *self.state.lock().unwrap(),
            RequestState::Complete(_) | RequestState::Taken
        )
    }

    pub(super) fn clone_for_render(&self) -> Self {
        Self {
            serial: self.serial,
            state: Arc::clone(&self.state),
        }
    }

    pub(super) fn begin(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if *state != RequestState::Requested {
            return false;
        }
        *state = RequestState::Presenting;
        true
    }

    fn complete(&self, result: wgpu::PresentationFeedbackResult) {
        let mut state = self.state.lock().unwrap();
        if *state == RequestState::Presenting {
            *state = RequestState::Complete(result);
        } else {
            bevy_log::error!(
                "window presentation feedback request {} completed outside presentation",
                self.serial
            );
        }
    }
}

/// Unique receiver for one window presentation-feedback request.
#[derive(Debug)]
pub struct WindowPresentationFeedbackReceiver {
    serial: u64,
    state: Arc<Mutex<RequestState>>,
}

impl WindowPresentationFeedbackReceiver {
    /// The caller-provided serial bound to this receiver.
    pub fn serial(&self) -> u64 {
        self.serial
    }

    /// Consume the terminal result once it is available.
    ///
    /// Returns `None` while presentation is pending and after the result has already been taken.
    pub fn try_take(&mut self) -> Option<WindowPresentationFeedback> {
        let mut state = self.state.lock().unwrap();
        let RequestState::Complete(result) = *state else {
            return None;
        };
        *state = RequestState::Taken;
        Some(WindowPresentationFeedback {
            serial: self.serial,
            result,
        })
    }
}

/// Terminal feedback bound to one caller-provided request serial.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct WindowPresentationFeedback {
    serial: u64,
    result: wgpu::PresentationFeedbackResult,
}

impl WindowPresentationFeedback {
    /// The exact request serial associated with this result.
    pub fn serial(self) -> u64 {
        self.serial
    }

    /// The terminal wgpu presentation result for the requested frame.
    pub fn result(self) -> wgpu::PresentationFeedbackResult {
        self.result
    }
}

pub(super) struct PendingWindowPresentationFeedback {
    request: WindowPresentationFeedbackRequest,
    future: Option<wgpu::PresentationFeedbackFuture>,
}

impl PendingWindowPresentationFeedback {
    pub(super) fn new(
        request: WindowPresentationFeedbackRequest,
        future: wgpu::PresentationFeedbackFuture,
    ) -> Self {
        Self {
            request,
            future: Some(future),
        }
    }

    pub(super) fn poll(&mut self) -> bool {
        let Some(future) = self.future.as_mut() else {
            return true;
        };
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let Poll::Ready(result) = Pin::new(future).poll(&mut context) else {
            return false;
        };
        self.future = None;
        self.request.complete(result);
        true
    }
}

impl Drop for PendingWindowPresentationFeedback {
    fn drop(&mut self) {
        if self.future.is_some() {
            self.request
                .complete(Err(wgpu::PresentationFeedbackError::Cancelled));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_begins_once_and_receiver_consumes_exact_serial_once() {
        let (request, mut receiver) = WindowPresentationFeedbackRequest::new(41);
        let render_request = request.clone_for_render();
        assert_eq!(request.serial(), 41);
        assert_eq!(receiver.serial(), 41);
        assert!(!request.is_presenting());
        assert!(!request.is_terminal());
        assert!(render_request.begin());
        assert!(!render_request.begin());
        assert!(render_request.is_presenting());
        assert!(receiver.try_take().is_none());

        render_request.complete(Ok(wgpu::PresentationFeedback::NotPresented));
        assert!(!render_request.is_presenting());
        assert!(render_request.is_terminal());
        let feedback = receiver.try_take().unwrap();
        assert_eq!(feedback.serial(), 41);
        assert_eq!(
            feedback.result(),
            Ok(wgpu::PresentationFeedback::NotPresented)
        );
        assert!(receiver.try_take().is_none());
        assert!(render_request.is_terminal());
        assert!(!request.begin());
    }
}
