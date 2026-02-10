use rhythm_core::{analyze, analyze_with_progress, CoreConfig, ProgressEvent, ProgressSink};
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::os::raw::{c_char, c_float, c_uint};
use std::sync::{Mutex, OnceLock};

static LAST_ERROR: OnceLock<Mutex<String>> = OnceLock::new();

fn set_last_error(err: &str) {
    let lock = LAST_ERROR.get_or_init(|| Mutex::new(String::new()));
    if let Ok(mut guard) = lock.lock() {
        *guard = err.to_string();
    }
}

#[no_mangle]
pub extern "C" fn rhythm_last_error_message() -> *mut c_char {
    let lock = LAST_ERROR.get_or_init(|| Mutex::new(String::new()));
    match lock.lock() {
        Ok(guard) => CString::new(guard.as_str()).unwrap_or_default().into_raw(),
        Err(_) => CString::new("failed to lock error mutex")
            .unwrap()
            .into_raw(),
    }
}

#[no_mangle]
/// # Safety
/// The pointer must be either null or previously returned by this library and
/// not already freed.
pub unsafe extern "C" fn rhythm_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    drop(CString::from_raw(s));
}

#[no_mangle]
/// # Safety
/// - `samples_ptr` must be valid for reads of `samples_len` floats.
/// - `config_json` must be null or a valid, null-terminated UTF-8 string.
pub unsafe extern "C" fn rhythm_analyze_json(
    samples_ptr: *const c_float,
    samples_len: usize,
    sample_rate: c_uint,
    config_json: *const c_char,
) -> *mut c_char {
    if samples_ptr.is_null() {
        set_last_error("samples_ptr is null");
        return std::ptr::null_mut();
    }

    let config = parse_config(config_json);
    let config = match config {
        Ok(c) => c,
        Err(err) => {
            set_last_error(&err);
            return std::ptr::null_mut();
        }
    };

    let samples = std::slice::from_raw_parts(samples_ptr, samples_len);
    match analyze(samples, sample_rate, &config) {
        Ok(output) => match serde_json::to_string(&output) {
            Ok(json) => CString::new(json).unwrap_or_default().into_raw(),
            Err(err) => {
                set_last_error(&err.to_string());
                std::ptr::null_mut()
            }
        },
        Err(err) => {
            set_last_error(&format!("{}", err));
            std::ptr::null_mut()
        }
    }
}

type ProgressCallback = Option<extern "C" fn(u32, f32, *mut c_void)>;

struct FfiProgress {
    cb: ProgressCallback,
    user_data: *mut c_void,
}

impl ProgressSink for FfiProgress {
    fn on_progress(&mut self, event: ProgressEvent) {
        if let Some(cb) = self.cb {
            cb(event.stage as u32, event.progress, self.user_data);
        }
    }
}

#[no_mangle]
/// # Safety
/// - `samples_ptr` must be valid for reads of `samples_len` floats.
/// - `config_json` must be null or a valid, null-terminated UTF-8 string.
/// - `progress_cb` must be safe to call from this thread if provided.
pub unsafe extern "C" fn rhythm_analyze_json_with_progress(
    samples_ptr: *const c_float,
    samples_len: usize,
    sample_rate: c_uint,
    config_json: *const c_char,
    progress_cb: ProgressCallback,
    user_data: *mut c_void,
) -> *mut c_char {
    if samples_ptr.is_null() {
        set_last_error("samples_ptr is null");
        return std::ptr::null_mut();
    }

    let config = parse_config(config_json);
    let config = match config {
        Ok(c) => c,
        Err(err) => {
            set_last_error(&err);
            return std::ptr::null_mut();
        }
    };

    let samples = std::slice::from_raw_parts(samples_ptr, samples_len);
    let mut sink = FfiProgress {
        cb: progress_cb,
        user_data,
    };
    match analyze_with_progress(samples, sample_rate, &config, Some(&mut sink)) {
        Ok(output) => match serde_json::to_string(&output) {
            Ok(json) => CString::new(json).unwrap_or_default().into_raw(),
            Err(err) => {
                set_last_error(&err.to_string());
                std::ptr::null_mut()
            }
        },
        Err(err) => {
            set_last_error(&format!("{}", err));
            std::ptr::null_mut()
        }
    }
}

unsafe fn parse_config(config_json: *const c_char) -> Result<CoreConfig, String> {
    if config_json.is_null() {
        return Ok(CoreConfig::default());
    }
    let cstr = CStr::from_ptr(config_json);
    let json = cstr.to_str().map_err(|e| e.to_string())?;
    if json.trim().is_empty() {
        return Ok(CoreConfig::default());
    }
    serde_json::from_str(json).map_err(|e| e.to_string())
}
