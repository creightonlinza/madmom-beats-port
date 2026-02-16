use madmom_beats_port_core::{
    analyze, analyze_with_progress, validate_core_config, CoreConfig, ProgressEvent, ProgressSink,
    RhythmError,
};
use serde::Serialize;
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::os::raw::{c_char, c_float, c_uint};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, Default)]
#[repr(u32)]
enum FfiErrorCode {
    #[default]
    Ok = 0,
    NullPointer = 1,
    Utf8 = 2,
    ConfigParse = 3,
    ConfigValidation = 4,
    InvalidInput = 5,
    Model = 6,
    Io = 7,
    NotImplemented = 8,
    Json = 9,
    Internal = 10,
}

impl FfiErrorCode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::NullPointer => "NULL_POINTER",
            Self::Utf8 => "UTF8_ERROR",
            Self::ConfigParse => "CONFIG_PARSE_ERROR",
            Self::ConfigValidation => "CONFIG_VALIDATION_ERROR",
            Self::InvalidInput => "INVALID_INPUT",
            Self::Model => "MODEL_ERROR",
            Self::Io => "IO_ERROR",
            Self::NotImplemented => "NOT_IMPLEMENTED",
            Self::Json => "JSON_ERROR",
            Self::Internal => "INTERNAL_ERROR",
        }
    }
}

#[derive(Debug, Clone, Default)]
struct LastErrorState {
    code: FfiErrorCode,
    message: String,
    path: Option<String>,
    context: Option<String>,
}

#[derive(Debug)]
struct FfiFailure {
    code: FfiErrorCode,
    message: String,
    path: Option<String>,
    context: Option<String>,
}

impl FfiFailure {
    fn new(
        code: FfiErrorCode,
        message: impl Into<String>,
        path: Option<impl Into<String>>,
        context: Option<impl Into<String>>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            path: path.map(|p| p.into()),
            context: context.map(|c| c.into()),
        }
    }
}

#[derive(Serialize)]
struct LastErrorPayload<'a> {
    code: u32,
    code_name: &'static str,
    message: &'a str,
    path: Option<&'a str>,
    context: Option<&'a str>,
}

static LAST_ERROR: OnceLock<Mutex<LastErrorState>> = OnceLock::new();

fn set_last_error(code: FfiErrorCode, message: impl Into<String>) {
    set_last_error_details(code, message, None::<String>, None::<String>);
}

fn set_last_error_details(
    code: FfiErrorCode,
    message: impl Into<String>,
    path: Option<impl Into<String>>,
    context: Option<impl Into<String>>,
) {
    let lock = LAST_ERROR.get_or_init(|| Mutex::new(LastErrorState::default()));
    if let Ok(mut guard) = lock.lock() {
        guard.code = code;
        guard.message = message.into();
        guard.path = path.map(|p| p.into());
        guard.context = context.map(|c| c.into());
    }
}

fn set_last_error_from_failure(failure: FfiFailure) {
    set_last_error_details(failure.code, failure.message, failure.path, failure.context);
}

fn set_last_error_from_core_error(err: &RhythmError) {
    let code = match err {
        RhythmError::InvalidInput(_) => FfiErrorCode::InvalidInput,
        RhythmError::Model(_) => FfiErrorCode::Model,
        RhythmError::Io(_) => FfiErrorCode::Io,
        RhythmError::NotImplemented(_) => FfiErrorCode::NotImplemented,
    };
    set_last_error(code, err.to_string());
}

fn clear_last_error() {
    set_last_error(FfiErrorCode::Ok, "");
}

fn get_last_error_snapshot() -> LastErrorState {
    let lock = LAST_ERROR.get_or_init(|| Mutex::new(LastErrorState::default()));
    match lock.lock() {
        Ok(guard) => guard.clone(),
        Err(_) => LastErrorState {
            code: FfiErrorCode::Internal,
            message: "failed to lock error mutex".to_string(),
            path: None,
            context: None,
        },
    }
}

fn last_error_json_string() -> String {
    let state = get_last_error_snapshot();
    let payload = LastErrorPayload {
        code: state.code as u32,
        code_name: state.code.as_str(),
        message: state.message.as_str(),
        path: state.path.as_deref(),
        context: state.context.as_deref(),
    };
    serde_json::to_string(&payload).unwrap_or_else(|_| {
        "{\"code\":10,\"code_name\":\"INTERNAL_ERROR\",\"message\":\"failed to serialize last error\",\"path\":null,\"context\":null}".to_string()
    })
}

fn into_c_string_ptr(s: String) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

#[no_mangle]
pub extern "C" fn madmom_beats_port_last_error_message() -> *mut c_char {
    let state = get_last_error_snapshot();
    into_c_string_ptr(state.message)
}

#[no_mangle]
pub extern "C" fn madmom_beats_port_last_error_code() -> c_uint {
    let state = get_last_error_snapshot();
    state.code as c_uint
}

#[no_mangle]
pub extern "C" fn madmom_beats_port_last_error_json() -> *mut c_char {
    into_c_string_ptr(last_error_json_string())
}

#[no_mangle]
pub extern "C" fn madmom_beats_port_default_config_json() -> *mut c_char {
    match serde_json::to_string_pretty(&CoreConfig::default()) {
        Ok(json) => into_c_string_ptr(json),
        Err(err) => {
            set_last_error(
                FfiErrorCode::Json,
                format!("failed to serialize default config: {err}"),
            );
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
/// # Safety
/// - `config_json` must be null or a valid, null-terminated UTF-8 string.
///
/// Returns NULL on success. On validation error, returns a newly allocated JSON
/// payload and also updates madmom_beats_port_last_error_*.
pub unsafe extern "C" fn madmom_beats_port_validate_config_json(
    config_json: *const c_char,
) -> *mut c_char {
    match parse_config(config_json) {
        Ok(_) => {
            clear_last_error();
            std::ptr::null_mut()
        }
        Err(failure) => {
            set_last_error_from_failure(failure);
            into_c_string_ptr(last_error_json_string())
        }
    }
}

#[no_mangle]
/// # Safety
/// The pointer must be either null or previously returned by this library and
/// not already freed.
pub unsafe extern "C" fn madmom_beats_port_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    drop(CString::from_raw(s));
}

#[no_mangle]
/// # Safety
/// - `samples_ptr` must be valid for reads of `samples_len` floats.
/// - `config_json` must be null or a valid, null-terminated UTF-8 string.
pub unsafe extern "C" fn madmom_beats_port_analyze_json(
    samples_ptr: *const c_float,
    samples_len: usize,
    sample_rate: c_uint,
    config_json: *const c_char,
) -> *mut c_char {
    if samples_ptr.is_null() {
        set_last_error_details(
            FfiErrorCode::NullPointer,
            "samples_ptr is null",
            Some("samples_ptr"),
            None::<String>,
        );
        return std::ptr::null_mut();
    }

    let config = match parse_config(config_json) {
        Ok(c) => c,
        Err(err) => {
            set_last_error_from_failure(err);
            return std::ptr::null_mut();
        }
    };

    let samples = std::slice::from_raw_parts(samples_ptr, samples_len);
    match analyze(samples, sample_rate, &config) {
        Ok(output) => match serde_json::to_string(&output) {
            Ok(json) => {
                clear_last_error();
                into_c_string_ptr(json)
            }
            Err(err) => {
                set_last_error(FfiErrorCode::Json, err.to_string());
                std::ptr::null_mut()
            }
        },
        Err(err) => {
            set_last_error_from_core_error(&err);
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
pub unsafe extern "C" fn madmom_beats_port_analyze_json_with_progress(
    samples_ptr: *const c_float,
    samples_len: usize,
    sample_rate: c_uint,
    config_json: *const c_char,
    progress_cb: ProgressCallback,
    user_data: *mut c_void,
) -> *mut c_char {
    if samples_ptr.is_null() {
        set_last_error_details(
            FfiErrorCode::NullPointer,
            "samples_ptr is null",
            Some("samples_ptr"),
            None::<String>,
        );
        return std::ptr::null_mut();
    }

    let config = match parse_config(config_json) {
        Ok(c) => c,
        Err(err) => {
            set_last_error_from_failure(err);
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
            Ok(json) => {
                clear_last_error();
                into_c_string_ptr(json)
            }
            Err(err) => {
                set_last_error(FfiErrorCode::Json, err.to_string());
                std::ptr::null_mut()
            }
        },
        Err(err) => {
            set_last_error_from_core_error(&err);
            std::ptr::null_mut()
        }
    }
}

fn validate_config(config: CoreConfig) -> Result<CoreConfig, FfiFailure> {
    validate_core_config(&config).map_err(|issue| {
        FfiFailure::new(
            FfiErrorCode::ConfigValidation,
            format!("invalid config: {}", issue.message),
            Some(issue.path),
            None::<String>,
        )
    })?;
    Ok(config)
}

unsafe fn parse_config(config_json: *const c_char) -> Result<CoreConfig, FfiFailure> {
    if config_json.is_null() {
        return validate_config(CoreConfig::default());
    }

    let cstr = CStr::from_ptr(config_json);
    let json = cstr.to_str().map_err(|err| {
        FfiFailure::new(
            FfiErrorCode::Utf8,
            format!("config_json is not valid UTF-8: {err}"),
            Some("config_json"),
            None::<String>,
        )
    })?;
    if json.trim().is_empty() {
        return validate_config(CoreConfig::default());
    }

    let config = serde_json::from_str::<CoreConfig>(json).map_err(|err| {
        let context = if err.line() > 0 {
            Some(format!("line {}, column {}", err.line(), err.column()))
        } else {
            None
        };
        FfiFailure::new(
            FfiErrorCode::ConfigParse,
            format!("failed to parse config_json: {err}"),
            Some("config_json"),
            context,
        )
    })?;

    validate_config(config)
}
