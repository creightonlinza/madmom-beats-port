# rhythm_ffi

C ABI wrapper around `rhythm_core`.

## Header

`include/rhythm.h` declares the public C API. The caller owns any strings
returned from the ABI and must free them with `rhythm_free_string`.

Versioned JSON schemas:

- `docs/schemas/config.v1.schema.json`
- `docs/schemas/analysis-output.v1.schema.json`
- `docs/schemas/config.v1.example.json`

`rhythm_analyze_json*` returns beat arrays:

```json
{
  "fps": 100,
  "beat_times": [0.2059, 0.645],
  "beat_numbers": [1, 2],
  "beat_confidences": [0.83, 0.79]
}
```

Ownership rules:

- `rhythm_analyze_json` returns a newly allocated C string; caller must call `rhythm_free_string`.
- `rhythm_last_error_message` returns a newly allocated C string; caller must call `rhythm_free_string`.
- `rhythm_last_error_json` returns a newly allocated C string; caller must call `rhythm_free_string`.
- `rhythm_default_config_json` returns a newly allocated C string; caller must call `rhythm_free_string`.
- Input samples remain owned by the caller.
- `rhythm_analyze_json_with_progress` accepts an optional progress callback. The callback
  is invoked on the same thread that calls the function.

## Build (local)

```bash
cd rust
cargo build -p rhythm_ffi --release
```

Outputs:

- `target/release/librhythm_ffi.a`
- `target/release/librhythm_ffi.dylib` (macOS)

## C usage example

`examples/example.c` shows basic usage and correct freeing.

Progress callback signature:

```c
typedef void (*rhythm_progress_cb)(uint32_t stage, float progress, void *user_data);
```

Stages:

- 0: features
- 1: inference
- 2: DBN decode

Cancellation:

- Not currently supported in the C ABI.
- Run analysis on a worker thread and cancel at the app/task level.

## Config helpers

```c
char *rhythm_default_config_json(void);
char *rhythm_validate_config_json(const char *config_json); // NULL on success
```

`rhythm_validate_config_json` returns an error JSON payload on failure:

```json
{
  "code": 4,
  "code_name": "CONFIG_VALIDATION_ERROR",
  "message": "invalid config: must be > 0",
  "path": "dbn.num_tempi",
  "context": null
}
```

## Structured error API

```c
uint32_t rhythm_last_error_code(void);
char *rhythm_last_error_json(void);
```

Error codes:

- 0: `OK`
- 1: `NULL_POINTER`
- 2: `UTF8_ERROR`
- 3: `CONFIG_PARSE_ERROR`
- 4: `CONFIG_VALIDATION_ERROR`
- 5: `INVALID_INPUT`
- 6: `MODEL_ERROR`
- 7: `IO_ERROR`
- 8: `NOT_IMPLEMENTED`
- 9: `JSON_ERROR`
- 10: `INTERNAL_ERROR`

## Swift usage snippet

```swift
// Import the C header via a bridging header:
// #include "rhythm.h"

let sampleRate: UInt32 = 44100
let samples: [Float] = loadYourSamples() // mono PCM
let jsonPtr = samples.withUnsafeBufferPointer { buf -> UnsafeMutablePointer<CChar>? in
    rhythm_analyze_json(buf.baseAddress, buf.count, sampleRate, nil)
}
if let jsonPtr {
    let json = String(cString: jsonPtr)
    rhythm_free_string(jsonPtr)
    print(json)
} else if let errPtr = rhythm_last_error_json() {
    let err = String(cString: errPtr)
    rhythm_free_string(errPtr)
    print("Error: \(err)")
}
```

## Android (JNI/NDK) snippet

```kotlin
// Kotlin: native method declaration
external fun rhythmAnalyzeJson(samples: FloatArray, sampleRate: Int, configJson: String?): String?

// JNI C/C++ glue (minimal)
#include <jni.h>
#include "rhythm.h"
JNIEXPORT jstring JNICALL
Java_com_example_Rhythm_rhythmAnalyzeJson(JNIEnv* env, jobject,
                                          jfloatArray samples,
                                          jint sampleRate,
                                          jstring configJson) {
    const jsize len = (*env)->GetArrayLength(env, samples);
    jfloat* data = (*env)->GetFloatArrayElements(env, samples, NULL);
    const char* cfg = configJson ? (*env)->GetStringUTFChars(env, configJson, NULL) : NULL;
    char* cfg_validation = rhythm_validate_config_json(cfg);
    if (cfg_validation) {
        jstring jerr = (*env)->NewStringUTF(env, cfg_validation);
        rhythm_free_string(cfg_validation);
        if (configJson) { (*env)->ReleaseStringUTFChars(env, configJson, cfg); }
        (*env)->ReleaseFloatArrayElements(env, samples, data, 0);
        return jerr;
    }
    char* out = rhythm_analyze_json(data, (size_t)len, (uint32_t)sampleRate, cfg);
    if (configJson) { (*env)->ReleaseStringUTFChars(env, configJson, cfg); }
    (*env)->ReleaseFloatArrayElements(env, samples, data, 0);
    if (!out) {
        char* err = rhythm_last_error_json();
        jstring jerr = (*env)->NewStringUTF(env, err ? err : "unknown error");
        rhythm_free_string(err);
        return jerr;
    }
    jstring jout = (*env)->NewStringUTF(env, out);
    rhythm_free_string(out);
    return jout;
}
```

## iOS

```bash
rustup target add aarch64-apple-ios x86_64-apple-ios
cargo build -p rhythm_ffi --release --target aarch64-apple-ios
cargo build -p rhythm_ffi --release --target x86_64-apple-ios
```

Link the static library into Xcode and include `include/rhythm.h`.

## Android

```bash
rustup target add aarch64-linux-android x86_64-linux-android
cargo build -p rhythm_ffi --release --target aarch64-linux-android
cargo build -p rhythm_ffi --release --target x86_64-linux-android
```

Use the resulting `librhythm_ffi.so` with JNI/NDK and include `include/rhythm.h`.

See `docs/android/jni-cmake-sample.md` for a complete JNI + CMake integration sketch.

Android build note:

- `rust/.cargo/config.toml` pins Android link args for `SONAME=librhythm_ffi.so`
  and `max-page-size=16384`.
