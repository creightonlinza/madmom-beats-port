# Android JNI + CMake Sample (v3.0.0)

This is a minimal integration sketch for `rhythm_ffi` in an Android app.

## 1) Package native libraries

Place ABI folders under your app:

- `src/main/jniLibs/arm64-v8a/librhythm_ffi.so`
- `src/main/jniLibs/armeabi-v7a/librhythm_ffi.so`
- `src/main/jniLibs/x86_64/librhythm_ffi.so`

Also ship:

- `downbeats_blstm.json`
- `downbeats_blstm_weights.npz`

Copy model files to app-private storage before analysis.

## 2) CMake

```cmake
cmake_minimum_required(VERSION 3.22.1)
project(rhythm_bridge C)

add_library(rhythm_ffi SHARED IMPORTED)
set_target_properties(rhythm_ffi PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/librhythm_ffi.so")

add_library(rhythm_bridge SHARED rhythm_bridge.c)
target_include_directories(rhythm_bridge PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rhythm_bridge rhythm_ffi log)
```

## 3) JNI bridge

```c
#include <jni.h>
#include <stdint.h>
#include "rhythm.h"

JNIEXPORT jstring JNICALL
Java_com_example_rhythm_RhythmBridge_analyze(JNIEnv* env, jclass,
                                             jfloatArray samples,
                                             jint sample_rate,
                                             jstring config_json) {
  jsize len = (*env)->GetArrayLength(env, samples);
  jfloat* data = (*env)->GetFloatArrayElements(env, samples, NULL);
  const char* cfg = config_json ? (*env)->GetStringUTFChars(env, config_json, NULL) : NULL;

  char* validation = rhythm_validate_config_json(cfg);
  if (validation != NULL) {
    jstring out = (*env)->NewStringUTF(env, validation);
    rhythm_free_string(validation);
    if (config_json) (*env)->ReleaseStringUTFChars(env, config_json, cfg);
    (*env)->ReleaseFloatArrayElements(env, samples, data, JNI_ABORT);
    return out;
  }

  char* out_json = rhythm_analyze_json(data, (size_t)len, (uint32_t)sample_rate, cfg);
  if (config_json) (*env)->ReleaseStringUTFChars(env, config_json, cfg);
  (*env)->ReleaseFloatArrayElements(env, samples, data, JNI_ABORT);

  if (out_json == NULL) {
    char* err = rhythm_last_error_json();
    jstring out = (*env)->NewStringUTF(env, err ? err : "{\"code\":10,\"message\":\"unknown\"}");
    rhythm_free_string(err);
    return out;
  }

  jstring out = (*env)->NewStringUTF(env, out_json);
  rhythm_free_string(out_json);
  return out;
}
```

## 4) Kotlin side

```kotlin
object RhythmBridge {
  init {
    System.loadLibrary("rhythm_bridge")
  }

  external fun analyze(samples: FloatArray, sampleRate: Int, configJson: String?): String
}
```

Progress callback stage IDs are:

- `0`: features
- `1`: inference
- `2`: DBN decode

Cancellation is not currently supported in the C ABI. Run analysis on a worker thread and cancel at the app-task level.
