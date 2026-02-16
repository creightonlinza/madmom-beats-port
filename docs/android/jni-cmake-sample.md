# Android JNI + CMake Sample (v4.0.0)

This is a minimal integration sketch for `madmom_beats_port_ffi` in an Android app.

## 1) Package native libraries

Place ABI folders under your app:

- `src/main/jniLibs/arm64-v8a/libmadmom_beats_port_ffi.so`
- `src/main/jniLibs/armeabi-v7a/libmadmom_beats_port_ffi.so`
- `src/main/jniLibs/x86_64/libmadmom_beats_port_ffi.so`

Also ship:

- `downbeats_blstm.json`
- `downbeats_blstm_weights.npz`

Copy model files to app-private storage before analysis.

## 2) CMake

```cmake
cmake_minimum_required(VERSION 3.22.1)
project(madmom_beats_port_bridge C)

add_library(madmom_beats_port_ffi SHARED IMPORTED)
set_target_properties(madmom_beats_port_ffi PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libmadmom_beats_port_ffi.so")

add_library(madmom_beats_port_bridge SHARED madmom_beats_port_bridge.c)
target_include_directories(madmom_beats_port_bridge PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(madmom_beats_port_bridge madmom_beats_port_ffi log)
```

## 3) JNI bridge

```c
#include <jni.h>
#include <stdint.h>
#include "madmom_beats_port.h"

JNIEXPORT jstring JNICALL
Java_com_example_madmom_beats_port_RhythmBridge_analyze(JNIEnv* env, jclass,
                                             jfloatArray samples,
                                             jint sample_rate,
                                             jstring config_json) {
  jsize len = (*env)->GetArrayLength(env, samples);
  jfloat* data = (*env)->GetFloatArrayElements(env, samples, NULL);
  const char* cfg = config_json ? (*env)->GetStringUTFChars(env, config_json, NULL) : NULL;

  char* validation = madmom_beats_port_validate_config_json(cfg);
  if (validation != NULL) {
    jstring out = (*env)->NewStringUTF(env, validation);
    madmom_beats_port_free_string(validation);
    if (config_json) (*env)->ReleaseStringUTFChars(env, config_json, cfg);
    (*env)->ReleaseFloatArrayElements(env, samples, data, JNI_ABORT);
    return out;
  }

  char* out_json = madmom_beats_port_analyze_json(data, (size_t)len, (uint32_t)sample_rate, cfg);
  if (config_json) (*env)->ReleaseStringUTFChars(env, config_json, cfg);
  (*env)->ReleaseFloatArrayElements(env, samples, data, JNI_ABORT);

  if (out_json == NULL) {
    char* err = madmom_beats_port_last_error_json();
    jstring out = (*env)->NewStringUTF(env, err ? err : "{\"code\":10,\"message\":\"unknown\"}");
    madmom_beats_port_free_string(err);
    return out;
  }

  jstring out = (*env)->NewStringUTF(env, out_json);
  madmom_beats_port_free_string(out_json);
  return out;
}
```

## 4) Kotlin side

```kotlin
object RhythmBridge {
  init {
    System.loadLibrary("madmom_beats_port_bridge")
  }

  external fun analyze(samples: FloatArray, sampleRate: Int, configJson: String?): String
}
```

Progress callback stage IDs are:

- `0`: features
- `1`: inference
- `2`: DBN decode

Cancellation is not currently supported in the C ABI. Run analysis on a worker thread and cancel at the app-task level.
