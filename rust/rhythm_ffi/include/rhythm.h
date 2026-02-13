#ifndef RHYTHM_FFI_H
#define RHYTHM_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Analyze mono PCM samples and return beat arrays as JSON:
// {
//   "fps": 100,
//   "beat_times": [...],
//   "beat_numbers": [...],
//   "beat_confidences": [...]
// }
// Returns NULL on error; call rhythm_last_error_message() for details.
// The returned string must be freed with rhythm_free_string().
char *rhythm_analyze_json(const float *samples,
                          size_t samples_len,
                          uint32_t sample_rate,
                          const char *config_json);

typedef void (*rhythm_progress_cb)(uint32_t stage, float progress, void *user_data);

// Analyze with optional progress callback (stage, progress 0..1).
// Returns NULL on error; call rhythm_last_error_message() for details.
// The returned string must be freed with rhythm_free_string().
char *rhythm_analyze_json_with_progress(const float *samples,
                                        size_t samples_len,
                                        uint32_t sample_rate,
                                        const char *config_json,
                                        rhythm_progress_cb progress_cb,
                                        void *user_data);

// Returns the last error message as a newly allocated string.
// The returned string must be freed with rhythm_free_string().
char *rhythm_last_error_message(void);

// Free a string returned by rhythm_analyze_json or rhythm_last_error_message.
void rhythm_free_string(char *s);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // RHYTHM_FFI_H
