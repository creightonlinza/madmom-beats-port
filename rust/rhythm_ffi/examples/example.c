#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/rhythm.h"

static float *load_samples_stub(size_t *out_len) {
    *out_len = 0;
    return NULL;
}

int main(void) {
    size_t samples_len = 0;
    float *samples = load_samples_stub(&samples_len);
    if (samples_len == 0 || samples == NULL) {
        printf("No samples loaded (stub). Replace load_samples_stub with real audio loading.\n");
        return 1;
    }

    char *json = rhythm_analyze_json(samples, samples_len, 44100, NULL);
    if (!json) {
        char *err = rhythm_last_error_message();
        fprintf(stderr, "rhythm_analyze_json failed: %s\n", err ? err : "unknown error");
        rhythm_free_string(err);
        free(samples);
        return 1;
    }

    printf("Analysis JSON length: %zu\n", strlen(json));
    // Parse JSON and count beats/downbeats using your JSON library of choice.
    // This example only demonstrates ownership and freeing.

    rhythm_free_string(json);
    free(samples);
    return 0;
}
