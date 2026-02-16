#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/madmom_beats_port.h"

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

    char *json = madmom_beats_port_analyze_json(samples, samples_len, 44100, NULL);
    if (!json) {
        char *err = madmom_beats_port_last_error_message();
        fprintf(stderr, "madmom_beats_port_analyze_json failed: %s\n", err ? err : "unknown error");
        madmom_beats_port_free_string(err);
        free(samples);
        return 1;
    }

    printf("Analysis JSON length: %zu\n", strlen(json));
    // Parse JSON beat_times/beat_numbers/beat_confidences with your JSON library.
    // This example only demonstrates ownership and freeing.

    madmom_beats_port_free_string(json);
    free(samples);
    return 0;
}
