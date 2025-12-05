#pragma once

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#define USE_FLOAT

struct sdl_frame_t {
    uint8_t * data;
    int nb_bytes;
    sdl_frame_t * prev;
    sdl_frame_t * next;
};

struct AudioState {
    SDL_AudioDeviceID device_id;
    pthread_mutex_t fifo_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t not_full = PTHREAD_COND_INITIALIZER;
    pthread_cond_t not_empty = PTHREAD_COND_INITIALIZER;

    sdl_frame_t * free = NULL;
    sdl_frame_t * head = NULL;
    sdl_frame_t * tail = NULL;

    bool log = false;
};

void sdl_init_frames( AudioState & state, int count, int nb_bytes ) {
    assert( count > 0 );
    for (int i = 0; i < count; i++) {
        state.free = new sdl_frame_t{
            new uint8_t[nb_bytes],
            nb_bytes,
            NULL,
            state.free
        };
    }
}

sdl_frame_t * sdl_get_frame( AudioState & state, bool block = true ) {
    pthread_mutex_lock(&state.fifo_mutex);
    auto frame = state.free;
    if ( block ) {
        while ( ! frame ) {
            pthread_cond_wait(&state.not_full, &state.fifo_mutex);
            frame = state.free;
        }
    }
    if ( frame ) {
        state.free = frame->next;
    }
    pthread_mutex_unlock(&state.fifo_mutex);
    return frame;
}

void sdl_free_frame( AudioState & state, sdl_frame_t * frame ) {
    pthread_mutex_lock(&state.fifo_mutex);
    frame->prev = NULL;
    frame->next = state.free;
    state.free = frame;
    pthread_cond_signal(&state.not_full);
    pthread_mutex_unlock(&state.fifo_mutex);
}

void sdl_send_frame( AudioState & state, sdl_frame_t * frame ) {
    pthread_mutex_lock(&state.fifo_mutex);
    frame->prev = NULL;
    frame->next = state.head;
    if ( ! state.head ) {
        assert ( ! state.tail );
        state.head = frame;
        state.tail = frame;
    } else {
        state.head->prev = frame;
        state.head = frame;
    }
    pthread_cond_signal(&state.not_empty);
    pthread_mutex_unlock(&state.fifo_mutex);
}

sdl_frame_t * sdl_receive_frame( AudioState & state, bool block ) {
    pthread_mutex_lock(&state.fifo_mutex);
    auto frame = state.tail;
    if ( block ) {
        while ( ! frame ) {
            pthread_cond_wait(&state.not_empty, &state.fifo_mutex);
            frame = state.tail;
        }
    }
    if ( frame ) {
        if (frame->prev == NULL) {
            assert( state.head == state.tail );
            state.head = NULL;
            state.tail = NULL;
        } else {
            state.tail = frame->prev;
            // maybe not set these for debug
            frame->prev = NULL;
            frame->next = NULL;
        }
    }
    pthread_mutex_unlock(&state.fifo_mutex);
    return frame;
}

void sdl_audio_callback(void *userdata, Uint8 *stream, int len) {
    AudioState  & state = *(AudioState *)userdata;
    sdl_frame_t * frame = sdl_receive_frame( state, false );
    if ( frame ) {
        memcpy( stream, frame->data, len );
        sdl_free_frame( state, frame );
    } else {
        memset( stream, 0, len );
    }
}

#ifdef USE_FLOAT
void sdl_capture_callback(void *userdata, Uint8 *stream, int len) {
    AudioState  & state = *(AudioState *)userdata;
    float max_amplitude = 0;
    auto samples = (float *)stream;
    for (int i = 0; i < len / 4; ++i) {
        float sample_value = samples[i];
        samples[i] *= 0.1f;
        sample_value = fabsf(sample_value);
        if (sample_value > max_amplitude) {
            max_amplitude = sample_value;
        }
    }

    sdl_frame_t * frame = sdl_get_frame( state, false );
    if ( frame ) {
        assert( len == frame->nb_bytes );
        memcpy( frame->data, stream, len );
        sdl_send_frame( state, frame );
    }

    if ( state.log ) {
        if ( frame ) {
            printf("\rMax Amplitude: %f      ", max_amplitude);
        } else {
            printf("\rno frame: %f           ", max_amplitude);

        }
        fflush(stdout);
    }
}
#else
void sdl_capture_callback(void *userdata, Uint8 *stream, int len) {
    int max_amplitude = 0;
    const int16_t *samples = (const int16_t *)stream;
    for (int i = 0; i < len / 2; ++i) {
        int sample_value = samples[i];
        if (abs(sample_value) > max_amplitude) {
            max_amplitude = abs(sample_value);
        }
    }

    if ( state.log ) {
        printf("\rMax Amplitude: %d      ", max_amplitude);
        fflush(stdout);
    }
}
#endif

