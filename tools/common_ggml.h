#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

struct common_ggml_t {
    ggml_backend * backend;
    ggml_backend * backend_cpu;
};

void init_ggml( common_ggml_t & ggml, const char * device = NULL, int n_threads = 0 ) {
    bool set_threads = false;
    ggml_backend_load_all();
    ggml_backend * backend, * backend_cpu;
    if ( device ) {
        backend = ggml_backend_init_by_name( device, NULL );
    } else {
        backend = ggml_backend_init_best();
    }
    if ( ! backend ) {
        if ( ! device ) device = "best";
        fprintf( stderr, "error: failed to initialize %s backend.\n", device );
        exit(1);
    }
    auto dev = ggml_backend_get_device( backend );
    if ( n_threads > 0 ) {
        auto reg = ggml_backend_dev_backend_reg( dev );
        auto set_n_threads = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if ( set_n_threads ) {
            set_n_threads( backend, n_threads );
            set_threads = true;
        }
    }

    if ( ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU ) {
        backend_cpu = backend;
    } else {
        backend_cpu = ggml_backend_init_by_type( GGML_BACKEND_DEVICE_TYPE_CPU, NULL );
        if ( ! backend_cpu ) {
            fprintf( stderr, "error: failed to initialize a cpu device.\n" );
            exit(1);
        }
        if ( n_threads > 0 ) {
            auto dev_cpu = ggml_backend_get_device( backend_cpu );
            auto reg_cpu = ggml_backend_dev_backend_reg( dev_cpu );
            auto set_n_threads_cpu = (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(reg_cpu, "ggml_backend_set_n_threads");
            if ( set_n_threads_cpu ) {
                set_n_threads_cpu( backend, n_threads );
            }
        }
    }

    auto dev_name = ggml_backend_dev_name( dev );
    printf( "using device: \"%s\"\n", dev_name );
    if ( set_threads ) {
        printf( "with threads: %d\n", n_threads );
    }

    ggml.backend = backend;
    ggml.backend_cpu = backend_cpu;
}


