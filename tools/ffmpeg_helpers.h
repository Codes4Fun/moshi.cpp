#pragma once

#include <stdexcept>

// Include the main FFmpeg library headers
extern "C" {
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
//#include <libavutil/opt.h>
//#include <libavutil/channel_layout.h>
//#include <libavutil/samplefmt.h>
//#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
//#include "../../FFmpeg/libswresample/swresample_internal.h"
}

// simple functions to check for errors and print a message
static void on_error(int ret, const char *message) { // break on this function
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
    throw std::runtime_error( errbuf );
}
static void check_error(int ret, const char *message) {
    if (ret < 0) {
        on_error( ret, message );
    }
}

void unref( AVFormatContext * format_ctx ) {
    avformat_close_input( &format_ctx );
}

void unref( AVFrame * frame ) {
    av_frame_free( &frame );
}

void unref( AVPacket *packet ) {
    av_packet_free( &packet );
}

void unref( AVCodecContext *codec_ctx ) {
    avcodec_free_context( &codec_ctx );
}

void unref( SwrContext *swr_ctx ) {
    swr_free( &swr_ctx );
}

class Decoder {
public:
    unref_ptr<AVFormatContext> format_ctx;
    int audio_stream_index;
    unref_ptr<AVCodecContext> codec_ctx;
    unref_ptr<AVPacket> packet;
    unref_ptr<AVFrame> dec_frame;
    AVPacket * prev_packet;

    Decoder() {
        audio_stream_index = -1;
        prev_packet = NULL;
    }

    ~Decoder() {
        if ( prev_packet ) {
            av_packet_unref( prev_packet );
            prev_packet = NULL;
        }
    }

    void init(const char * filename) {
        // Open the input file and read its header.
        AVFormatContext * format_ctx = NULL;
        check_error( avformat_open_input( &format_ctx, filename, NULL, NULL ),
            filename );
        this->format_ctx = format_ctx;

        // Get stream information
        check_error(avformat_find_stream_info(format_ctx, NULL),
            "Could not retrieve stream info");

        // --- 2. Find the audio stream ---
        check_error(
            audio_stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0),
            "Could not find an audio stream\n"
        );

        auto stream = format_ctx->streams[audio_stream_index];
        const AVCodecParameters * params = stream->codecpar;
        const AVCodec * codec = avcodec_find_decoder(params->codec_id);
        codec_ctx = avcodec_alloc_context3(codec);
        check_error(avcodec_parameters_to_context(codec_ctx, params), "Failed to copy input codec params");
        check_error(avcodec_open2(codec_ctx, codec, NULL), "Failed to open input codec");

        codec_ctx->pkt_timebase = stream->time_base;

        // if not specified use default
        if ( codec_ctx->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC ) {
            av_channel_layout_default(&codec_ctx->ch_layout, codec_ctx->ch_layout.nb_channels);
        }

        packet = av_packet_alloc();
        prev_packet = NULL;
        dec_frame = av_frame_alloc();
    }

    AVFrame * frame() {
        if (prev_packet) {
            if (avcodec_receive_frame(codec_ctx, dec_frame) == 0) {
                dec_frame->pts = dec_frame->best_effort_timestamp;
                return dec_frame;
            }
            av_packet_unref(prev_packet);
            prev_packet = NULL;
        }
        while( av_read_frame(format_ctx, packet) >= 0 ) {
            if (packet->stream_index == audio_stream_index) {
                check_error(
                    avcodec_send_packet(codec_ctx, packet),
                    "Error sending packet to decoder"
                );
                if (avcodec_receive_frame(codec_ctx, dec_frame) == 0) {
                    prev_packet = packet;
                    dec_frame->pts = dec_frame->best_effort_timestamp;
                    return dec_frame;
                }
            }
            av_packet_unref(packet);
        }
        return NULL;
    }

};

class Resampler {
public:
    bool input_set;
    bool output_set;
    bool is_init;
    int in_sample_rate;
    int frame_size;
    unref_ptr<SwrContext> swr_ctx;
    unref_ptr<AVFrame> swr_frame;

    Resampler() {
        input_set = false;
        output_set = false;
        is_init = false;
        swr_ctx = swr_alloc();
        assert( swr_ctx );
        swr_frame = av_frame_alloc();
        assert( swr_frame );
    }

    void set_input(
        int sample_rate,
        AVSampleFormat sample_fmt,
        AVChannelLayout & ch_layout,
        int frame_size = 0
    ) {
        assert( ! input_set );
        assert( ch_layout.order != AV_CHANNEL_ORDER_UNSPEC ); // needs to be specified
        in_sample_rate = sample_rate;
        // Set input parameters (from the decoder)
        check_error( av_opt_set_chlayout(swr_ctx, "in_chlayout", &ch_layout, 0), "chlayout" );
        check_error( av_opt_set_int(swr_ctx, "in_sample_rate",    sample_rate,    0), "sample_rate" );
        check_error( av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", sample_fmt, 0), "sample_fmt" );
        this->frame_size = frame_size;
        input_set = true;
    }

    void set_input( AVCodecContext * codec_ctx ) {
        set_input(
            codec_ctx->sample_rate,
            codec_ctx->sample_fmt,
            codec_ctx->ch_layout
        );
    }

    void set_output(
        int sample_rate,
        AVSampleFormat sample_fmt,
        AVChannelLayout & ch_layout,
        int frame_size
    ) {
        assert( ! output_set );
        //this->sample_rate = sample_rate;
        // Set output parameters (for the encoder)
        check_error( av_opt_set_chlayout(swr_ctx, "out_chlayout", &ch_layout, 0), "chlayout" );
        check_error( av_opt_set_int(swr_ctx, "out_sample_rate",    sample_rate,    0), "sample_rate" );
        check_error( av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", sample_fmt, 0), "sample_fmt" );
        // create frame
        if ( frame_size == 0 ) {
            assert( input_set );
            frame_size = this->frame_size;
        }
        swr_frame->nb_samples     = frame_size;
        swr_frame->ch_layout      = ch_layout;
        swr_frame->format         = sample_fmt;
        swr_frame->sample_rate    = sample_rate;
        check_error( av_frame_get_buffer( swr_frame, 0 ),
            "Error making frame buffer" );
        output_set = true;
    }

    void set_output( AVCodecContext * codec_ctx ) {
        set_output(
            codec_ctx->sample_rate,
            codec_ctx->sample_fmt,
            codec_ctx->ch_layout,
            codec_ctx->frame_size
        );
    }

    void init() {
        assert( !is_init );
        assert( input_set );
        assert( output_set );
        // Initialize the context
        check_error( swr_init(swr_ctx), "swr_init failed" );
        is_init = true;
    }

    AVFrame * frame( AVFrame * frame = NULL ) {
        assert( is_init );
        if ( frame ) {
            check_error( swr_convert_frame( swr_ctx, NULL, frame ),
                "Error resampling" );
        }
        int nb_samples = swr_get_delay( swr_ctx, swr_frame->sample_rate );
        if ( nb_samples < swr_frame->nb_samples )
            return NULL;
        check_error( swr_convert_frame(swr_ctx, swr_frame, NULL),
            "Error resampling" );
        return swr_frame;
    }

    AVFrame * flush( bool inject_silence = false ) {
        int nb_samples = swr_get_delay( swr_ctx, swr_frame->sample_rate );
        if ( nb_samples < 1 && ! inject_silence )
            return NULL;
        if ( inject_silence ) {
            int nb_silence = swr_frame->nb_samples - nb_samples;
            swr_inject_silence( swr_ctx, nb_silence * in_sample_rate / swr_frame->sample_rate );
        }
        check_error( swr_convert_frame(swr_ctx, swr_frame, NULL),
            "Error resampling" );
        return swr_frame;
    }
};


class Encoder {
public:
    unref_ptr<AVFormatContext> format_ctx;
    unref_ptr<AVCodecContext> codec_ctx;
    AVStream * stream;
    int64_t pts_counter;

    Encoder() {}

    void init_from(
        const char * filename,
        int in_sample_rate,
        AVSampleFormat in_sample_fmt,
        AVChannelLayout & in_ch_layout
    ) {
        // --- OUTPUT SETUP ---
        auto format = av_guess_format (NULL, filename, NULL );
        assert( format );

        auto format_ctx = (AVFormatContext*)NULL;
        check_error(
            avformat_alloc_output_context2(&format_ctx, format, NULL, filename),
            "Could not create output context");
        assert( format_ctx );
        this->format_ctx = format_ctx;

        auto id = av_guess_codec( format, NULL, filename, NULL, AVMEDIA_TYPE_AUDIO );

        // Find and open the ENCODER for the output format
        const AVCodec *codec = avcodec_find_encoder(id);
        if (!codec) {
            fprintf(stderr, "Could not find encoder for %s\n", avcodec_get_name(AV_CODEC_ID_PCM_S16LE));
            //goto cleanup;
            exit(0);
        }

        codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            fprintf(stderr, "Failed to allocate output codec context\n");
            //goto cleanup;
            exit(0);
        }

        const int                *formats;
        const int                *sample_rates;
        const AVChannelLayout    *ch_layouts;

        check_error( avcodec_get_supported_config(
                codec_ctx, NULL,
                AV_CODEC_CONFIG_SAMPLE_FORMAT, 0,
                (const void **) &formats, NULL),
                "avcodec_get_supported_config");
        check_error( avcodec_get_supported_config(
                codec_ctx, NULL,
                AV_CODEC_CONFIG_SAMPLE_RATE, 0,
                (const void **) &sample_rates, NULL),
                "avcodec_get_supported_config");
        check_error( avcodec_get_supported_config(
                codec_ctx, NULL,
                AV_CODEC_CONFIG_CHANNEL_LAYOUT, 0,
                (const void **) &ch_layouts, NULL),
                "avcodec_get_supported_config");
        int nformats = 0;
        int nrates = 0;
        int nlayouts = 0;
        printf("formats:\n");
        for (; nformats < 18 && formats[nformats] != AV_SAMPLE_FMT_NONE; nformats++) {
            auto format = (AVSampleFormat)formats[nformats];
            printf("%d %s\n", format, av_get_sample_fmt_name(format));
        }
        if ( sample_rates ) {
            printf("sample rates:\n");
            for (; nrates < 18 && sample_rates[nrates]; nrates++) {
                printf("%d\n", sample_rates[nrates]);
            }
        }
        if ( ch_layouts ) {
            for (; nlayouts < 18 && ch_layouts[nlayouts].nb_channels != 0; nlayouts++) {
                auto & layout = ch_layouts[nlayouts];
                printf("%d %d %ld\n", layout.order, layout.nb_channels, layout.u.mask);
            }
        }

        int planar = av_sample_fmt_is_planar( in_sample_fmt );
        auto alt_sample_fmt = av_get_alt_sample_fmt( in_sample_fmt, 1 - planar );
        auto sample_fmt = (AVSampleFormat)formats[0];
        for (int i = 0; i < nformats; i++) {
            if (formats[i] == in_sample_fmt) {
                sample_fmt = in_sample_fmt;
                break;
            } else if (formats[i] == alt_sample_fmt) {
                sample_fmt = alt_sample_fmt;
            }
        }

        codec_ctx->sample_fmt = sample_fmt;
        codec_ctx->sample_rate = in_sample_rate;
        if ( ch_layouts ) {
            int layout_found = 0;
            for (int i = 0; i < nlayouts; i++) {
                if (ch_layouts[i].nb_channels == in_ch_layout.nb_channels) {
                    layout_found = i;
                    break;
                } else if (ch_layouts[i].nb_channels < in_ch_layout.nb_channels) {
                    layout_found = i;
                }
            }
            av_channel_layout_copy(
                &codec_ctx->ch_layout,
                &ch_layouts[layout_found]
            );
        } else {
            // I guess that means it supports any channel layout?
            av_channel_layout_copy( &codec_ctx->ch_layout, &in_ch_layout );
        }
        codec_ctx->time_base = (AVRational){1, in_sample_rate};
        
        // Some formats require a global header
        if (format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
            codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        check_error(avcodec_open2(codec_ctx, codec, NULL), "Could not open output codec");

        // Create a new audio stream in the output file
        stream = avformat_new_stream(format_ctx, codec);
        if (!stream) {
            fprintf(stderr, "Failed to create output stream\n");
            //goto cleanup;
            exit(0);
        }
        check_error(avcodec_parameters_from_context(stream->codecpar, codec_ctx), "Failed to copy output codec params");
        stream->time_base = codec_ctx->time_base;

        // Open the output file for writing
        if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
            check_error(avio_open(&format_ctx->pb, filename, AVIO_FLAG_WRITE), "Could not open output file");
        }
        check_error(avformat_write_header(format_ctx, NULL), "Error writing header");

        pts_counter = 0;
    }

    void init_from(const char * filename, AVCodecContext * other) {
        auto in_sample_rate = other->sample_rate;
        auto in_sample_fmt = other->sample_fmt;
        auto & in_ch_layout = other->ch_layout;
        init_from( filename, in_sample_rate, in_sample_fmt, in_ch_layout );
    }

    void frame( AVFrame * frame ) {
        frame->pts = av_rescale_q(
            pts_counter,
            (AVRational){ 1, codec_ctx->sample_rate },
            codec_ctx->time_base );
        pts_counter += frame->nb_samples;

        check_error(
            avcodec_send_frame( codec_ctx, frame ),
            "Error sending frame to encoder"
        );

        AVPacket *packet = av_packet_alloc();
        while (avcodec_receive_packet(codec_ctx, packet) == 0) {
            // write data
            av_packet_rescale_ts(packet, codec_ctx->time_base, stream->time_base);
            packet->stream_index = stream->index;
            check_error(av_interleaved_write_frame(format_ctx, packet), "Error writing packet to output");
            av_packet_unref(packet);
        }
        av_packet_free(&packet);
    }

    void flush() {
        check_error(avcodec_send_frame(codec_ctx, NULL), "Error flushing encoder");
        AVPacket *packet = av_packet_alloc();
        while (avcodec_receive_packet(codec_ctx, packet) == 0) {
            av_packet_rescale_ts(packet, codec_ctx->time_base, stream->time_base);
            packet->stream_index = stream->index;
            check_error(av_interleaved_write_frame(format_ctx, packet), "Error writing flush packet to output");
            av_packet_unref(packet);
        }
        av_packet_free(&packet);

        check_error(av_write_trailer(format_ctx), "Error writing trailer");
    }
};



class ResampleEncoder {
public:
    AVFormatContext *format_ctx;
    AVCodecContext *pOutCodecContext;
    AVStream *pOutStream;
    AVFrame *enc_frame = av_frame_alloc();
    SwrContext *swr_ctx;
    int64_t pts_counter;

    ResampleEncoder() {}

    void init(const char * filename, AVCodecContext * other) {
        // --- OUTPUT SETUP ---

        auto format = av_guess_format (NULL, filename, NULL );
        assert( format );

        format_ctx = NULL;
        check_error(
            avformat_alloc_output_context2(&format_ctx, format, NULL, filename),
            "Could not create output context");
        assert( format_ctx );

        auto id = av_guess_codec( format, NULL, filename, NULL, AVMEDIA_TYPE_AUDIO );

        // Find and open the ENCODER for the output format
        const AVCodec *pOutCodec = avcodec_find_encoder(id);
        if (!pOutCodec) {
            fprintf(stderr, "Could not find encoder for %s\n", avcodec_get_name(AV_CODEC_ID_PCM_S16LE));
            //goto cleanup;
            exit(0);
        }

        pOutCodecContext = avcodec_alloc_context3(pOutCodec);
        if (!pOutCodecContext) {
            fprintf(stderr, "Failed to allocate output codec context\n");
            //goto cleanup;
            exit(0);
        }

        const int                *formats;
        const int                *sample_rates;
        const AVChannelLayout    *ch_layouts;

        check_error( avcodec_get_supported_config(
                pOutCodecContext, NULL,
                AV_CODEC_CONFIG_SAMPLE_FORMAT, 0,
                (const void **) &formats, NULL),
                "avcodec_get_supported_config");
        check_error( avcodec_get_supported_config(
                pOutCodecContext, NULL,
                AV_CODEC_CONFIG_SAMPLE_RATE, 0,
                (const void **) &sample_rates, NULL),
                "avcodec_get_supported_config");
        check_error( avcodec_get_supported_config(
                pOutCodecContext, NULL,
                AV_CODEC_CONFIG_CHANNEL_LAYOUT, 0,
                (const void **) &ch_layouts, NULL),
                "avcodec_get_supported_config");
        int nformats = 0;
        int nrates = 0;
        int nlayouts = 0;
        printf("formats:\n");
        for (; nformats < 18 && formats[nformats] != AV_SAMPLE_FMT_NONE; nformats++) {
            auto format = (AVSampleFormat)formats[nformats];
            printf("%d %s\n", format, av_get_sample_fmt_name(format));
        }
        if ( sample_rates ) {
            printf("sample rates:\n");
            for (; nrates < 18 && sample_rates[nrates]; nrates++) {
                printf("%d\n", sample_rates[nrates]);
            }
        }
        if ( ch_layouts ) {
            for (; nlayouts < 18 && ch_layouts[nlayouts].nb_channels != 0; nlayouts++) {
                auto & layout = ch_layouts[nlayouts];
                printf("%d %d %ld\n", layout.order, layout.nb_channels, layout.u.mask);
            }
        }

        int planar = av_sample_fmt_is_planar( other->sample_fmt );
        auto in_sample_fmt = other->sample_fmt;
        auto alt_sample_fmt = av_get_alt_sample_fmt( in_sample_fmt, 1 - planar );
        auto sample_fmt = (AVSampleFormat)formats[0];
        for (int i = 0; i < nformats; i++) {
            if (formats[i] == in_sample_fmt) {
                sample_fmt = in_sample_fmt;
                break;
            } else if (formats[i] == alt_sample_fmt) {
                sample_fmt = alt_sample_fmt;
            }
        }

        pOutCodecContext->sample_fmt = sample_fmt;
        pOutCodecContext->sample_rate = other->sample_rate;
        if ( ch_layouts ) {
            int layout_found = 0;
            for (int i = 0; i < nlayouts; i++) {
                if (ch_layouts[i].nb_channels == other->ch_layout.nb_channels) {
                    layout_found = i;
                    break;
                } else if (ch_layouts[i].nb_channels < other->ch_layout.nb_channels) {
                    layout_found = i;
                }
            }
            av_channel_layout_copy(
                &pOutCodecContext->ch_layout,
                &ch_layouts[layout_found]
            );
        } else {
            // I guess that means it supports any channel layout?
            av_channel_layout_copy( &pOutCodecContext->ch_layout, &other->ch_layout );
        }
        pOutCodecContext->time_base = (AVRational){1, other->sample_rate};
        
        // Some formats require a global header
        if (format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
            pOutCodecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        check_error(avcodec_open2(pOutCodecContext, pOutCodec, NULL), "Could not open output codec");

        // Create a new audio stream in the output file
        pOutStream = avformat_new_stream(format_ctx, pOutCodec);
        if (!pOutStream) {
            fprintf(stderr, "Failed to create output stream\n");
            //goto cleanup;
            exit(0);
        }
        check_error(avcodec_parameters_from_context(pOutStream->codecpar, pOutCodecContext), "Failed to copy output codec params");
        pOutStream->time_base = pOutCodecContext->time_base;

        // Open the output file for writing
        if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
            check_error(avio_open(&format_ctx->pb, filename, AVIO_FLAG_WRITE), "Could not open output file");
        }
        check_error(avformat_write_header(format_ctx, NULL), "Error writing header");

        enc_frame = av_frame_alloc();
        enc_frame->nb_samples     = pOutCodecContext->frame_size;
        enc_frame->ch_layout      = pOutCodecContext->ch_layout;
        enc_frame->format         = pOutCodecContext->sample_fmt;
        enc_frame->sample_rate    = pOutCodecContext->sample_rate;
        check_error( av_frame_get_buffer( enc_frame, 0 ),
            "Error making frame buffer" );

        swr_ctx = swr_alloc();
        assert( swr_ctx );

        // needs to be specified
        if ( other->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC ) {
            av_channel_layout_default(&other->ch_layout, other->ch_layout.nb_channels);
        }

        // Set input parameters (from the decoder)
        check_error( av_opt_set_chlayout(swr_ctx, "in_chlayout", &other->ch_layout, 0), "chlayout" );
        check_error( av_opt_set_int(swr_ctx, "in_sample_rate",    other->sample_rate,    0), "sample_rate" );
        check_error( av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", other->sample_fmt, 0), "sample_fmt" );

        // Set output parameters (for the encoder)
        check_error( av_opt_set_chlayout(swr_ctx, "out_chlayout", &pOutCodecContext->ch_layout, 0), "chlayout" );
        check_error( av_opt_set_int(swr_ctx, "out_sample_rate",    pOutCodecContext->sample_rate,    0), "sample_rate" );
        check_error( av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", pOutCodecContext->sample_fmt, 0), "sample_fmt" );

        // Initialize the context
        check_error( swr_init(swr_ctx), "swr_init failed" );

        pts_counter = 0;
    }

    void frame( AVFrame * frame ) {
        check_error( swr_convert_frame(swr_ctx, NULL, frame),
            "Error resampling" );
        while ( swr_get_delay(swr_ctx, enc_frame->sample_rate) > enc_frame->nb_samples ) {
            check_error( swr_convert_frame(swr_ctx, enc_frame, NULL),
                "Error resampling" );

            enc_frame->pts = av_rescale_q(
                pts_counter,
                (AVRational){1, pOutCodecContext->sample_rate},
                pOutCodecContext->time_base);
            pts_counter += enc_frame->nb_samples;

            check_error(
                avcodec_send_frame( pOutCodecContext, enc_frame ),
                "Error sending frame to encoder"
            );

            AVPacket *pOutPacket = av_packet_alloc();
            while (avcodec_receive_packet(pOutCodecContext, pOutPacket) == 0) {
                // write data
                av_packet_rescale_ts(pOutPacket, pOutCodecContext->time_base, pOutStream->time_base);
                pOutPacket->stream_index = pOutStream->index;
                check_error(av_interleaved_write_frame(format_ctx, pOutPacket), "Error writing packet to output");
                av_packet_unref(pOutPacket);
            }
            av_packet_free(&pOutPacket);
        }
    }

    void flush() {
        check_error(avcodec_send_frame(pOutCodecContext, NULL), "Error flushing encoder");
        AVPacket *pOutPacket = av_packet_alloc();
        while (avcodec_receive_packet(pOutCodecContext, pOutPacket) == 0) {
            av_packet_rescale_ts(pOutPacket, pOutCodecContext->time_base, pOutStream->time_base);
            pOutPacket->stream_index = pOutStream->index;
            check_error(av_interleaved_write_frame(format_ctx, pOutPacket), "Error writing flush packet to output");
            av_packet_unref(pOutPacket);
        }
        av_packet_free(&pOutPacket);

        check_error(av_write_trailer(format_ctx), "Error writing trailer");
    }
};

void load_voice_prefix(
    std::deque<int> & text_prefixes,
    std::deque<std::vector<int>> & audio_prefixes,
    Decoder * decoder,
    mimi_codec_t * codec,
    moshi_lm_t * lm,
    int n_q
) {
    const int lm_ungenerated_token_id = -2;
    const int frame_size = mimi_frame_size( codec );
    const int max_delay = moshi_lm_get_max_delay( lm );
    const int delay_steps = moshi_lm_get_delay_steps( lm );

    unref_ptr<mimi_encode_context_t> encoder = mimi_encode_alloc_context( codec );

    AVChannelLayout mono;
    av_channel_layout_default( &mono, 1 );
    Resampler resampler;
    resampler.set_input( decoder->codec_ctx );
    resampler.set_output( 24000, AV_SAMPLE_FMT_FLT, mono, frame_size );
    resampler.init();

    // prepend delays
    for ( int i = 0; i < max_delay + delay_steps; i++ ) {
        audio_prefixes.push_back({});
        auto & codes = audio_prefixes.back();
        codes.resize( n_q );
        for ( int j = 0; j < n_q; j++ ) {
            codes[j] = lm_ungenerated_token_id;
        }
    }

    std::vector<int16_t> tokens(n_q);

    // main loop
    AVFrame * dec_frame;
    while ( ( dec_frame = decoder->frame() ) ) {
        auto frame = resampler.frame( dec_frame );
        while ( frame ) {
            mimi_encode_send( encoder, (float*)frame->data[0] );
            mimi_encode_receive( encoder, tokens.data() );

            text_prefixes.push_back( -1 ); // zero_id
            audio_prefixes.push_back({});
            std::vector<int> & audio_codes = audio_prefixes.back();
            audio_codes.resize( n_q );
            // 16bit to 32bit
            for ( int i = 0; i < n_q; ++i ) {
                audio_codes[i] = tokens[i];
            }
            // HACK: moving known delayed code from config
            auto nprefixes = audio_prefixes.size();
            audio_prefixes[nprefixes-3][0] = audio_codes[0];
            audio_codes[0] = lm_ungenerated_token_id;

            frame = resampler.frame();
        }
    }
    auto frame = resampler.flush( true ); // inject silence
    if ( frame ) {
        mimi_encode_send( encoder, (float*)frame->data[0] );
        mimi_encode_receive( encoder, tokens.data() );

        text_prefixes.push_back( -1 ); // zero_id
        audio_prefixes.push_back({});
        std::vector<int> & audio_codes = audio_prefixes.back();
        audio_codes.resize( n_q );
        // 16bit to 32bit
        for ( int i = 0; i < n_q; ++i ) {
            audio_codes[i] = tokens[i];
        }
        // HACK: moving known delayed code from config
        auto nprefixes = audio_prefixes.size();
        audio_prefixes[nprefixes-3][0] = audio_codes[0];
        audio_codes[0] = lm_ungenerated_token_id;
    }
}

