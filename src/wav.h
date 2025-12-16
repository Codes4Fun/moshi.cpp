#pragma once

struct WaveHeader {
    uint32_t riff;
    uint32_t size; // Size of the rest of the file in bytes.
    uint32_t wave;
    uint32_t fmt_tag;
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    uint32_t data_tag;
    uint32_t data_size;
};

void save_wav(
    const std::string filename,
    const std::vector<short> &data,
    int sample_rate
) {
    int dataSize = (int) data.size() * 2;
    WaveHeader header;
    header.riff = 0x46464952; // "RIFF"
    header.size = dataSize + sizeof(WaveHeader) - 8;
    header.wave = 0x45564157; // "WAVE"
    header.fmt_tag = 0x20746d66; // "fmt "
    header.fmt_size = 16;
    header.audio_format = 1; // PCM
    header.num_channels = 1;
    header.sample_rate = sample_rate;
    header.byte_rate = sample_rate * 16 / 8; // bytes per second
    header.block_align = 16 / 8; // bytes per sample
    header.bits_per_sample = 16;
    header.data_tag = 0x61746164; // "data"
    header.data_size = dataSize;
    FILE *f = fopen(filename.c_str(), "wb");
    if (f == nullptr)
        throw std::runtime_error("failed to open file for writing");
    fwrite(&header, sizeof(WaveHeader), 1, f);
    size_t offset = ftell(f);
    printf("header: %zu / %zu\n", offset, sizeof(WaveHeader));
    fwrite(data.data(), dataSize, 1, f);
    offset = ftell(f);
    //printf("data: %ld / %ld\n", offset, sizeof(WaveHeader) + dataSize);
    printf("length: %f seconds\n", data.size() / (float)sample_rate);
    fclose(f);
}

int load_wav(
    const std::string filename,
    std::vector<short> & samples
) {
    auto f = fopen( filename.c_str(), "rb" );
    if (f == nullptr)
        throw std::runtime_error("failed to open file for reading");
    WaveHeader header;
    if ( fread( &header, sizeof(header), 1, f ) != 1 )
        throw std::runtime_error("failed to read header from wave file");
    if ( header.riff != 0x46464952 ) // "RIFF"
        throw std::runtime_error("'RIFF' not found");
    //header.size
    if ( header.wave != 0x45564157 ) // "WAVE"
        throw std::runtime_error("'WAVE' not found");
    if ( header.fmt_tag != 0x20746d66 ) // "fmt "
        throw std::runtime_error("'fmt ' not found");
    if ( header.fmt_size != 16
      || header.audio_format != 1
      || header.num_channels != 1
      || header.block_align != 2
      || header.bits_per_sample != 16 )
        throw std::runtime_error("unsupported format, only pcm mono 16bit supported");
    while ( header.data_tag != 0x61746164 ) {
        if ( fseek( f, header.data_size, SEEK_CUR ) != 0 )
            throw std::runtime_error("failed to seek to data");
        if ( fread( &header.data_tag, 8, 1, f ) != 1 )
            throw std::runtime_error("failed to read data from wave file");
    }
    uint32_t nsamples = header.data_size / 2;
    samples.resize( nsamples );
    if ( fread( samples.data(), nsamples * 2, 1, f ) != 1 )
        throw std::runtime_error("failed to read data from wave file");
    return header.sample_rate;
}

