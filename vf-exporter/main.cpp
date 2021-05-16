#include <fstream>
#include <iostream>
#include <cstddef>
#include <memory>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <vector>
#include <string_view>
#include <vf_format.h>
#include <tga.h>

void print_vf_metadata(const vf_header_t &hdr)
{
    std::cout << "VF Metadata:\n"
        << "Version: " << hdr.version << "\n"
        << "Format: " << std::hex << hdr.format << std::dec << "\n"
        << "First ASCII: " << hdr.first_ascii << "\n"
        << "Character size: " << hdr.default_spacing << "x" << hdr.height << "\n"
        << "Number of chars: " << hdr.num_chars << "\n"
        << "Number of kerning pairs: " << hdr.num_kern_pairs << "\n\n";
}

std::string get_basename_without_ext(std::string_view path)
{
    size_t dot_pos = path.rfind('.');
    size_t dir_sep_pos = path.find_last_of("/\\");
    size_t basename_pos = dir_sep_pos == std::string::npos ? 0 : dir_sep_pos + 1;
    size_t basename_len = dot_pos == std::string::npos ? std::string::npos : dot_pos - basename_pos;
    return std::string{path.substr(basename_pos, basename_len)};
}

void read_vf_header(vf_header_t &hdr, std::ifstream &vf_stream)
{
    memset(&hdr, 0, sizeof(hdr));
    vf_stream.read(reinterpret_cast<char*>(&hdr.signature), sizeof(hdr.signature));
    vf_stream.read(reinterpret_cast<char*>(&hdr.version), sizeof(hdr.version));
    if (hdr.version >= 1)
        vf_stream.read(reinterpret_cast<char*>(&hdr.format), sizeof(hdr.format));
    else
        hdr.format = VF_FMT_MONO_4;
    vf_stream.read(reinterpret_cast<char*>(&hdr.num_chars), sizeof(hdr.num_chars));
    vf_stream.read(reinterpret_cast<char*>(&hdr.first_ascii), sizeof(hdr.first_ascii));
    vf_stream.read(reinterpret_cast<char*>(&hdr.default_spacing), sizeof(hdr.default_spacing));
    vf_stream.read(reinterpret_cast<char*>(&hdr.height), sizeof(hdr.height));
    vf_stream.read(reinterpret_cast<char*>(&hdr.num_kern_pairs), sizeof(hdr.num_kern_pairs));
    if (hdr.version == 0)
    {
        vf_stream.read(reinterpret_cast<char*>(&hdr.kern_data_size), sizeof(hdr.kern_data_size));
        vf_stream.read(reinterpret_cast<char*>(&hdr.char_data_size), sizeof(hdr.char_data_size));
    }
    vf_stream.read(reinterpret_cast<char*>(&hdr.pixel_data_size), sizeof(hdr.pixel_data_size));
}

void determine_output_image_size(size_t num_pixels, size_t &w, size_t &h)
{
    if (num_pixels < 0x400)
        w = h = 64;
    else if (num_pixels < 0x1000)
        w = h = 128;
    else if (num_pixels < 0x4000)
        w = h = 256;
    else
        w = h = 512;
}

void write_font_tga(const std::string &filename, const vf_header_t &hdr, const vf_char_desc_t char_desc[],
    const std::byte pixel_data[], const uint32_t palette[])
{
    std::ofstream output_tga_stream(filename, std::ofstream::out | std::ofstream::binary);
    output_tga_stream.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    size_t bytes_per_pixel = hdr.format == VF_FMT_RGBA_4444 ? 2 : 1;
    size_t num_pixels = hdr.pixel_data_size / bytes_per_pixel;
    size_t w;
    size_t h;
    determine_output_image_size(num_pixels, w, h);
    
    tga_header tga_hdr;
    init_tga_header(tga_hdr, w, h, 32);
    output_tga_stream.write(reinterpret_cast<char*>(&tga_hdr), sizeof(tga_hdr));

    auto bitmap_data = std::make_unique<uint32_t[]>(w * h); // RGBA
    std::memset(bitmap_data.get(), 0, w * h * sizeof(uint32_t));

    int dst_x = 0;
    int dst_y = 1;
    for (unsigned char_idx = 0; char_idx < hdr.num_chars; ++char_idx)
    {
        const std::byte *in_pixel_data = pixel_data + char_desc[char_idx].pixel_data_offset;
        if (dst_x + char_desc[char_idx].width + 3 > w)
        {
            dst_x = 0;
            dst_y += hdr.height + 3;
        }
        if (dst_y + hdr.height + 3 > h)
            throw std::runtime_error("Font is too big");
        
        // Top and buttom border
        uint32_t border_clr = 0;//0xFF00FF00;
        for (unsigned off_x = 0; off_x < char_desc[char_idx].width + 2; ++off_x)
        {
            bitmap_data[dst_y * w + dst_x + off_x] = border_clr;
            bitmap_data[(dst_y + 1 + hdr.height) * w + dst_x + off_x] = border_clr;
        }
        // Left and right border
        for (unsigned off_y = 0; off_y < hdr.height + 2; ++off_y)
        {
            bitmap_data[(dst_y + off_y) * w + dst_x] = border_clr;
            bitmap_data[(dst_y + off_y) * w + dst_x + 1 + char_desc[char_idx].width] = border_clr;
        }

        // Copy character pixels
        for (unsigned off_y = 1; off_y < hdr.height + 1; ++off_y)
        {
            for (unsigned off_x = 1; off_x < char_desc[char_idx].width + 1; ++off_x)
            {
                uint32_t &out_pixel = bitmap_data[(dst_y + off_y) * w + dst_x + off_x];
                auto *out_pixel_bytes = reinterpret_cast<uint8_t*>(&out_pixel);
                if (hdr.format == VF_FMT_MONO_4)
                {
                    int in_pixel_value = *reinterpret_cast<const uint8_t*>(in_pixel_data);
                    out_pixel_bytes[0] = out_pixel_bytes[1] = out_pixel_bytes[2] = out_pixel_bytes[3] = 
                        static_cast<uint8_t>(std::min(in_pixel_value, 14) * 255 / 14);
                }
                else if (hdr.format == VF_FMT_RGBA_4444)
                {
                    uint16_t in_pixel_value = *reinterpret_cast<const uint16_t*>(in_pixel_data);
                    int r = (in_pixel_value >> 0) & 0xF;
                    int g = (in_pixel_value >> 4) & 0xF;
                    int b = (in_pixel_value >> 8) & 0xF;
                    int a = (in_pixel_value >> 12) & 0xF;
                    out_pixel_bytes[0] = static_cast<uint8_t>(r * 0xFF / 0xF);
                    out_pixel_bytes[1] = static_cast<uint8_t>(g * 0xFF / 0xF);
                    out_pixel_bytes[2] = static_cast<uint8_t>(b * 0xFF / 0xF);
                    out_pixel_bytes[3] = static_cast<uint8_t>(a * 0xFF / 0xF);
                }
                else if (hdr.format == VF_FMT_INDEXED)
                {
                    int index = *reinterpret_cast<const uint8_t*>(in_pixel_data);
                    out_pixel = palette[index];
                }
                else
                    assert(false);
                in_pixel_data += bytes_per_pixel;
            }
        }
        dst_x += char_desc[char_idx].width + 3;
    }
    output_tga_stream.write(reinterpret_cast<char*>(bitmap_data.get()), w * h * sizeof(uint32_t));
}

void export_font(const char *vf_filename, const std::string &output_prefix)
{
    std::ifstream vf_stream(vf_filename, std::ifstream::in | std::ifstream::binary);
    vf_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    std::cout << "Processing " << vf_filename << "...\n";
    vf_header_t hdr;
    read_vf_header(hdr, vf_stream);
    if (hdr.signature != VF_SIGNATURE)
        throw std::runtime_error("Invalid VF signature!");
    if (hdr.version > 1)
        throw std::runtime_error("Unsupported VF version!");

    print_vf_metadata(hdr);

    auto kern_pairs = std::make_unique<vf_kern_pair_t[]>(hdr.num_kern_pairs);
    vf_stream.read(reinterpret_cast<char*>(kern_pairs.get()), hdr.num_kern_pairs * sizeof(vf_kern_pair_t));

    auto char_desc = std::make_unique<vf_char_desc_t[]>(hdr.num_chars);
    vf_stream.read(reinterpret_cast<char*>(char_desc.get()), hdr.num_chars * sizeof(vf_char_desc_t));

    auto pixel_data = std::make_unique<std::byte[]>(hdr.pixel_data_size);
    vf_stream.read(reinterpret_cast<char*>(pixel_data.get()), hdr.pixel_data_size);

    auto palette = std::make_unique<uint32_t[]>(256);
    if (hdr.format == VF_FMT_INDEXED)
        vf_stream.read(reinterpret_cast<char*>(palette.get()), 256 * 4);
    
    std::string output_filename = output_prefix + get_basename_without_ext(vf_filename) + ".tga";
    write_font_tga(output_filename, hdr, char_desc.get(), pixel_data.get(), palette.get());
}

int main(int argc, char *argv[]) try
{
    std::string output_prefix;
    bool help = true;
    std::vector<const char*> input_files;

    for (int i = 1; i < argc; ++i)
    {
        std::string_view arg{argv[i]};
        if (arg == "-O" && i + 1 < argc)
        {
            output_prefix = argv[++i];
            if (!output_prefix.empty() && output_prefix.back() != '/' && output_prefix.back() != '\\')
                output_prefix += '/';
        }
        else if (arg == "-h")
            help = true;
        else
        {
            input_files.emplace_back(argv[i]);
            help = false;
        }
    }

    if (help)
    {
        std::cerr << "Usage: " << argv[0] << " [-O output_dir] vf_files...\n";
        return -1;
    }

    for (const char* input_file : input_files)
        export_font(input_file, output_prefix);

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return -1;
}