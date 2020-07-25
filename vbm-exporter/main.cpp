#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <iomanip>
#include <cstdint>
#include <vector>
#include <cstring>
#include <vbm_format.h>
#include <tga.h>

const char *vbm_format_str(uint32_t format)
{
    if (format == VBM_CF_1555) return "1555";
    else if (format == VBM_CF_4444) return "4444";
    else if (format == VBM_CF_565) return "565";
    else return nullptr;
}

void print_vbm_metadata(const vbm_header_t &hdr)
{
    std::cout << "VBM Metadata:\n"
        << "Version: " << hdr.version << "\n"
        << "Size: " << hdr.width << "x" << hdr.height << "\n"
        << "Format: " << vbm_format_str(hdr.format) << "\n"
        << "FPS: " << hdr.fps << "\n"
        << "Number of frames: " << hdr.num_frames << "\n"
        << "Number of mipmaps: " << hdr.num_mipmaps << "\n\n";
}

void write_tga_frame(const std::string &filename, const char *pixel_data, int w, int h, const vbm_header_t &hdr)
{
    std::ofstream output_tga_stream(filename, std::ofstream::out | std::ofstream::binary);
    output_tga_stream.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    tga_header tga_hdr;
    init_tga_header(tga_hdr, w, h, hdr.format == VBM_CF_565 ? 24 : 32);
    output_tga_stream.write(reinterpret_cast<char*>(&tga_hdr), sizeof(tga_hdr));

    const uint16_t *input_pixels = reinterpret_cast<const uint16_t*>(pixel_data);
    for (int i = 0; i < w * h; ++i)
    {
        uint16_t input_pixel = input_pixels[i];
        if (hdr.format == VBM_CF_1555)
        {
            char output_pixel[] = {
                static_cast<char>(static_cast<int>((input_pixel >> 0)  & 0x1F) * 255 / 0x1F), // B
                static_cast<char>(static_cast<int>((input_pixel >> 5)  & 0x1F) * 255 / 0x1F), // G
                static_cast<char>(static_cast<int>((input_pixel >> 10) & 0x1F) * 255 / 0x1F), // R
                static_cast<char>(static_cast<int>(!(input_pixel >> 15)) * 255), // A
            };
            output_tga_stream.write(output_pixel, sizeof(output_pixel));
        }
        else if (hdr.format == VBM_CF_4444)
        {
            char output_pixel[] = {
                static_cast<char>(static_cast<int>((input_pixel >> 0)  & 0xF) * 255 / 0xF), // B
                static_cast<char>(static_cast<int>((input_pixel >> 4)  & 0xF) * 255 / 0xF), // G
                static_cast<char>(static_cast<int>((input_pixel >> 8)  & 0xF) * 255 / 0xF), // R
                static_cast<char>(static_cast<int>((input_pixel >> 12) & 0xF) * 255 / 0xF), // A
            };
            output_tga_stream.write(output_pixel, sizeof(output_pixel));
        }
        else if (hdr.format == VBM_CF_565)
        {
            char output_pixel[] = {
                static_cast<char>(static_cast<int>((input_pixel >> 0)  & 0xF) * 255 / 0x1F), // B
                static_cast<char>(static_cast<int>((input_pixel >> 5)  & 0xF) * 255 / 0x3F), // G
                static_cast<char>(static_cast<int>((input_pixel >> 11) & 0xF) * 255 / 0x1F), // R
            };
            output_tga_stream.write(output_pixel, sizeof(output_pixel));
        }
    }
}

std::string get_basename_without_ext(const std::string &path)
{
    size_t dot_pos = path.rfind('.');
    size_t dir_sep_pos = path.find_last_of("/\\");
    size_t basename_pos = dir_sep_pos == std::string::npos ? 0 : dir_sep_pos + 1;
    size_t basename_len = dot_pos == std::string::npos ? std::string::npos : dot_pos - basename_pos;
    return path.substr(basename_pos, basename_len);
}

int export_vbm(const char *vbm_filename, const std::string &output_prefix, bool verbose)
{
    std::ifstream vbm_stream(vbm_filename, std::ifstream::in | std::ifstream::binary);
    vbm_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    vbm_header_t hdr;
    vbm_stream.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));

    if (hdr.signature != VBM_SIGNATURE)
    {
        std::cerr << "Invalid VBM signature!\n";
        return -1;
    }

    if (verbose)
    {
        std::cout << "Processing " << vbm_filename << "...\n";
        print_vbm_metadata(hdr);
    }

    std::string prefix = output_prefix + get_basename_without_ext(vbm_filename);

    auto pixel_data = std::make_unique<char[]>(hdr.width * hdr.height * 2); // 16 bit
    
    for (unsigned frame_idx = 0; frame_idx < hdr.num_frames; ++frame_idx)
    {
        int w = hdr.width;
        int h = hdr.height;
        
        for (unsigned mipmap_idx = 0; mipmap_idx < hdr.num_mipmaps + 1; ++mipmap_idx)
        {
            vbm_stream.read(pixel_data.get(), w * h * 2);

            std::ostringstream ss;
            ss << prefix << "-";
            if (mipmap_idx > 0)
                ss << static_cast<char>('a' + mipmap_idx - 1);
            ss << std::setfill('0') << std::setw(4) << frame_idx << ".tga";

            write_tga_frame(ss.str(), pixel_data.get(), w, h, hdr);

            w = std::max(w / 2, 1);
            h = std::max(h / 2, 1);
        }
    }

    std::cout << "makevbm " << vbm_format_str(hdr.format) << " " << hdr.fps << " " << prefix << ".tga\n";
    return 0;
}

int main(int argc, char *argv[])
{
    std::string output_prefix;
    bool help = true;
    bool verbose = false;
    std::vector<const char*> input_files;

    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "-O") && i + 1 < argc)
        {
            output_prefix = argv[++i];
            if (!output_prefix.empty() && output_prefix.back() != '/' && output_prefix.back() != '\\')
                output_prefix += '/';
        }
        else if (!std::strcmp(argv[i], "-h"))
            help = true;
        else if (!std::strcmp(argv[i], "-v"))
            verbose = true;
        else
        {
            input_files.push_back(argv[i]);
            help = false;
        }
    }

    if (help)
    {
        std::cerr << "Usage: " << argv[0] << " [-O output_dir] [-v] vbm_files...\n";
        return -1;
    }

    for (auto input_file : input_files)
    {
        if (export_vbm(input_file, output_prefix, verbose) != 0)
            return -1;
    }

    return 0;
}
