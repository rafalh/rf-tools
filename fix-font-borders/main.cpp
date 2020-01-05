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
#include <tga.h>

std::string get_basename_without_ext(std::string_view path)
{
    size_t dot_pos = path.rfind('.');
    size_t dir_sep_pos = path.find_last_of("/\\");
    size_t basename_pos = dir_sep_pos == std::string::npos ? 0 : dir_sep_pos + 1;
    size_t basename_len = dot_pos == std::string::npos ? std::string::npos : dot_pos - basename_pos;
    return std::string{path.substr(basename_pos, basename_len)};
}

struct Size2d
{
    int w;
    int h;
};

struct Coords2d
{
    int x;
    int y;
};

bool is_border_pixel(const uint32_t pixels[], Size2d size, Coords2d pos)
{
    uint32_t pixel = pixels[pos.y * size.w + pos.x];
    uint32_t border_clr = 0xFF00FF00;
    return pixel == border_clr;
}

void fix_borders_in_font_bitmap(uint32_t pixels[], Size2d size)
{
    for (int y = 0; y < size.h; ++y)
    {
        bool in_border = false;
        int num_border_pixels = 0;
        for (int x = 0; x < size.w; ++x)
        {
            bool is_redundant_border = false;
            bool border_center = is_border_pixel(pixels, size, {x, y});
            bool border_right = x + 1 < size.w && is_border_pixel(pixels, size, {x + 1, y});
            bool border_left = x > 0 && is_border_pixel(pixels, size, {x - 1, y});
            bool border_down = y + 1 < size.h && is_border_pixel(pixels, size, {x, y + 1});
            bool border_up = y > 0 && is_border_pixel(pixels, size, {x, y - 1});

            if (!in_border)
            {
                // additional border on the top
                is_redundant_border |= border_center && border_down;
                // additional border on the left
                is_redundant_border |= border_center && border_right;
            }
            else
            {
                // additional border on the right
                is_redundant_border |= border_left && border_center;
                // additional border on the bottom
                is_redundant_border |= border_up && border_center;
            }

            if (border_center)
                ++num_border_pixels;
            else
                num_border_pixels = 0;
            
            if (border_center && !border_right && num_border_pixels <= 2)
                in_border = !in_border;

            if (is_redundant_border)
                pixels[y * size.w + x] = 0;
        }
    }
}

bool process_file(const char *input_file, const std::string &output_prefix)
{
    std::ifstream input_stream{input_file, std::ifstream::in | std::ifstream::binary};
    input_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    tga_header tga_hdr;
    input_stream.read(reinterpret_cast<char*>(&tga_hdr), sizeof(tga_hdr));

    if (tga_hdr.bitsperpixel != 32)
    {
        std::cerr << "Unsupported TGA format";
        return false;
    }

    auto pixels = std::make_unique<uint32_t[]>(tga_hdr.width * tga_hdr.height);
    input_stream.read(reinterpret_cast<char*>(pixels.get()), tga_hdr.width * tga_hdr.height * sizeof(pixels[0]));

    fix_borders_in_font_bitmap(pixels.get(), {tga_hdr.width, tga_hdr.height});

    std::string output_filename = output_prefix + get_basename_without_ext(input_file) + ".tga";
    std::ofstream output_stream{output_filename, std::ofstream::out | std::ofstream::binary};
    output_stream.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    output_stream.write(reinterpret_cast<char*>(&tga_hdr), sizeof(tga_hdr));
    output_stream.write(reinterpret_cast<char*>(pixels.get()), tga_hdr.width * tga_hdr.height * sizeof(pixels[0]));
    return true;
}

int main(int argc, char *argv[])
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
            input_files.push_back(argv[i]);
            help = false;
        }
    }

    if (help)
    {
        std::cerr << "Usage: " << argv[0] << " [-O output_dir] tga_files...\n";
        return -1;
    }

    for (auto input_file : input_files)
    {
        if (!process_file(input_file, output_prefix))
            return -1;
    }

    return 0;
}
