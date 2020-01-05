#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <vpp_format.h>
#include <cstdint>

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: " << argv[0] << " vpp_file [output_dir]\n";
        return -1;
    }

    const char *vpp_filename = argv[1];
    const char *output_dir = argc == 3 ? argv[2] : ".";

    std::ifstream vpp_stream(vpp_filename, std::ifstream::in | std::ifstream::binary);
    vpp_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    vpp_header_t hdr;
    vpp_stream.read((char*)&hdr, sizeof(hdr));

    if (hdr.signature != VPP_SIGNATURE)
    {
        std::cerr << "Invalid VPP signature!\n";
        return -1;
    }

    vpp_stream.seekg(VPP_ALIGNMENT, std::ios_base::beg);

    std::vector<vpp_entry_t> entries;
    for (unsigned i = 0; i < hdr.file_count; ++i)
    {
        vpp_entry_t entry;
        vpp_stream.read((char*)&entry, sizeof(entry));
        entries.push_back(entry);
    }

    std::string prefix = output_dir;
    prefix += '/';
    for (unsigned i = 0; i < hdr.file_count; ++i)
    {
        std::cout << entries[i].file_name << "\n";
        vpp_stream.seekg(((size_t)vpp_stream.tellg() + VPP_ALIGNMENT - 1) & ~(VPP_ALIGNMENT - 1));
        auto file_data = std::make_unique<char[]>(entries[i].file_size);
        vpp_stream.read(file_data.get(), entries[i].file_size);

        std::ofstream output_stream(prefix + entries[i].file_name, std::ofstream::out | std::ofstream::binary);
        output_stream.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        output_stream.write(file_data.get(), entries[i].file_size);
    }
    
    std::cout << "Exported.\n";

    return 0;
}
