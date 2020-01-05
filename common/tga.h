#pragma once

#include <cstdint>

#pragma pack(push, 1)
struct tga_header {
   int8_t   idlength;
   int8_t   colourmaptype;
   int8_t   datatypecode;
   uint16_t colourmaporigin;
   uint16_t colourmaplength;
   int8_t   colourmapdepth;
   uint16_t x_origin;
   uint16_t y_origin;
   int16_t  width;
   int16_t  height;
   int8_t   bitsperpixel;
   int8_t   imagedescriptor;
};
#pragma pack(pop)
static_assert(sizeof(tga_header) == 18, "invalid tga_header size");

inline void init_tga_header(tga_header &tga_hdr, int16_t w, int16_t h, int8_t bits_per_pixel)
{
    tga_hdr.idlength = 0;
    tga_hdr.colourmaptype = 0;
    tga_hdr.datatypecode = 2;
    tga_hdr.colourmaporigin = 0;
    tga_hdr.colourmaplength = 0;
    tga_hdr.colourmapdepth = 0;
    tga_hdr.x_origin = 0;
    tga_hdr.y_origin = 0;
    tga_hdr.width = w;
    tga_hdr.height = h;
    tga_hdr.bitsperpixel = bits_per_pixel;
    tga_hdr.imagedescriptor = 1 << 5; // Origin in upper left-hand corner
}
