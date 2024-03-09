use binrw::BinRead;

#[allow(unused)]
#[derive(BinRead)]
#[br(magic = b"VFNT")]
pub struct VfHeader {
    #[br(assert(version <= 1, "Unsupported VF version!"))]
    pub version: u32,          // font version (0 or 1)
    #[br(if(version >= 1, VfFormat::Mono4))]
    pub format: VfFormat,      // exists if version >= 1, else font has VF_FMT_MONO_4 format, for values
                               // description see vf_format_t
    pub num_chars: u32,        // length of chars array
    pub first_ascii: u32,      // ascii code of first character supported by this font, usually equals
                               // 0x20 (space character)
    pub default_spacing: u32,  // spacing used for characters missing in the font (lower than first_ascii and
                               // greater than first_ascii + num_chars)
    pub height: u32,           // font height (height is the same for all characters)
    pub num_kern_pairs: u32,   // length of kern_pairs array
    #[br(if(version == 0))]
    kern_data_size: u32,       // exists if version == 0, unused by RF (can be calculated from num_kern_pairs)
    #[br(if(version == 0))]
    char_data_size: u32,       // exists if version == 0, unused by RF (can be calculated from num_chars)
    pub pixel_data_size: u32,  // size of pixel array
}

#[allow(unused)]
#[derive(BinRead)]
pub struct VfKernPair  {
    char_before_idx: u8, // index of character before spacing
    char_after_idx: u8,  // index of character after spacing
    offset: i8,          // value added to vf_char_desc_t::spacing
}

#[allow(unused)]
#[derive(BinRead)]
pub struct VfCharDesc {
    pub spacing: u32,              // base spacing for this character (can be modified by kerning data), spacing is
                                   // similar to width but is used only during text rendering to update X coordinate
    pub width: u32,                // character width in pixels, not to be confused with spacing
    pub pixel_data_offset: u32,    // offset in vf_file::pixels, all pixels for one characters are stored in one run,
                                   // total number of pixels for a character equals:
                                   // vf_char_desc_t::width * vf_header_t::height
    pub first_kerning_entry: u16,  // index in vf_file::kern_pairs array, entries can be checked from there because
                                   // array is sorted by character indices
    pub user_data: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, BinRead)]
pub enum VfFormat {
    #[br(magic = 0xFu32)] Mono4,           // 1 byte per pixel, monochromatic, only values in range 0-14 are used
	#[br(magic = 0xF0F0F0Fu32)] Rgba4444,  // 2 byte per pixel, RGBA 4444 (4 bits per channel)
	#[br(magic = 0xFFFFFFF0u32)] Indexed,  // 1 byte per pixel, indexed (palette is at the end of the file)
}

#[derive(BinRead)]
pub struct VfPalette {
    pub entries: [[u8; 4]; 256],
}
