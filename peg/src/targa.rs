use std::io::{Write, Result};
use byteorder::{WriteBytesExt, LittleEndian};

pub struct TgaFileHeader {
    idlength: i8,
    colourmaptype: i8,
    datatypecode: i8,
    colourmaporigin: u16,
    colourmaplength: u16,
    colourmapdepth: i8,
    x_origin: u16,
    y_origin: u16,
    width: i16,
    height: i16,
    bitsperpixel: i8,
    imagedescriptor: i8,
}

impl TgaFileHeader {
    pub fn new(w: i16, h: i16, bpp: i8, indexed: bool) -> Self {
        Self {
            idlength: 0,
            colourmaptype: if indexed { 1 } else { 0 },
            datatypecode: if indexed { 1 } else { 2 },
            colourmaporigin: 0,
            colourmaplength: if indexed { 256 } else { 0 },
            colourmapdepth: if indexed { bpp } else { 0 },
            x_origin: 0,
            y_origin: 0,
            width: w,
            height: h,
            bitsperpixel: if indexed { 8 } else { bpp },
            imagedescriptor: 1 << 5, // Origin in upper left-hand corner
        }
    }

    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_i8(self.idlength)?;
        wrt.write_i8(self.colourmaptype)?;
        wrt.write_i8(self.datatypecode)?;
        wrt.write_u16::<LittleEndian>(self.colourmaporigin)?;
        wrt.write_u16::<LittleEndian>(self.colourmaplength)?;
        wrt.write_i8(self.colourmapdepth)?;
        wrt.write_u16::<LittleEndian>(self.x_origin)?;
        wrt.write_u16::<LittleEndian>(self.y_origin)?;
        wrt.write_i16::<LittleEndian>(self.width)?;
        wrt.write_i16::<LittleEndian>(self.height)?;
        wrt.write_i8(self.bitsperpixel)?;
        wrt.write_i8(self.imagedescriptor)?;
        Ok(())
    }
}
