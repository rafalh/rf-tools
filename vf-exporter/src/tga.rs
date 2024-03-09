use binrw::BinWrite;

#[derive(BinWrite)]
pub struct TgaHeader {
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

impl TgaHeader {
    pub fn new(w: i16, h: i16, bits_per_pixel: i8) -> Self {
        Self { 
            idlength: 0,
            colourmaptype: 0,
            datatypecode: 2,
            colourmaporigin: 0,
            colourmaplength: 0,
            colourmapdepth: 0,
            x_origin: 0,
            y_origin: 0,
            width: w,
            height: h,
            bitsperpixel: bits_per_pixel,
            imagedescriptor: 1 << 5, // Origin in upper left-hand corner
        }
    }
}
