use std::io::Write;
use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};

pub(crate) trait WriteExt: Write {
    fn write_f32_slice_le(&mut self, slice: &[f32]) -> std::io::Result<()> {
        for val in slice {
            self.write_f32::<LittleEndian>(*val)?;
        }
        Ok(())
    }

    fn write_f32_slice<T: ByteOrder>(&mut self, slice: &[f32]) -> std::io::Result<()> {
        for val in slice {
            self.write_f32::<T>(*val)?;
        }
        Ok(())
    }

    fn write_i16_slice<T: ByteOrder>(&mut self, slice: &[i16]) -> std::io::Result<()> {
        for val in slice {
            self.write_i16::<T>(*val)?;
        }
        Ok(())
    }
    
    fn write_char_array(&mut self, string: &str, size: usize) -> std::io::Result<()> {
        let bytes = string.as_bytes();
        if bytes.len() >= size {
            return Err(new_custom_error(format!("string value {} is too long (max {})", string, size - 1)));
        }
        self.write_all(bytes)?;
        let padding = vec![0u8; size - bytes.len()];
        self.write_all(&padding)?;
        Ok(())
    }
}

impl<T: Write> WriteExt for T {}

pub(crate) fn new_custom_error<S: Into<String>>(msg: S) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, msg.into())
}
