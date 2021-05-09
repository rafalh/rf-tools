// This file is based on glfw crate.
// It is needed to handle loading buffers data without loading images.
// It allows this crate to have less dependencies, lower executable size and much faster conversion time.
// It is workaround for: https://github.com/gltf-rs/gltf/issues/222 

use gltf::{self, buffer};
use std::{fs, io, ops};
use std::path::Path;
use gltf::{Document, Error, Result};

#[derive(Clone, Debug)]
pub struct BufferData(pub Vec<u8>);

impl ops::Deref for BufferData {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

/// Represents the set of URI schemes the importer supports.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum Scheme<'a> {
    /// `data:[<media type>];base64,<data>`.
    Data(Option<&'a str>, &'a str),

    /// `file:[//]<absolute file path>`.
    ///
    /// Note: The file scheme does not implement authority.
    File(&'a str),

    /// `../foo`, etc.
    Relative,

    /// Placeholder for an unsupported URI scheme identifier.
    Unsupported,
}

impl<'a> Scheme<'a> {
    fn parse(uri: &str) -> Scheme {
        if uri.contains(':') {
            if let Some(stripped_uri) = uri.strip_prefix("data:") {
                let mut iter = stripped_uri.split(";base64,");
                if let Some(match0) = iter.next() {
                    if let Some(match1) = iter.next() {
                        Scheme::Data(Some(match0), match1)
                    } else {
                        Scheme::Data(None, match0)
                    }
                } else {
                    Scheme::Unsupported
                }
            } else if let Some(stripped_uri) = uri.strip_prefix("file://") {
                Scheme::File(stripped_uri)
            } else if let Some(stripped_uri) = uri.strip_prefix("file:") {
                Scheme::File(stripped_uri)
            } else {
                Scheme::Unsupported
            }
        } else {
            Scheme::Relative
        }
    }

    fn read(base: &Path, uri: &str) -> Result<Vec<u8>> {
        match Scheme::parse(uri) {
            Scheme::Data(_, base64) => base64::decode(&base64).map_err(|_| Error::Io(io::Error::new(io::ErrorKind::Other, "Base64"))),
            Scheme::File(path) => read_to_end(path),
            Scheme::Relative => read_to_end(base.join(uri)),
            Scheme::Unsupported => Err(Error::Io(io::Error::new(io::ErrorKind::Other, "Unsupported"))),
        }
    }
}

fn read_to_end<P>(path: P) -> Result<Vec<u8>>
where P: AsRef<Path>
{
    use io::Read;
    let file = fs::File::open(path.as_ref()).map_err(Error::Io)?;
    let length = file.metadata().map(|x| x.len()).unwrap_or(0);
    let mut reader = io::BufReader::new(file);
    let mut data = Vec::with_capacity(length as usize);
    reader.read_to_end(&mut data).map_err(Error::Io)?;
    Ok(data)
}

pub fn import_buffer_data(
    document: &Document,
    base: Option<&Path>,
    mut blob: Option<Vec<u8>>,
) -> Result<Vec<BufferData>> {
    let mut buffers = Vec::new();
    for buffer in document.buffers() {
        let mut data = match buffer.source() {
            buffer::Source::Uri(uri) if base.is_some() => Scheme::read(base.unwrap(), uri),
            buffer::Source::Bin => blob.take().ok_or_else(|| Error::Io(io::Error::new(io::ErrorKind::Other, "MissingBlob"))),
            _ => Err(Error::Io(io::Error::new(io::ErrorKind::Other, "ExternalReferenceInSliceImport")))
        }?;
        if data.len() < buffer.length() {
            return Err(Error::Io(io::Error::new(io::ErrorKind::Other, "BufferLength")));
        }
        while data.len() % 4 != 0 {
            data.push(0);
        }
        buffers.push(BufferData(data));
    }
    Ok(buffers)
}
