//#![cfg_attr(not(feature = "std"), no_std)]

use core::cmp::min;
use core::fmt::{self, Debug, Display};

pub trait IOError: Sized + Display + Debug {
}
impl<T: Sized + Display + Debug> IOError for T {
}

#[cfg(feature = "std")]
impl IOError for std::io::Error {
	fn unexpected_eof() -> Self {
		std::io::ErrorKind::UnexpectedEof.into()
	}
}

#[derive(Debug)]
#[non_exhaustive]
pub enum Error<I: IOError> {
	IO(I),
	NotPng,
	BadIhdr,
	UnrecognizedChunk([u8; 4]),
	BadBlockType,
	TooMuchData,
	UnexpectedEob,
	BadZlibHeader,
}

impl<I: IOError> From<I> for Error<I> {
	fn from(value: I) -> Self {
		Self::IO(value)
	}
}

impl<I: IOError> Display for Error<I> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::IO(e) => write!(f, "{e}"),
			Self::NotPng => write!(f, "not a png file"),
			Self::BadIhdr => write!(f, "bad IHDR chunk"),
			Self::UnrecognizedChunk([a, b, c, d]) => write!(f, "unrecognized chunk type: {a} {b} {c} {d}"),
			Self::BadBlockType => write!(f, "bad DEFLATE block type"),
			Self::TooMuchData => write!(f, "decompressed data is larger than it should be"),
			Self::UnexpectedEob => write!(f, "unexpected end of block"),
			Self::BadZlibHeader => write!(f, "bad zlib header"),
		}
	}
}

#[cfg(feature = "std")]
impl<I: IOError> std::error::Error for Error<I> {}

pub trait Read {
	type Error: IOError;
	fn read(&mut self, buf: &mut [u8]) -> Result<(), Self::Error>;	
	fn skip_bytes(&mut self, count: usize) -> Result<(), Self::Error> {
		let mut count = count;
		const BUF_LEN: usize = 128;
		let mut buf = [0; BUF_LEN];
		while count > 0 {
			let c = min(BUF_LEN, count);
			self.read(&mut buf[..c])?;
			count -= c;
		}
		Ok(())
	}
}

#[derive(Debug)]
pub struct UnexpectedEofError;

impl core::fmt::Display for UnexpectedEofError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "unexpected EOF")
	}
}

impl<'a> Read for &'a [u8] {
	type Error = UnexpectedEofError;
	fn read(&mut self, buf: &mut [u8]) -> Result<(), Self::Error> {
		if self.len() < buf.len() {
			return Err(UnexpectedEofError);
		}
		buf.copy_from_slice(&self[..buf.len()]);
		*self = &self[buf.len()..];
		Ok(())
	}
	// TODO: skip_bytes implementation
}

struct BlockReader<'a, R: Read> {
	inner: &'a mut R,
	bytes_left: usize
}

impl<'a, R: Read> BlockReader<'_, R> {
	fn read(&mut self, buf: &mut [u8]) -> Result<(), Error<R::Error>> {
		if buf.len() > self.bytes_left {
			return Err(Error::UnexpectedEob);
		}
		Ok(self.inner.read(buf)?)
	}
	
	fn read_partial(&mut self, buf: &mut [u8]) -> Result<usize, Error<R::Error>> {
		let count = min(self.bytes_left, buf.len());
		self.read(&mut buf[..count])?;
		Ok(count)
	}
}

#[derive(Debug)]
pub struct ImageHeader {
	width: u32,
	height: u32,
	bit_depth: BitDepth,
	color_type: ColorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitDepth {
	One,
	Two,
	Four,
	Eight,
	Sixteen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorType {
	Gray,
	GrayAlpha,
	Rgb,
	Rgba,
	Indexed,
}

impl BitDepth {
	fn from_byte(x: u8) -> Option<Self> {
		Some(match x {
			1 => Self::One,
			2 => Self::Two,
			4 => Self::Four,
			8 => Self::Eight,
			16 => Self::Sixteen,
			_ => return None
		})
	}
}

impl ColorType {
	fn from_byte(x: u8) -> Option<Self> {
		Some(match x {
			0 => Self::Gray,
			2 => Self::Rgb,
			3 => Self::Indexed,
			4 => Self::GrayAlpha,
			6 => Self::Rgba,
			_ => return None,
		})
	}
}

impl ImageHeader {
	pub fn width(&self) -> u32 {
		self.width
	}
	pub fn height(&self) -> u32 {
		self.height
	}
	pub fn bit_depth(&self) -> BitDepth {
		self.bit_depth
	}
	pub fn color_type(&self) -> ColorType {
		self.color_type
	}
}

/// number of bits to read in each [`Read::read`] call.
///
/// don't change this to something bigger than `u32`, since we don't want to overread past the zlib checksum.
type ReadBits = u32;
/// number of bits to store in the [`BitReader`] buffer.
type Bits = u64;

struct BitReader<'a, R: Read> {
	inner: BlockReader<'a, R>,
	bits: Bits,
	bits_left: u8,
}

impl<'a, R: Read> From<BlockReader<'a, R>> for BitReader<'a, R> {
	fn from(inner: BlockReader<'a, R>) -> Self {
		Self {
			inner,
			bits: 0,
			bits_left: 0,
		}
	}
}

impl<R: Read> BitReader<'_, R> {
	fn peek_bits(&mut self, count: u8) -> Result<u32, Error<R::Error>> {
		debug_assert!(count > 0 && u32::from(count) <= 31);
		if self.bits_left < count {
			// read more bits
			let mut new_bits = [0; ReadBits::BITS as usize / 8];
			self.inner.read_partial(&mut new_bits)?;
			let new_bits = Bits::from(ReadBits::from_le_bytes(new_bits));
			self.bits |= new_bits << self.bits_left;
			self.bits_left += ReadBits::BITS as u8;
		}
		Ok((self.bits as u32) & ((1 << count) - 1))
	}
	
	fn read_bits(&mut self, count: u8) -> Result<u32, Error<R::Error>> {
		let bits = self.peek_bits(count)?;
		self.bits_left -= count;
		self.bits >>= count;
		Ok(bits)
	}
}


pub fn read_png_header<R: Read>(reader: &mut R) -> Result<ImageHeader, Error<R::Error>> {
	let mut signature = [0; 8];
	reader.read(&mut signature)?;
	if signature != [137, 80, 78, 71, 13, 10, 26, 10] {
		return Err(Error::NotPng);
	}
	let mut ihdr = [0; 25];
	reader.read(&mut ihdr)?;
	let ihdr_len = (u32::from_be_bytes([ihdr[0], ihdr[1], ihdr[2], ihdr[3]]) + 12) as usize;
	if &ihdr[4..8] != b"IHDR" || ihdr_len < ihdr.len() {
		return Err(Error::BadIhdr);
	}
	reader.skip_bytes(ihdr_len - ihdr.len())?;

	let width = u32::from_be_bytes([ihdr[8], ihdr[9], ihdr[10], ihdr[11]]);
	let height = u32::from_be_bytes([ihdr[12], ihdr[13], ihdr[14], ihdr[15]]);
	let bit_depth = BitDepth::from_byte(ihdr[16]).ok_or(Error::BadIhdr)?;
	let color_type = ColorType::from_byte(ihdr[17]).ok_or(Error::BadIhdr)?;
	let compression = ihdr[18];
	let filter = ihdr[19];
	if compression != 0 || filter != 0 {
		return Err(Error::BadIhdr);
	}
	
	Ok(ImageHeader { width, height, bit_depth, color_type })
}

struct ByteWriter<'a> {
	slice: &'a mut [u8],
}

fn read_idat<R: Read>(mut reader: BlockReader<'_, R>, writer: &mut ByteWriter) -> Result<(), Error<R::Error>> {
	let mut zlib_header = [0; 2];
	reader.read(&mut zlib_header)?;
	
	let mut reader = BitReader::from(reader);
	loop {
		let bfinal = reader.read_bits(1)?;
		let btype = reader.read_bits(2)?;
		if btype == 0 {
			// uncompressed block
			let len = reader.read_bits(16)? as usize;
			reader.read_bits(16)?; // nlen
			if writer.slice.len() < len {
				return Err(Error::TooMuchData);
			}
			reader.inner.read(&mut writer.slice[..len])?;
		} else if btype == 1 || btype == 2 {
			// compressed block
			todo!("{btype}")
		} else {
			// 0b11 is not a valid block type
			return Err(Error::BadBlockType);
		}
		
		if bfinal != 0 {
			break;
		}
	}
	
	// skip checksum
	let mut trash = [0; 4];
	reader.inner.read(&mut trash[..4 - usize::from(reader.bits_left) / 8])?;
	
	Ok(())
}

pub fn read_png<R: Read>(
	reader: &mut R,
	header: Option<ImageHeader>,
	buf: &mut [u8],
) -> Result<(), Error<R::Error>> {
	let _header = match header {
		None => read_png_header(reader)?,
		Some(h) => h,
	};
	let mut writer = ByteWriter { slice: buf };
	loop {
		let mut chunk_header = [0; 8];
		reader.read(&mut chunk_header[..])?;
		let chunk_len = u32::from_be_bytes([chunk_header[0], chunk_header[1], chunk_header[2], chunk_header[3]]) as usize;
		let chunk_type = [chunk_header[4], chunk_header[5], chunk_header[6], chunk_header[7]];
		if &chunk_type == b"IEND" {
			break;
		} else if &chunk_type == b"IDAT" {
			read_idat(BlockReader {
				inner: reader,
				bytes_left: chunk_len
			}, &mut writer)?;
		} else if &chunk_type == b"PLTE" {
			todo!();
		} else if chunk_type[0].is_ascii_lowercase() {
			// non-essential chunk
		} else {
			return Err(Error::UnrecognizedChunk(chunk_type));
		}
		
		reader.skip_bytes(chunk_len + 4)?;
	}
	
	Ok(())
}

