#![no_std]
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

use core::cmp::{max, min};
use core::fmt::{self, Debug, Display};

/// decoding error
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
	/// unexpected end-of-file
	UnexpectedEof,
	/// the buffer you provided is too small
	/// (see [`ImageHeader::required_bytes()`])
	BufferTooSmall,
	/// having the whole image in memory would require close to `usize::MAX` bytes
	TooLargeForUsize,
	/// this file is not a PNG file (missing PNG signature).
	NotPng,
	/// bad IHDR block (invalid PNG file)
	BadIhdr,
	/// unrecognized critical PNG chunk (invalid PNG file)
	UnrecognizedChunk,
	/// bad ZLIB block type (invalid PNG file)
	BadBlockType,
	/// ZLIB LEN doesn't match NLEN (invalid PNG file)
	BadNlen,
	/// decompressed data is larger than it should be (invalid PNG file)
	TooMuchData,
	/// unexpected end of PNG block (invalid PNG file)
	UnexpectedEob,
	/// bad zlib header (invalid PNG file)
	BadZlibHeader,
	/// bad huffman code (invalid PNG file)
	BadCode,
	/// bad huffman dictionary definition (invalid PNG file)
	BadHuffmanDict,
	/// bad LZ77 back reference (invalid PNG file)
	BadBackReference,
	/// unsupported interlace method (Adam7 interlacing is not currently supported)
	UnsupportedInterlace,
	/// bad filter number (invalid PNG file)
	BadFilter,
	/// bad PLTE chunk (invalid PNG file)
	BadPlteChunk,
	/// bad tRNS chunk (invalid PNG file)
	BadTrnsChunk,
	/// missing IDAT chunk (invalid PNG file)
	NoIdat,
	/// Adler-32 checksum doesn't check out (invalid PNG file)
	BadAdlerChecksum,
}

/// alias for `Result<T, Error>`
pub type Result<T> = core::result::Result<T, Error>;

impl Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::UnexpectedEof => write!(f, "unexpected end-of-file"),
			Self::NotPng => write!(f, "not a png file"),
			Self::BadIhdr => write!(f, "bad IHDR chunk"),
			Self::BufferTooSmall => write!(f, "provided buffer is too small"),
			Self::UnrecognizedChunk => write!(f, "unrecognized chunk type"),
			Self::BadBlockType => write!(f, "bad DEFLATE block type"),
			Self::TooMuchData => write!(f, "decompressed data is larger than it should be"),
			Self::UnexpectedEob => write!(f, "unexpected end of block"),
			Self::BadZlibHeader => write!(f, "bad zlib header"),
			Self::BadCode => write!(f, "bad code in DEFLATE data"),
			Self::BadHuffmanDict => write!(f, "bad Huffman dictionary definition"),
			Self::BadBackReference => {
				write!(f, "bad DEFLATE back reference (goes past start of stream)")
			}
			Self::TooLargeForUsize => write!(f, "decompressed data larger than usize::MAX bytes"),
			Self::UnsupportedInterlace => write!(f, "unsupported interlacing method"),
			Self::BadFilter => write!(f, "bad PNG filter"),
			Self::BadPlteChunk => write!(f, "bad PLTE chunk"),
			Self::BadTrnsChunk => write!(f, "bad tRNS chunk"),
			Self::NoIdat => write!(f, "missing IDAT chunk"),
			Self::BadNlen => write!(f, "LEN doesn't match NLEN"),
			Self::BadAdlerChecksum => write!(f, "bad adler-32 checksum"),
		}
	}
}

struct SliceReader<'a>(&'a [u8]);

impl<'a> From<&'a [u8]> for SliceReader<'a> {
	fn from(value: &'a [u8]) -> Self {
		Self(value)
	}
}

impl<'a> SliceReader<'a> {
	fn read(&mut self, buf: &mut [u8]) -> usize {
		let count = min(buf.len(), self.0.len());
		buf[..count].copy_from_slice(&self.0[..count]);
		self.0 = &self.0[count..];
		count
	}
	fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
		if self.read(buf) == buf.len() {
			Ok(())
		} else {
			Err(Error::UnexpectedEof)
		}
	}
	fn skip_bytes(&mut self, bytes: usize) -> Result<()> {
		if self.0.len() < bytes {
			return Err(Error::UnexpectedEof);
		}
		self.0 = &self.0[bytes..];
		Ok(())
	}
	fn empty_out(&mut self) {
		self.0 = &[][..];
	}
	fn take(&self, count: usize) -> SliceReader<'a> {
		self.0[..min(count, self.0.len())].into()
	}
}

struct IdatReader<'a> {
	block_reader: SliceReader<'a>,
	full_reader: &'a mut SliceReader<'a>,
	palette: Palette,
	header: ImageHeader,
	eof: bool,
}

impl<'a> IdatReader<'a> {
	fn new(reader: &'a mut SliceReader<'a>, header: ImageHeader) -> Result<Self> {
		let mut palette = [[0, 0, 0, 255]; 256];
		let Some(idat_len) = read_non_idat_chunks(reader, &header, &mut palette)? else {
			return Err(Error::NoIdat);
		};
		let idat_len: usize = idat_len.try_into().map_err(|_| Error::TooLargeForUsize)?;
		let block_reader = reader.take(idat_len);
		reader.skip_bytes(idat_len + 4)?;

		Ok(IdatReader {
			full_reader: reader,
			block_reader,
			header,
			palette,
			eof: false,
		})
	}

	fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
		let count = self.block_reader.read(buf);
		if count == buf.len() {
			Ok(buf.len())
		} else {
			match read_non_idat_chunks(self.full_reader, &self.header, &mut self.palette)? {
				None => {
					self.block_reader.empty_out();
					self.eof = true;
					Ok(count)
				}
				Some(n) => {
					let n = n as usize;
					self.block_reader = self.full_reader.take(n);
					self.full_reader.skip_bytes(n + 4)?; // skip block + CRC in full_reader
					Ok(self.read(&mut buf[count..])? + count)
				}
			}
		}
	}

	fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
		if self.read(buf)? == buf.len() {
			Ok(())
		} else {
			Err(Error::UnexpectedEof)
		}
	}

	fn read_to_end(&mut self) -> Result<()> {
		if !self.eof {
			self.block_reader.empty_out();
			self.eof = true;
			loop {
				match read_non_idat_chunks(self.full_reader, &self.header, &mut self.palette)? {
					None => break,
					Some(n) => self.full_reader.skip_bytes(n as usize + 4)?,
				}
			}
		}
		Ok(())
	}
}

/// color bit depth
///
/// note that [`Self::One`], [`Self::Two`], [`Self::Four`] are only used with
/// indexed and grayscale images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BitDepth {
	/// 1 bit per pixel
	One = 1,
	/// 2 bits per pixel
	Two = 2,
	/// 4 bits per pixel
	Four = 4,
	/// 8 bits per channel (most common)
	Eight = 8,
	/// 16 bits per channel
	Sixteen = 16,
}

/// color format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorType {
	/// grayscale
	Gray,
	/// grayscale + alpha
	GrayAlpha,
	/// RGB
	Rgb,
	/// RGBA
	Rgba,
	/// indexed color (each pixel is an index to be passed into [`ImageData::palette`])
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
			_ => return None,
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

	fn channels(self) -> u8 {
		match self {
			Self::Gray | Self::Indexed => 1,
			Self::GrayAlpha => 2,
			Self::Rgb => 3,
			Self::Rgba => 4,
		}
	}
}

/// image metadata found at the start of the PNG file.
#[derive(Debug, Clone, Copy)]
pub struct ImageHeader {
	width: u32,
	height: u32,
	length: usize,
	bit_depth: BitDepth,
	color_type: ColorType,
}

impl ImageHeader {
	/// width of image in pixels
	pub fn width(&self) -> u32 {
		self.width
	}
	/// height of image in pixels
	pub fn height(&self) -> u32 {
		self.height
	}

	/// bits per sample of image
	pub fn bit_depth(&self) -> BitDepth {
		self.bit_depth
	}
	/// number and type of color channels
	pub fn color_type(&self) -> ColorType {
		self.color_type
	}
	fn decompressed_size(&self) -> usize {
		(self.bytes_per_row() + 1) * self.height() as usize
	}

	/// number of bytes needed for [`decode_png`]
	pub fn required_bytes(&self) -> usize {
		self.decompressed_size()
	}

	/// number of bytes needed for [`decode_png`], followed by [`ImageData::convert_to_rgba8bpc`]
	pub fn required_bytes_rgba8bpc(&self) -> usize {
		max(
			self.required_bytes(),
			4 * self.width() as usize * self.height() as usize,
		)
	}

	/// number of bytes in a single row of pixels
	pub fn bytes_per_row(&self) -> usize {
		(self.width() as usize
			* usize::from(self.bit_depth() as u8)
			* usize::from(self.color_type().channels())
			+ 7) / 8
	}

	fn data_size(&self) -> usize {
		self.bytes_per_row() * self.height() as usize
	}
}

type Palette = [[u8; 4]; 256];

/// number of bits to read in each [`Read::read`] call.
type ReadBits = u32;
/// number of bits to store in the [`BitReader`] buffer.
type Bits = u64;

struct BitReader<'a> {
	inner: IdatReader<'a>,
	bits: Bits,
	bits_left: u8,
}

impl<'a> From<IdatReader<'a>> for BitReader<'a> {
	fn from(inner: IdatReader<'a>) -> Self {
		Self {
			inner,
			bits: 0,
			bits_left: 0,
		}
	}
}

impl BitReader<'_> {
	fn read_more_bits(&mut self) -> Result<()> {
		let mut new_bits = [0; ReadBits::BITS as usize / 8];
		self.inner.read(&mut new_bits)?;
		let new_bits = Bits::from(ReadBits::from_le_bytes(new_bits));
		self.bits |= new_bits << self.bits_left;
		self.bits_left += ReadBits::BITS as u8;
		Ok(())
	}

	fn peek_bits(&mut self, count: u8) -> Result<u32> {
		debug_assert!(count > 0 && u32::from(count) <= 31);
		if self.bits_left < count {
			self.read_more_bits()?;
		}
		Ok((self.bits as u32) & ((1 << count) - 1))
	}

	fn read_bits(&mut self, count: u8) -> Result<u32> {
		let bits = self.peek_bits(count)?;
		self.bits_left -= count;
		self.bits >>= count;
		Ok(bits)
	}

	/// at least `count` bits MUST have been peeked before calling this!
	fn skip_peeked_bits(&mut self, count: u8) {
		debug_assert!(self.bits_left >= count);
		self.bits_left -= count;
		self.bits >>= count;
	}

	fn read_bits_usize(&mut self, count: u8) -> Result<usize> {
		debug_assert!(u32::from(count) <= usize::BITS);
		self.read_bits(count).map(|x| x as usize)
	}

	fn read_bits_u8(&mut self, count: u8) -> Result<u8> {
		debug_assert!(count <= 8);
		self.read_bits(count).map(|x| x as u8)
	}

	fn read_bits_u16(&mut self, count: u8) -> Result<u16> {
		debug_assert!(count <= 16);
		self.read_bits(count).map(|x| x as u16)
	}

	fn read_aligned_bytes_exact(&mut self, buf: &mut [u8]) -> Result<()> {
		debug_assert_eq!(self.bits_left % 8, 0);
		let mut i = 0;
		while self.bits_left > 0 && i < buf.len() {
			buf[i] = self.read_bits_u8(8)?;
			i += 1;
		}
		self.inner.read_exact(&mut buf[i..])
	}
}

#[derive(Debug)]
struct DecompressedDataWriter<'a> {
	slice: &'a mut [u8],
	pos: usize,
}

impl<'a> From<&'a mut [u8]> for DecompressedDataWriter<'a> {
	fn from(slice: &'a mut [u8]) -> Self {
		Self { slice, pos: 0 }
	}
}

impl<'a> DecompressedDataWriter<'a> {
	fn write_byte(&mut self, byte: u8) -> Result<()> {
		match self.slice.get_mut(self.pos) {
			None => return Err(Error::TooMuchData),
			Some(p) => *p = byte,
		}
		self.pos += 1;
		Ok(())
	}

	fn copy(&mut self, distance: usize, length: usize) -> Result<()> {
		if self.pos < distance {
			return Err(Error::BadBackReference);
		}

		let mut src = self.pos - distance;
		let mut dest = self.pos;
		if length > self.slice.len() - dest {
			return Err(Error::TooMuchData);
		}
		for _ in 0..length {
			self.slice[dest] = self.slice[src];
			dest += 1;
			src += 1;
		}
		self.pos = dest;
		Ok(())
	}
}

const HUFFMAN_MAX_CODES: usize = 286;
const HUFFMAN_MAX_BITS: u8 = 15;
/// wow i benchmarked this and got the same optimal number as miniz. cool.
const HUFFMAN_MAIN_TABLE_BITS: u8 = 10;
const HUFFMAN_MAIN_TABLE_SIZE: usize = 1 << HUFFMAN_MAIN_TABLE_BITS;

/// table used for huffman lookup
///
/// the idea for this huffman table is stolen from miniz.
/// it's a combination of a look-up table and huffman tree.
/// for short codes, the look-up table returns a positive value
/// which is just the encoded value and length.
/// for long codes, the look-up table returns a position in the tree
/// to start from.
#[derive(Debug, Clone, Copy)]
struct HuffmanTable {
	main_table: [i16; HUFFMAN_MAIN_TABLE_SIZE],
	tree: [i16; HUFFMAN_MAX_CODES * 2 + 1],
	tree_used: i16,
}

impl Default for HuffmanTable {
	fn default() -> Self {
		Self {
			main_table: [0; HUFFMAN_MAIN_TABLE_SIZE],
			tree: [0; HUFFMAN_MAX_CODES * 2 + 1],
			// reserve "null" tree index
			tree_used: 1,
		}
	}
}

impl HuffmanTable {
	fn assign(&mut self, code: u16, length: u8, value: u16) {
		if length == 0 {
			return;
		}
		// reverse code
		let code = code.reverse_bits() >> (16 - length);

		if length <= HUFFMAN_MAIN_TABLE_BITS {
			// just throw it in the main table
			let increment = 1 << length;
			let mut i = usize::from(code);
			let entry = value as i16 | i16::from(length) << 9;
			// we need to account for all the possible bits that could appear after the code
			//  (since when we're decoding we read HUFFMAN_MAX_BITS bits regardless of the code length)
			for _ in 0..1u16 << (HUFFMAN_MAIN_TABLE_BITS - length) {
				self.main_table[i] = entry;
				i += increment;
			}
		} else {
			// put it in the tree
			let main_table_entry = usize::from(code) & (HUFFMAN_MAIN_TABLE_SIZE - 1);
			let mut code = code >> HUFFMAN_MAIN_TABLE_BITS;
			let mut entry = &mut self.main_table[main_table_entry];
			for _ in 0..length - HUFFMAN_MAIN_TABLE_BITS {
				if *entry == 0 {
					let i = self.tree_used;
					// allocate "left" and "right" branches of entry
					self.tree_used += 2;
					*entry = -i;
				} else {
					debug_assert!(*entry < 0);
				};
				entry = &mut self.tree[usize::from((-*entry) as u16 + (code & 1))];
				code >>= 1;
			}
			*entry = value as i16 | i16::from(length) << 9;
		}
	}

	fn from_code_lengths(code_lengths: &[u8]) -> Self {
		let mut bl_count = [0; HUFFMAN_MAX_BITS as usize + 1];
		for l in code_lengths.iter().copied() {
			bl_count[usize::from(l)] += 1;
		}
		bl_count[0] = 0;
		let mut next_code = [0; HUFFMAN_MAX_BITS as usize + 1];
		let mut code = 0;
		for bits in 1..=usize::from(HUFFMAN_MAX_BITS) {
			code = (code + bl_count[bits - 1]) << 1;
			next_code[bits] = code;
		}
		let mut table = HuffmanTable::default();
		for (i, l) in code_lengths.iter().copied().enumerate() {
			table.assign(next_code[usize::from(l)], l, i as u16);
			next_code[usize::from(l)] += 1;
		}
		table
	}

	fn lookup_slow(&self, mut entry: i16, mut code: u16) -> u16 {
		code >>= HUFFMAN_MAIN_TABLE_BITS;
		while entry < 0 {
			entry = self.tree[usize::from(code & 1) + (-entry) as usize];
			code >>= 1;
		}
		entry as u16
	}

	fn read_value(&self, reader: &mut BitReader) -> Result<u16> {
		let code = reader.peek_bits(HUFFMAN_MAX_BITS)? as u16;
		let entry = self.main_table[usize::from(code) & (HUFFMAN_MAIN_TABLE_SIZE - 1)];
		let entry = if entry > 0 {
			entry as u16
		} else {
			self.lookup_slow(entry, code)
		};
		let length = (entry >> 9) as u8;
		if length == 0 {
			return Err(Error::BadCode);
		}
		reader.skip_peeked_bits(length);
		Ok(entry & 0x1ff)
	}
}

/// image data
#[derive(Debug)]
pub struct ImageData<'a> {
	header: ImageHeader,
	buffer: &'a mut [u8],
	palette: Palette,
}

impl ImageData<'_> {
	/// get pixel values encoded as bytes.
	///
	/// this is guaranteed to be a prefix of the buffer passed to [`decode_png`].
	pub fn pixels(&self) -> &[u8] {
		&self.buffer[..self.header.data_size()]
	}

	/// get color in palette at index.
	///
	/// returns `[0, 0, 0, 255]` if `index` is out of range.
	pub fn palette(&self, index: u8) -> [u8; 4] {
		self.palette
			.get(usize::from(index))
			.copied()
			.unwrap_or([0, 0, 0, 255])
	}

	/// image width in pixels
	pub fn width(&self) -> u32 {
		self.header.width
	}

	/// image height in pixels
	pub fn height(&self) -> u32 {
		self.header.height
	}

	/// bits per sample of image
	pub fn bit_depth(&self) -> BitDepth {
		self.header.bit_depth
	}

	/// number and type of color channels
	pub fn color_type(&self) -> ColorType {
		self.header.color_type
	}

	/// number of bytes in a single row of pixels
	pub fn bytes_per_row(&self) -> usize {
		self.header.bytes_per_row()
	}

	/// convert `self` to 8-bits-per-channel RGBA
	///
	/// note: this function can fail with [`Error::BufferTooSmall`]
	///       if the buffer you allocated is too small!
	///       make sure to use [`ImageHeader::required_bytes_rgba8bpc`] for this.
	pub fn convert_to_rgba8bpc(&mut self) -> Result<()> {
		let bit_depth = self.bit_depth();
		let color_type = self.color_type();
		let row_bytes = self.bytes_per_row();
		let width = self.width() as usize;
		let height = self.height() as usize;
		let area = width * height;
		let palette = self.palette;
		let buffer = &mut self.buffer[..];
		if buffer.len() < 4 * area {
			return Err(Error::BufferTooSmall);
		}
		match (bit_depth, color_type) {
			(BitDepth::Eight, ColorType::Rgba) => {}
			(BitDepth::Eight, ColorType::Rgb) => {
				// we have to process the pixels in reverse
				// to avoid overwriting data we'll need later
				let mut dest = 4 * area;
				let mut src = 3 * area;
				for _ in 0..area {
					buffer[dest - 1] = 255;
					buffer[dest - 2] = buffer[src - 1];
					buffer[dest - 3] = buffer[src - 2];
					buffer[dest - 4] = buffer[src - 3];
					dest -= 4;
					src -= 3;
				}
			}
			(BitDepth::Sixteen, ColorType::Rgba) => {
				let mut dest = 0;
				let mut src = 0;
				for _ in 0..area {
					buffer[dest] = buffer[src];
					buffer[dest + 1] = buffer[src + 2];
					buffer[dest + 2] = buffer[src + 4];
					buffer[dest + 3] = buffer[src + 8];
					dest += 4;
					src += 8;
				}
			}
			(BitDepth::Sixteen, ColorType::Rgb) => {
				let mut dest = 0;
				let mut src = 0;
				for _ in 0..area {
					buffer[dest] = buffer[src];
					buffer[dest + 1] = buffer[src + 2];
					buffer[dest + 2] = buffer[src + 4];
					buffer[dest + 3] = 255;
					dest += 4;
					src += 6;
				}
			}
			(BitDepth::Eight, ColorType::Gray) => {
				let mut dest = 4 * area;
				let mut src = area;
				for _ in 0..area {
					buffer[dest - 1] = 255;
					buffer[dest - 2] = buffer[src - 1];
					buffer[dest - 3] = buffer[src - 1];
					buffer[dest - 4] = buffer[src - 1];
					dest -= 4;
					src -= 1;
				}
			}
			(BitDepth::Eight, ColorType::GrayAlpha) => {
				let mut dest = 4 * area;
				let mut src = 2 * area;
				for _ in 0..area {
					buffer[dest - 1] = buffer[src - 1];
					buffer[dest - 2] = buffer[src - 2];
					buffer[dest - 3] = buffer[src - 2];
					buffer[dest - 4] = buffer[src - 2];
					dest -= 4;
					src -= 2;
				}
			}
			(BitDepth::Sixteen, ColorType::Gray) => {
				let mut dest = 4 * area;
				let mut src = 2 * area;
				for _ in 0..area {
					buffer[dest - 1] = 255;
					buffer[dest - 2] = buffer[src - 2];
					buffer[dest - 3] = buffer[src - 2];
					buffer[dest - 4] = buffer[src - 2];
					dest -= 4;
					src -= 2;
				}
			}
			(BitDepth::Sixteen, ColorType::GrayAlpha) => {
				let mut i = 0;
				for _ in 0..area {
					// Ghi Glo Ahi Alo
					// i   i+1 i+2 i+3
					buffer[i + 3] = buffer[i + 2];
					buffer[i + 1] = buffer[i];
					buffer[i + 2] = buffer[i];
					i += 4;
				}
			}
			(BitDepth::Eight, ColorType::Indexed) => {
				let mut dest = 4 * area;
				let mut src = area;
				for _ in 0..area {
					let index: usize = buffer[src - 1].into();
					buffer[dest - 4..dest].copy_from_slice(&palette[index]);
					dest -= 4;
					src -= 1;
				}
			}
			(
				BitDepth::One | BitDepth::Two | BitDepth::Four,
				ColorType::Indexed | ColorType::Gray,
			) => {
				let mut dest = 4 * area;
				let bit_depth = bit_depth as u8;
				for row in (0..height).rev() {
					let mut src = row * row_bytes + row_bytes - 1;
					let excess_bits = (width % (8 / usize::from(bit_depth))) as u8 * bit_depth;
					let mut src_bit = if excess_bits > 0 { excess_bits } else { 8 };
					for _ in 0..width {
						if src_bit == 0 {
							src -= 1;
							src_bit = 8;
						}
						src_bit -= bit_depth;
						// NOTE: PNG uses most-significant-bit first, unlike everyone else in the world.
						let index: usize = ((buffer[src] >> (8 - bit_depth - src_bit))
							& ((1 << bit_depth) - 1))
							.into();
						buffer[dest - 4..dest].copy_from_slice(&palette[index]);
						dest -= 4;
					}
				}
			}
			(
				BitDepth::One | BitDepth::Two | BitDepth::Four,
				ColorType::Rgb | ColorType::Rgba | ColorType::GrayAlpha,
			)
			| (BitDepth::Sixteen, ColorType::Indexed) => unreachable!(),
		}

		self.header.bit_depth = BitDepth::Eight;
		self.header.color_type = ColorType::Rgba;
		Ok(())
	}
}

/// decode image metadata.
///
/// this function only needs to read a few bytes from the start of the file,
/// so it should be very fast.
pub fn decode_png_header(bytes: &[u8]) -> Result<ImageHeader> {
	let mut signature = [0; 8];
	let mut reader = SliceReader::from(bytes);
	if reader.read(&mut signature) < signature.len()
		|| signature != [137, 80, 78, 71, 13, 10, 26, 10]
	{
		return Err(Error::NotPng);
	}
	let mut ihdr = [0; 25];
	reader.read_exact(&mut ihdr)?;
	let ihdr_len = (u32::from_be_bytes([ihdr[0], ihdr[1], ihdr[2], ihdr[3]]) + 12) as usize;
	if &ihdr[4..8] != b"IHDR" || ihdr_len < ihdr.len() {
		return Err(Error::BadIhdr);
	}
	reader.skip_bytes(ihdr_len - ihdr.len())?;

	let width = u32::from_be_bytes([ihdr[8], ihdr[9], ihdr[10], ihdr[11]]);
	let height = u32::from_be_bytes([ihdr[12], ihdr[13], ihdr[14], ihdr[15]]);
	if width == 0 || height == 0 || width > 0x7FFF_FFFF || height > 0x7FFF_FFFF {
		return Err(Error::BadIhdr);
	}

	// worst-case scenario this is a RGBA 16bpc image
	// we could do a tighter check here but whatever
	//   on 32-bit this is only relevant for, like, >23000x23000 images
	if usize::try_from(width + 1)
		.ok()
		.and_then(|x| {
			usize::try_from(height)
				.ok()
				.and_then(|y| x.checked_mul(8).and_then(|c| c.checked_mul(y)))
		})
		.is_none()
	{
		return Err(Error::TooLargeForUsize);
	}

	let bit_depth = BitDepth::from_byte(ihdr[16]).ok_or(Error::BadIhdr)?;
	let color_type = ColorType::from_byte(ihdr[17]).ok_or(Error::BadIhdr)?;
	match (bit_depth, color_type) {
		(BitDepth::One | BitDepth::Two | BitDepth::Four, ColorType::Indexed | ColorType::Gray) => {}
		(
			BitDepth::One | BitDepth::Two | BitDepth::Four,
			ColorType::Rgb | ColorType::Rgba | ColorType::GrayAlpha,
		)
		| (BitDepth::Sixteen, ColorType::Indexed) => {
			return Err(Error::BadIhdr);
		}
		(BitDepth::Eight, _) => {}
		(
			BitDepth::Sixteen,
			ColorType::Rgb | ColorType::Rgba | ColorType::Gray | ColorType::GrayAlpha,
		) => {}
	}
	let compression = ihdr[18];
	let filter = ihdr[19];
	let interlace = ihdr[20];
	if compression != 0 || filter != 0 {
		return Err(Error::BadIhdr);
	}
	if interlace != 0 {
		return Err(Error::UnsupportedInterlace);
	}

	let hdr = ImageHeader {
		width,
		height,
		bit_depth,
		color_type,
		length: 8 + ihdr_len,
	};
	Ok(hdr)
}

fn read_dynamic_huffman_dictionary(reader: &mut BitReader) -> Result<(HuffmanTable, HuffmanTable)> {
	let literal_length_code_lengths_count = reader.read_bits_usize(5)? + 257;
	let distance_code_lengths_count = reader.read_bits_usize(5)? + 1;
	let code_length_code_lengths_count = reader.read_bits_usize(4)? + 4;
	let mut code_length_code_lengths = [0; 19];
	for i in 0..code_length_code_lengths_count {
		const ORDER: [u8; 19] = [
			16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
		];
		code_length_code_lengths[usize::from(ORDER[i])] = reader.read_bits_u8(3)?;
	}
	let code_length_table = HuffmanTable::from_code_lengths(&code_length_code_lengths);
	let mut code_lengths = [0; 286 + 32];
	let mut i = 0;
	let total_code_lengths = literal_length_code_lengths_count + distance_code_lengths_count;
	loop {
		let op = code_length_table.read_value(reader)? as u8;
		if op < 16 {
			code_lengths[i] = op;
			i += 1;
		} else if op == 16 {
			let rep = reader.read_bits_usize(2)? + 3;
			if i == 0 || i + rep > total_code_lengths {
				return Err(Error::BadHuffmanDict);
			}
			let l = code_lengths[i - 1];
			for _ in 0..rep {
				code_lengths[i] = l;
				i += 1;
			}
		} else if op == 17 {
			let rep = reader.read_bits_usize(3)? + 3;
			if i + rep > total_code_lengths {
				return Err(Error::BadHuffmanDict);
			}
			for _ in 0..rep {
				code_lengths[i] = 0;
				i += 1;
			}
		} else if op == 18 {
			let rep = reader.read_bits_usize(7)? + 11;
			if i + rep > total_code_lengths {
				return Err(Error::BadHuffmanDict);
			}
			for _ in 0..rep {
				code_lengths[i] = 0;
				i += 1;
			}
		} else {
			// since we only assigned 0..=18 in the huffman table,
			// we should never get a value outside that range.
			debug_assert!(false, "should not be reachable");
		}
		if i >= total_code_lengths {
			break;
		}
	}
	let literal_length_code_lengths = &code_lengths[0..min(literal_length_code_lengths_count, 286)];
	let distance_code_lengths = &code_lengths[literal_length_code_lengths_count
		..min(total_code_lengths, literal_length_code_lengths_count + 30)];
	Ok((
		HuffmanTable::from_code_lengths(literal_length_code_lengths),
		HuffmanTable::from_code_lengths(distance_code_lengths),
	))
}

fn get_fixed_huffman_dictionaries() -> (HuffmanTable, HuffmanTable) {
	let mut lit = HuffmanTable::default();
	let mut dist = HuffmanTable::default();
	for i in 0..=143 {
		lit.assign(0b00110000 + i, 8, i);
	}
	for i in 144..=255 {
		lit.assign(0b110010000 + (i - 144), 9, i);
	}
	for i in 256..=279 {
		lit.assign(i - 256, 7, i);
	}
	for i in 280..=285 {
		lit.assign(0b11000000 + (i - 280), 8, i);
	}
	for i in 0..30 {
		dist.assign(i, 5, i);
	}
	(lit, dist)
}

fn read_compressed_block(
	reader: &mut BitReader,
	writer: &mut DecompressedDataWriter,
	dynamic: bool,
) -> Result<()> {
	let (literal_length_table, distance_table) = if dynamic {
		read_dynamic_huffman_dictionary(reader)?
	} else {
		get_fixed_huffman_dictionaries()
	};

	fn parse_length(reader: &mut BitReader, literal_length: u16) -> Result<u16> {
		Ok(match literal_length {
			257..=264 => literal_length - 254,
			265..=284 => {
				const BASES: [u8; 20] = [
					11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195,
					227,
				];
				let base: u16 = BASES[usize::from(literal_length - 265)].into();
				let extra_bits = (literal_length - 261) as u8 / 4;
				let extra = reader.read_bits_u16(extra_bits)?;
				base + extra
			}
			285 => 258,
			_ => unreachable!(), // we only could've assigned up to 285.
		})
	}

	fn parse_distance(reader: &mut BitReader, distance_code: u16) -> Result<u16> {
		Ok(match distance_code {
			0..=3 => distance_code + 1,
			4..=29 => {
				const BASES: [u16; 26] = [
					5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
					2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
				];
				let base = BASES[usize::from(distance_code - 4)];
				let extra_bits = (distance_code - 2) as u8 / 2;
				let extra = reader.read_bits_u16(extra_bits)?;
				base + extra
			}
			_ => unreachable!(), // we only could've assigned up to 29.
		})
	}

	loop {
		let literal_length = literal_length_table.read_value(reader)?;
		match literal_length {
			0..=255 => {
				// literal
				writer.write_byte(literal_length as u8)?;
			}
			257.. => {
				// length + distance
				let length = parse_length(reader, literal_length)?;
				let distance_code = distance_table.read_value(reader)?;
				let distance = parse_distance(reader, distance_code)?;
				writer.copy(usize::from(distance), usize::from(length))?;
			}
			256 => {
				// end of block
				break;
			}
		}
	}
	Ok(())
}

fn read_uncompressed_block(
	reader: &mut BitReader,
	writer: &mut DecompressedDataWriter,
) -> Result<()> {
	reader.bits >>= reader.bits_left % 8;
	reader.bits_left -= reader.bits_left % 8;
	let len = reader.read_bits_u16(16)?;
	let nlen = reader.read_bits_u16(16)?;
	if len ^ nlen != 0xffff {
		return Err(Error::BadNlen);
	}
	let len: usize = len.into();
	if len > writer.slice.len() - writer.pos {
		return Err(Error::TooMuchData);
	}
	reader.read_aligned_bytes_exact(&mut writer.slice[writer.pos..writer.pos + len])?;
	writer.pos += len;
	Ok(())
}

fn read_image(reader: IdatReader, writer: &mut DecompressedDataWriter) -> Result<Palette> {
	let mut reader = BitReader::from(reader);
	// zlib header
	let cmf = reader.read_bits(8)?;
	let flags = reader.read_bits(8)?;
	// check zlib checksum
	if (cmf * 256 + flags) % 31 != 0 {
		return Err(Error::BadZlibHeader);
	}
	let compression_method = cmf & 0xf;
	let compression_info = cmf >> 4;
	if compression_method != 8 || compression_info > 7 {
		return Err(Error::BadZlibHeader);
	}
	// no preset dictionary
	if (flags & 0x100) != 0 {
		return Err(Error::BadZlibHeader);
	}

	let decompressed_size = reader.inner.header.decompressed_size();
	loop {
		let bfinal = reader.read_bits(1)?;
		let btype = reader.read_bits(2)?;
		match btype {
			0 => {
				// uncompressed block
				read_uncompressed_block(&mut reader, writer)?;
			}
			1 | 2 => {
				// compressed block
				read_compressed_block(&mut reader, writer, btype == 2)?;
			}
			_ => {
				// 0b11 is not a valid block type
				return Err(Error::BadBlockType);
			}
		}
		if bfinal != 0 {
			break;
		}
	}

	if cfg!(feature = "adler") {
		// Adler-32 checksum
		let padding = reader.bits_left % 8;
		if padding > 0 {
			reader.bits >>= padding;
			reader.bits_left -= padding;
		}
		// NOTE: currently `read_bits` doesn't support reads of 32 bits.
		let mut expected_adler = reader.read_bits(16)?;
		expected_adler |= reader.read_bits(16)? << 16;
		expected_adler = expected_adler.swap_bytes();

		const BASE: u32 = 65521;
		let mut s1: u32 = 1;
		let mut s2: u32 = 0;
		for byte in writer.slice[..decompressed_size].iter().copied() {
			s1 += u32::from(byte);
			if s1 > BASE {
				s1 -= BASE;
			}
			s2 += s1;
			if s2 > BASE {
				s2 -= BASE;
			}
		}
		let got_adler = s2 << 16 | s1;
		if got_adler != expected_adler {
			return Err(Error::BadAdlerChecksum);
		}
	}

	// padding bytes
	reader.inner.read_to_end()?;

	Ok(reader.inner.palette)
}

fn apply_filters(header: &ImageHeader, data: &mut [u8]) -> Result<()> {
	let mut s = 0;
	let mut d = 0;

	let x_byte_offset = max(
		1,
		usize::from(header.bit_depth as u8) * usize::from(header.color_type.channels()) / 8,
	);
	let scanline_bytes = header.bytes_per_row();
	for scanline in 0..header.height() {
		let filter = data[s];
		const FILTER_NONE: u8 = 0;
		const FILTER_SUB: u8 = 1;
		const FILTER_UP: u8 = 2;
		const FILTER_AVG: u8 = 3;
		const FILTER_PAETH: u8 = 4;

		s += 1;
		data.copy_within(s..s + scanline_bytes, d);
		match (filter, scanline == 0) {
			(FILTER_NONE, _) | (FILTER_UP, true) => {}
			(FILTER_SUB, _) => {
				for i in d + x_byte_offset..d + scanline_bytes {
					data[i] = data[i].wrapping_add(data[i - x_byte_offset]);
				}
			}
			(FILTER_UP, false) => {
				for i in d..d + scanline_bytes {
					data[i] = data[i].wrapping_add(data[i - scanline_bytes]);
				}
			}
			(FILTER_AVG, false) => {
				for i in d..d + x_byte_offset {
					data[i] = data[i].wrapping_add(data[i - scanline_bytes] / 2);
				}
				for i in d + x_byte_offset..d + scanline_bytes {
					data[i] = data[i].wrapping_add(
						((u32::from(data[i - scanline_bytes]) + u32::from(data[i - x_byte_offset]))
							/ 2) as u8,
					);
				}
			}
			(FILTER_AVG, true) => {
				for i in d + x_byte_offset..d + scanline_bytes {
					data[i] = data[i].wrapping_add(data[i - x_byte_offset] / 2);
				}
			}
			(FILTER_PAETH, false) => {
				for i in d..d + x_byte_offset {
					data[i] = data[i].wrapping_add(data[i - scanline_bytes]);
				}
				for i in d + x_byte_offset..d + scanline_bytes {
					let a = data[i - x_byte_offset];
					let b = data[i - scanline_bytes];
					let c = data[i - scanline_bytes - x_byte_offset];

					let p = i32::from(a) + i32::from(b) - i32::from(c);
					let pa = (p - i32::from(a)).abs();
					let pb = (p - i32::from(b)).abs();
					let pc = (p - i32::from(c)).abs();
					let paeth = if pa <= pb && pa <= pc {
						a
					} else if pb <= pc {
						b
					} else {
						c
					};
					data[i] = data[i].wrapping_add(paeth);
				}
			}
			(FILTER_PAETH, true) => {
				for i in d + x_byte_offset..d + scanline_bytes {
					data[i] = data[i].wrapping_add(data[i - x_byte_offset]);
				}
			}
			(5.., _) => return Err(Error::BadFilter),
		}

		s += scanline_bytes;
		d += scanline_bytes;
	}
	Ok(())
}

fn read_non_idat_chunks(
	reader: &mut SliceReader,
	header: &ImageHeader,
	palette: &mut Palette,
) -> Result<Option<u32>> {
	loop {
		let mut chunk_header = [0; 8];
		reader.read_exact(&mut chunk_header[..])?;
		let chunk_len: usize = u32::from_be_bytes([
			chunk_header[0],
			chunk_header[1],
			chunk_header[2],
			chunk_header[3],
		])
		.try_into()
		.map_err(|_| Error::TooLargeForUsize)?;
		let chunk_type = [
			chunk_header[4],
			chunk_header[5],
			chunk_header[6],
			chunk_header[7],
		];
		if &chunk_type == b"IEND" {
			reader.skip_bytes(4)?; // CRC
			break;
		} else if &chunk_type == b"IDAT" {
			return Ok(Some(chunk_len as u32));
		} else if &chunk_type == b"PLTE" && header.color_type == ColorType::Indexed {
			if chunk_len > 256 * 3 || chunk_len % 3 != 0 {
				return Err(Error::BadPlteChunk);
			}
			let count = chunk_len / 3;
			let mut data = [0; 256 * 3];
			reader.read_exact(&mut data[..chunk_len])?;
			for i in 0..count {
				palette[i][0..3].copy_from_slice(&data[3 * i..3 * i + 3]);
			}
			reader.skip_bytes(4)?; // CRC
		} else if &chunk_type == b"tRNS" && header.color_type == ColorType::Indexed {
			if chunk_len > 256 {
				return Err(Error::BadTrnsChunk);
			}
			let mut data = [0; 256];
			reader.read_exact(&mut data[..chunk_len])?;
			for i in 0..chunk_len {
				palette[i][3] = data[i];
			}
			reader.skip_bytes(4)?; // CRC
		} else if (chunk_type[0] & 0x20) != 0 || &chunk_type == b"PLTE" {
			// non-essential chunk
			reader.skip_bytes(chunk_len + 4)?;
		} else {
			return Err(Error::UnrecognizedChunk);
		}
	}
	Ok(None)
}

/// decode image data.
///
/// the only non-stack memory used by this function is `buf` — it should be at least
/// [`ImageHeader::required_bytes()`] bytes long, otherwise an [`Error::BufferTooSmall`]
/// will be returned.
pub fn decode_png<'a>(bytes: &[u8], buf: &'a mut [u8]) -> Result<ImageData<'a>> {
	let header = decode_png_header(bytes)?;
	let bytes = &bytes[header.length..];
	if buf.len() < header.required_bytes() {
		return Err(Error::BufferTooSmall);
	}

	let mut reader = SliceReader::from(bytes);
	let mut writer = DecompressedDataWriter::from(buf);
	let mut palette = read_image(IdatReader::new(&mut reader, header)?, &mut writer)?;

	if header.color_type == ColorType::Gray {
		// set palette appropriately so that conversion functions don't have
		// to deal with grayscale/indexed <8bpp separately.
		match header.bit_depth {
			BitDepth::One => {
				palette[0] = [0, 0, 0, 255];
				palette[1] = [255, 255, 255, 255];
			}
			BitDepth::Two => {
				// clippy's suggestion here is more unreadable imo
				#[allow(clippy::needless_range_loop)]
				for i in 0..4 {
					let v = (255 * i / 3) as u8;
					palette[i] = [v, v, v, 255];
				}
			}
			BitDepth::Four =>
			{
				#[allow(clippy::needless_range_loop)]
				for i in 0..16 {
					let v = (255 * i / 15) as u8;
					palette[i] = [v, v, v, 255];
				}
			}
			BitDepth::Eight | BitDepth::Sixteen => {}
		}
	}

	let buf = writer.slice;
	apply_filters(&header, buf)?;
	Ok(ImageData {
		buffer: buf,
		header,
		palette,
	})
}

#[cfg(test)]
mod tests {
	use super::*;
	extern crate alloc;

	fn assert_eq_bytes(bytes1: &[u8], bytes2: &[u8]) {
		assert_eq!(bytes1.len(), bytes2.len());
		for i in 0..bytes1.len() {
			assert_eq!(bytes1[i], bytes2[i]);
		}
	}

	fn test_bytes(bytes: &[u8]) {
		let decoder = png::Decoder::new(bytes);
		let mut reader = decoder.read_info().unwrap();

		let mut png_buf = alloc::vec![0; reader.output_buffer_size()];
		let png_header = reader.next_frame(&mut png_buf).unwrap();
		let png_bytes = &png_buf[..png_header.buffer_size()];

		let tiny_header = decode_png_header(bytes).unwrap();
		let mut tiny_buf = alloc::vec![0; tiny_header.required_bytes_rgba8bpc()];
		let mut image = decode_png(bytes, &mut tiny_buf).unwrap();
		let tiny_bytes = image.pixels();
		assert_eq_bytes(png_bytes, tiny_bytes);

		let (_, data) = png_decoder::decode(bytes).unwrap();
		image.convert_to_rgba8bpc().unwrap();
		assert_eq_bytes(&data[..], image.pixels());
	}

	macro_rules! test {
		($file:literal) => {
			test_bytes(include_bytes!(concat!("../", $file)));
		};
	}

	#[test]
	fn test_small() {
		test!("test/small.png");
	}
	#[test]
	fn test_small_rgb() {
		test!("test/small_rgb.png");
	}
	#[test]
	fn test_tiny1bpp_gray() {
		test!("test/tiny-1bpp-gray.png");
	}
	#[test]
	fn test_tiny2bpp() {
		test!("test/tiny-2bpp.png");
	}
	#[test]
	fn test_tiny_plte8bpp() {
		test!("test/tinyplte-8bpp.png");
	}
	#[test]
	fn test_gray_alpha() {
		test!("test/gray_alpha.png");
	}
	#[test]
	fn test_earth0() {
		test!("test/earth0.png");
	}
	#[test]
	fn test_earth9() {
		test!("test/earth9.png");
	}
	#[test]
	fn test_photograph() {
		test!("test/photograph.png");
	}
	#[test]
	fn test_earth_palette() {
		test!("test/earth_palette.png");
	}
	#[test]
	fn test_württemberg() {
		test!("test/württemberg.png");
	}
	#[test]
	fn test_endsleigh() {
		test!("test/endsleigh.png");
	}
	#[test]
	fn test_1qps() {
		test!("test/1QPS.png");
	}
	#[test]
	fn test_rabbit() {
		test!("test/rabbit.png");
	}
	#[test]
	fn test_basketball() {
		test!("test/basketball.png");
	}
	#[test]
	fn test_triangle() {
		test!("test/triangle.png");
	}
	#[test]
	fn test_iroquois() {
		test!("test/iroquois.png");
	}
	#[test]
	fn test_canada() {
		test!("test/canada.png");
	}
	#[test]
	fn test_berry() {
		test!("test/berry.png");
	}
	#[test]
	fn test_adam() {
		test!("test/adam.png");
	}
	#[test]
	fn test_nightingale() {
		test!("test/nightingale.png");
	}
	#[test]
	fn test_ratatoskr() {
		test!("test/ratatoskr.png");
	}
	#[test]
	fn test_cheerios() {
		test!("test/cheerios.png");
	}
	#[test]
	fn test_cavendish() {
		test!("test/cavendish.png");
	}
	#[test]
	fn test_ouroboros() {
		test!("test/ouroboros.png");
	}
	#[test]
	fn test_bad_png() {
		let mut data = &b"hello"[..];
		let err = decode_png_header(&mut data).unwrap_err();
		assert!(matches!(err, Error::NotPng));
	}
	#[test]
	fn test_buffer_too_small() {
		let png = &include_bytes!("../test/ouroboros.png")[..];
		let mut buffer = [0; 128];
		let err = decode_png(png, &mut buffer[..]).unwrap_err();
		assert!(matches!(err, Error::BufferTooSmall));
	}
}
