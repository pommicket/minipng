#![cfg_attr(not(feature = "std"), no_std)]

use core::cmp::{max, min};
use core::fmt::{self, Debug, Display};

pub trait IOError: Sized + Display + Debug {}
impl<T: Sized + Display + Debug> IOError for T {}

#[derive(Debug)]
#[non_exhaustive]
pub enum Error<I: IOError> {
	IO(I),
	NotPng,
	BadIhdr,
	UnrecognizedChunk([u8; 4]),
	BadBlockType,
	TooLargeForUsize,
	TooMuchData,
	UnexpectedEob,
	BadZlibHeader,
	BadCode,
	BadHuffmanCodes,
	BadBackReference,
	UnsupportedInterlace,
	BadFilter,
	BadPlteChunk,
	BadTrnsChunk,
	BadNlen,
	NoIdat,
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
			Self::UnrecognizedChunk([a, b, c, d]) => {
				write!(f, "unrecognized chunk type: {a} {b} {c} {d}")
			}
			Self::BadBlockType => write!(f, "bad DEFLATE block type"),
			Self::TooMuchData => write!(f, "decompressed data is larger than it should be"),
			Self::UnexpectedEob => write!(f, "unexpected end of block"),
			Self::BadZlibHeader => write!(f, "bad zlib header"),
			Self::BadCode => write!(f, "bad code in DEFLATE data"),
			Self::BadHuffmanCodes => write!(f, "bad Huffman codes"),
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

#[cfg(feature = "std")]
impl<T: std::io::Read + std::io::Seek> Read for std::io::BufReader<T> {
	type Error = std::io::Error;

	fn read(&mut self, buf: &mut [u8]) -> Result<(), Self::Error> {
		use std::io::Read;
		self.read_exact(buf)
	}
	fn skip_bytes(&mut self, bytes: usize) -> Result<(), Self::Error> {
		use std::io::Seek;
		self.seek(std::io::SeekFrom::Current(bytes as i64))
			.map(|_| ())
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
	fn skip_bytes(&mut self, bytes: usize) -> Result<(), Self::Error> {
		if self.len() < bytes {
			return Err(UnexpectedEofError);
		}
		*self = &self[bytes..];
		Ok(())
	}
}

struct IdatReader<'a, R: Read> {
	inner: &'a mut R,
	bytes_left_in_block: usize,
	palette: &'a mut [[u8; 4]; 256],
	header: &'a ImageHeader,
}

impl<R: Read> IdatReader<'_, R> {
	fn read_partial(&mut self, buf: &mut [u8]) -> Result<usize, Error<R::Error>> {
		if self.bytes_left_in_block >= buf.len() {
			self.inner.read(buf)?;
			self.bytes_left_in_block -= buf.len();
			Ok(buf.len())
		} else {
			if self.bytes_left_in_block > 0 {
				self.inner.read(&mut buf[..self.bytes_left_in_block])?;
			}
			let bytes_read = self.bytes_left_in_block;

			// CRC
			self.inner.skip_bytes(4)?;
			match read_non_idat_chunks(self.inner, self.header, self.palette)? {
				None => Ok(bytes_read),
				Some(n) => {
					self.bytes_left_in_block = n;
					Ok(self.read_partial(&mut buf[bytes_read..])? + bytes_read)
				}
			}
		}
	}

	fn read(&mut self, buf: &mut [u8]) -> Result<(), Error<R::Error>> {
		let count = self.read_partial(buf)?;
		if count == buf.len() {
			Ok(())
		} else {
			Err(Error::UnexpectedEob)
		}
	}

	fn read_to_end(&mut self) -> Result<(), Error<R::Error>> {
		self.inner.skip_bytes(self.bytes_left_in_block)?;
		// CRC
		self.inner.skip_bytes(4)?;
		loop {
			match read_non_idat_chunks(self.inner, self.header, self.palette)? {
				None => break,
				Some(n) => self.inner.skip_bytes(n + 4)?,
			}
		}
		Ok(())
	}
}

#[derive(Debug, Clone, Copy)]
pub struct ImageHeader {
	width: u32,
	height: u32,
	bit_depth: BitDepth,
	color_type: ColorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BitDepth {
	One = 1,
	Two = 2,
	Four = 4,
	Eight = 8,
	Sixteen = 16,
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
	fn checked_decompressed_size(&self) -> Option<usize> {
		let row_bytes = 1 + usize::try_from(self.width())
			.ok()?
			.checked_mul(usize::from(self.bit_depth() as u8))?
			.checked_mul(usize::from(self.color_type().channels()))?
			.checked_add(7)?
			/ 8;
		row_bytes.checked_mul(usize::try_from(self.height()).ok()?)
	}

	fn decompressed_size(&self) -> usize {
		self.checked_decompressed_size().unwrap()
	}

	fn checked_required_bytes(&self) -> Option<usize> {
		self.checked_decompressed_size()
	}
	pub fn required_bytes(&self) -> usize {
		self.checked_required_bytes().unwrap()
	}

	pub fn bytes_per_scanline(&self) -> usize {
		(self.width() as usize
			* usize::from(self.bit_depth() as u8)
			* usize::from(self.color_type().channels())
			+ 7) / 8
	}
	fn data_size(&self) -> usize {
		let scanline_bytes = self.bytes_per_scanline();
		scanline_bytes * self.height() as usize
	}
}

/// number of bits to read in each [`Read::read`] call.
type ReadBits = u32;
/// number of bits to store in the [`BitReader`] buffer.
type Bits = u64;

struct BitReader<'a, R: Read> {
	inner: IdatReader<'a, R>,
	bits: Bits,
	bits_left: u8,
}

impl<'a, R: Read> From<IdatReader<'a, R>> for BitReader<'a, R> {
	fn from(inner: IdatReader<'a, R>) -> Self {
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

	/// at least `count` bits MUST have been peeked before calling this!
	fn skip_peeked_bits(&mut self, count: u8) {
		debug_assert!(self.bits_left >= count);
		self.bits_left -= count;
		self.bits >>= count;
	}

	fn read_bits_usize(&mut self, count: u8) -> Result<usize, Error<R::Error>> {
		debug_assert!(u32::from(count) <= usize::BITS);
		self.read_bits(count).map(|x| x as usize)
	}

	fn read_bits_u8(&mut self, count: u8) -> Result<u8, Error<R::Error>> {
		debug_assert!(count <= 8);
		self.read_bits(count).map(|x| x as u8)
	}

	fn read_bits_u16(&mut self, count: u8) -> Result<u16, Error<R::Error>> {
		debug_assert!(count <= 16);
		self.read_bits(count).map(|x| x as u16)
	}

	fn read_aligned_bytes(&mut self, buf: &mut [u8]) -> Result<(), Error<R::Error>> {
		debug_assert_eq!(self.bits_left % 8, 0);
		let mut i = 0;
		while self.bits_left > 0 && i < buf.len() {
			buf[i] = self.read_bits_u8(8)?;
			i += 1;
		}
		self.inner.read(&mut buf[i..])
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
	};
	if hdr.checked_required_bytes().is_none() {
		return Err(Error::TooLargeForUsize);
	}
	Ok(hdr)
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
	fn write_byte<I: IOError>(&mut self, byte: u8) -> Result<(), Error<I>> {
		match self.slice.get_mut(self.pos) {
			None => return Err(Error::TooMuchData),
			Some(p) => *p = byte,
		}
		self.pos += 1;
		Ok(())
	}

	fn copy<I: IOError>(&mut self, distance: usize, length: usize) -> Result<(), Error<I>> {
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
const HUFFMAN_MAIN_TABLE_BITS: u8 = 11;
const HUFFMAN_MAIN_TABLE_SIZE: usize = 1 << HUFFMAN_MAIN_TABLE_BITS;
const HUFFMAN_SUBTABLE_SIZE: usize = 1 << (HUFFMAN_MAX_BITS - HUFFMAN_MAIN_TABLE_BITS);
#[derive(Debug)]
struct HuffmanTable {
	main_table: [u16; HUFFMAN_MAIN_TABLE_SIZE],
	subtables: [[u16; HUFFMAN_SUBTABLE_SIZE]; HUFFMAN_MAX_CODES],
	subtables_used: u16,
}

impl Default for HuffmanTable {
	fn default() -> Self {
		Self {
			main_table: [0; HUFFMAN_MAIN_TABLE_SIZE],
			subtables: [[0; HUFFMAN_SUBTABLE_SIZE]; HUFFMAN_MAX_CODES],
			subtables_used: 0,
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
			for i in 0..1u16 << (HUFFMAN_MAIN_TABLE_BITS - length) {
				self.main_table[usize::from(i << length | code)] = value | u16::from(length) << 10;
			}
		} else {
			// put it in a subtable.
			let main_table_entry = usize::from(code) & (HUFFMAN_MAIN_TABLE_SIZE - 1);
			let subtable_index = if self.main_table[main_table_entry] == 0 {
				let i = self.subtables_used;
				self.main_table[main_table_entry] = 0x200 | i;
				self.subtables_used += 1;
				i
			} else {
				self.main_table[main_table_entry] & 0x1ff
			};
			let subtable = &mut self.subtables[usize::from(subtable_index)];
			for i in 0..1u16 << (HUFFMAN_MAX_BITS - length) {
				subtable[usize::from(
					i << (length - HUFFMAN_MAIN_TABLE_BITS) | code >> HUFFMAN_MAIN_TABLE_BITS,
				)] = value | u16::from(length) << 10;
			}
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

	fn read_value<R: Read>(&self, reader: &mut BitReader<'_, R>) -> Result<u16, Error<R::Error>> {
		let code = reader.peek_bits(HUFFMAN_MAX_BITS)? as u16;
		let entry = self.main_table[usize::from(code) & (HUFFMAN_MAIN_TABLE_SIZE - 1)];
		let entry = if (entry & 0x200) == 0 {
			entry
		} else {
			self.subtables[usize::from(entry & 0x1ff)][usize::from(code >> HUFFMAN_MAIN_TABLE_BITS)]
		};
		let length = (entry >> 10) as u8;
		if length == 0 {
			return Err(Error::BadCode);
		}
		reader.skip_peeked_bits(length);
		Ok(entry & 0x1ff)
	}
}

fn read_compressed_block<R: Read>(
	reader: &mut BitReader<'_, R>,
	writer: &mut DecompressedDataWriter,
	dynamic: bool,
) -> Result<(), Error<R::Error>> {
	let literal_length_table;
	let distance_table;

	if dynamic {
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
					return Err(Error::BadHuffmanCodes);
				}
				let l = code_lengths[i - 1];
				for _ in 0..rep {
					code_lengths[i] = l;
					i += 1;
				}
			} else if op == 17 {
				let rep = reader.read_bits_usize(3)? + 3;
				if i + rep > total_code_lengths {
					return Err(Error::BadHuffmanCodes);
				}
				for _ in 0..rep {
					code_lengths[i] = 0;
					i += 1;
				}
			} else if op == 18 {
				let rep = reader.read_bits_usize(7)? + 11;
				if i + rep > total_code_lengths {
					return Err(Error::BadHuffmanCodes);
				}
				for _ in 0..rep {
					code_lengths[i] = 0;
					i += 1;
				}
			} else {
				debug_assert!(false, "should not be reachable");
			}
			if i >= total_code_lengths {
				break;
			}
		}
		let literal_length_code_lengths = &code_lengths[0..literal_length_code_lengths_count];
		let distance_code_lengths =
			&code_lengths[literal_length_code_lengths_count..total_code_lengths];
		literal_length_table = HuffmanTable::from_code_lengths(literal_length_code_lengths);
		distance_table = HuffmanTable::from_code_lengths(distance_code_lengths);
	} else {
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
		for i in 280..=287 {
			lit.assign(0b11000000 + (i - 280), 8, i);
		}
		for i in 0..30 {
			dist.assign(i, 5, i);
		}

		literal_length_table = lit;
		distance_table = dist;
	}
	loop {
		let literal_length = literal_length_table.read_value(reader)?;
		match literal_length {
			0..=255 => {
				// literal
				writer.write_byte(literal_length as u8)?;
			}
			256 => {
				// end of block
				break;
			}
			_ => {
				// length + distance
				let length = match literal_length {
					257..=264 => literal_length - 254,
					265..=284 => {
						const BASES: [u16; 20] = [
							11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
							163, 195, 227,
						];
						let base = BASES[usize::from(literal_length - 265)];
						let extra_bits = (literal_length - 261) as u8 / 4;
						let extra = reader.read_bits_u16(extra_bits)?;
						base + extra
					}
					285 => 258,
					_ => return Err(Error::BadCode),
				};

				let distance_code = distance_table.read_value(reader)?;
				let distance = match distance_code {
					0..=3 => distance_code + 1,
					4..=29 => {
						const BASES: [u16; 26] = [
							5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
							1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
						];
						let base = BASES[usize::from(distance_code - 4)];
						let extra_bits = (distance_code - 2) as u8 / 2;
						let extra = reader.read_bits_u16(extra_bits)?;
						base + extra
					}
					_ => return Err(Error::BadCode),
				};
				writer.copy(usize::from(distance), usize::from(length))?;
			}
		}
	}
	Ok(())
}

fn read_idat<R: Read>(
	reader: IdatReader<'_, R>,
	writer: &mut DecompressedDataWriter,
) -> Result<(), Error<R::Error>> {
	let mut reader = BitReader::from(reader);
	let _zlib_header = reader.read_bits(16);
	let decompressed_size = reader.inner.header.decompressed_size();
	while writer.pos < decompressed_size {
		let bfinal = reader.read_bits(1)?;
		let btype = reader.read_bits(2)?;
		if btype == 0 {
			// uncompressed block
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
			reader.read_aligned_bytes(&mut writer.slice[writer.pos..writer.pos + len])?;
			writer.pos += len;
		} else if btype == 1 || btype == 2 {
			// compressed block
			read_compressed_block(&mut reader, writer, btype == 2)?;
		} else {
			// 0b11 is not a valid block type
			return Err(Error::BadBlockType);
		}
		if bfinal != 0 {
			break;
		}
	}
	reader.inner.read_to_end()?;

	Ok(())
}

fn apply_filters<I: IOError>(header: &ImageHeader, data: &mut [u8]) -> Result<(), Error<I>> {
	let mut s = 0;
	let mut d = 0;

	let x_byte_offset = max(
		1,
		usize::from(header.bit_depth as u8) * usize::from(header.color_type.channels()) / 8,
	);
	let scanline_bytes = header.bytes_per_scanline();
	for scanline in 0..header.height() {
		let filter = data[s];
		s += 1;

		for i in 0..scanline_bytes {
			let x = i32::from(data[s]);
			let a = i32::from(if i < x_byte_offset {
				0
			} else {
				data[d - x_byte_offset]
			});
			let b = i32::from(if scanline == 0 {
				0
			} else {
				data[d - scanline_bytes]
			});
			let c = i32::from(if scanline == 0 || i < x_byte_offset {
				0
			} else {
				data[d - x_byte_offset - scanline_bytes]
			});

			fn paeth(a: i32, b: i32, c: i32) -> i32 {
				let p = a + b - c;
				let pa = (p - a).abs();
				let pb = (p - b).abs();
				let pc = (p - c).abs();
				if pa <= pb && pa <= pc {
					a
				} else if pb <= pc {
					b
				} else {
					c
				}
			}
			data[d] = (match filter {
				// none
				0 => x,
				// sub
				1 => x + a,
				// up
				2 => x + b,
				// average
				3 => x + (a + b) / 2,
				// paeth
				4 => x + paeth(a, b, c),
				_ => return Err(Error::BadFilter),
			}) as u8;
			s += 1;
			d += 1;
		}
	}
	Ok(())
}

#[derive(Debug)]
pub struct ImageData<'a> {
	header: ImageHeader,
	pixels: &'a [u8],
	palette: [[u8; 4]; 256],
}

impl ImageData<'_> {
	pub fn pixels(&self) -> &[u8] {
		self.pixels
	}
	pub fn palette(&self) -> &[[u8; 4]] {
		&self.palette[..]
	}
	pub fn width(&self) -> u32 {
		self.header.width
	}
	pub fn height(&self) -> u32 {
		self.header.height
	}
	pub fn bit_depth(&self) -> BitDepth {
		self.header.bit_depth
	}
	pub fn color_type(&self) -> ColorType {
		self.header.color_type
	}
}

fn read_non_idat_chunks<R: Read>(
	reader: &mut R,
	header: &ImageHeader,
	palette: &mut [[u8; 4]; 256],
) -> Result<Option<usize>, Error<R::Error>> {
	loop {
		let mut chunk_header = [0; 8];
		reader.read(&mut chunk_header[..])?;
		let chunk_len = u32::from_be_bytes([
			chunk_header[0],
			chunk_header[1],
			chunk_header[2],
			chunk_header[3],
		]) as usize;
		let chunk_type = [
			chunk_header[4],
			chunk_header[5],
			chunk_header[6],
			chunk_header[7],
		];
		if &chunk_type == b"IEND" {
			break;
		} else if &chunk_type == b"IDAT" {
			return Ok(Some(chunk_len));
		} else if &chunk_type == b"PLTE" && header.color_type == ColorType::Indexed {
			if chunk_len > 256 * 3 || chunk_len % 3 != 0 {
				return Err(Error::BadPlteChunk);
			}
			let count = chunk_len / 3;
			let mut data = [0; 256 * 3];
			reader.read(&mut data[..chunk_len])?;
			for i in 0..count {
				palette[i][0..3].copy_from_slice(&data[3 * i..3 * i + 3]);
			}
			// checksum
			reader.skip_bytes(4)?;
		} else if &chunk_type == b"tRNS" && header.color_type == ColorType::Indexed {
			if chunk_len > 256 {
				return Err(Error::BadTrnsChunk);
			}
			let mut data = [0; 256];
			reader.read(&mut data[..chunk_len])?;
			for i in 0..chunk_len {
				palette[i][3] = data[i];
			}
			// checksum
			reader.skip_bytes(4)?;
		} else if chunk_type[0].is_ascii_lowercase() || &chunk_type == b"PLTE" {
			// non-essential chunk
			reader.skip_bytes(chunk_len + 4)?;
		} else {
			return Err(Error::UnrecognizedChunk(chunk_type));
		}
	}
	Ok(None)
}

pub fn read_png<'a, R: Read>(
	reader: &mut R,
	header: Option<&ImageHeader>,
	buf: &'a mut [u8],
) -> Result<ImageData<'a>, Error<R::Error>> {
	let header = match header {
		None => read_png_header(reader)?,
		Some(h) => *h,
	};
	let mut writer = DecompressedDataWriter::from(buf);
	let mut palette = [[0, 0, 0, 0]; 256];
	let Some(idat_len) = read_non_idat_chunks(reader, &header, &mut palette)? else {
		return Err(Error::NoIdat);
	};
	read_idat(
		IdatReader {
			inner: reader,
			bytes_left_in_block: idat_len,
			header: &header,
			palette: &mut palette,
		},
		&mut writer,
	)?;

	let buf = writer.slice;
	apply_filters(&header, buf)?;
	Ok(ImageData {
		pixels: &buf[..header.data_size()],
		header,
		palette,
	})
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::fs::File;

	fn test_file(path: &str) {
		let decoder = png::Decoder::new(File::open(path).expect("file not found"));
		let mut reader = decoder.read_info().unwrap();

		let mut png_buf = vec![0; reader.output_buffer_size()];
		let png_header = reader.next_frame(&mut png_buf).unwrap();
		let png_bytes = &png_buf[..png_header.buffer_size()];

		let mut r = std::io::BufReader::new(File::open(path).expect("file not found"));
		let tiny_header = read_png_header(&mut r).unwrap();
		let mut tiny_buf = vec![0; tiny_header.required_bytes()];
		let tiny_bytes = read_png(&mut r, Some(&tiny_header), &mut tiny_buf)
			.unwrap()
			.pixels;

		assert_eq!(png_bytes.len(), tiny_bytes.len());
		assert_eq!(png_bytes, tiny_bytes);
	}

	fn test_bytes(mut bytes: &[u8]) {
		let decoder = png::Decoder::new(bytes);
		let mut reader = decoder.read_info().unwrap();

		let mut png_buf = vec![0; reader.output_buffer_size()];
		let png_header = reader.next_frame(&mut png_buf).unwrap();
		let png_bytes = &png_buf[..png_header.buffer_size()];

		let tiny_header = read_png_header(&mut bytes).unwrap();
		let mut tiny_buf = vec![0; tiny_header.required_bytes()];
		let tiny_bytes = read_png(&mut bytes, Some(&tiny_header), &mut tiny_buf)
			.unwrap()
			.pixels;

		assert_eq!(png_bytes.len(), tiny_bytes.len());
		assert_eq!(png_bytes, tiny_bytes);
	}

	macro_rules! test_both {
		($file:literal) => {
			test_file($file);
			test_bytes(include_bytes!(concat!("../", $file)));
		};
	}

	#[test]
	fn test_small() {
		test_both!("examples/small.png");
	}
	#[test]
	fn test_small_rgb() {
		test_both!("test/small_rgb.png");
	}
	#[test]
	fn test_small_rgba() {
		test_both!("test/small_rgba.png");
	}
	#[test]
	fn test_earth0() {
		test_both!("test/earth0.png");
	}
	#[test]
	fn test_earth9() {
		test_both!("test/earth9.png");
	}
	#[test]
	fn test_earth_palette() {
		test_both!("test/earth_palette.png");
	}
}
