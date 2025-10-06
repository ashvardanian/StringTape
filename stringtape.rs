#![cfg_attr(not(feature = "std"), no_std)]

//! # StringTape
//!
//! Memory-efficient string and bytes storage compatible with Apache Arrow.
//!
//! ## CharsTape - Sequential String Storage
//!
//! ```rust
//! use stringtape::{CharsTapeI32, StringTapeError};
//!
//! let mut tape = CharsTapeI32::new();
//! tape.push("hello")?;
//! tape.push("world")?;
//!
//! assert_eq!(tape.len(), 2);
//! assert_eq!(&tape[0], "hello");
//!
//! // Iterate over strings
//! for s in &tape {
//!     println!("{}", s);
//! }
//! # Ok::<(), StringTapeError>(())
//! ```
//!
//! ## CharsCows - Compressed Arbitrary-Order Slices
//!
//! For extremely large datasets, use `CharsCows` with configurable offset/length types:
//!
//! ```rust
//! use stringtape::{CharsCowsU32U16, StringTapeError};
//! use std::borrow::Cow;
//!
//! let data = "hello world foo bar";
//! // 6 bytes per entry (u32 offset + u16 length) vs 24+ bytes for Vec<String>
//! let cows = CharsCowsU32U16::from_iter_and_data(
//!     data.split_whitespace(),
//!     Cow::Borrowed(data.as_bytes())
//! )?;
//!
//! assert_eq!(&cows[0], "hello");
//! assert_eq!(&cows[3], "bar");
//! # Ok::<(), StringTapeError>(())
//! ```
//!
//! ## BytesTape - Binary Data
//!
//! ```rust
//! use stringtape::{BytesTapeI32, StringTapeError};
//!
//! let mut tape = BytesTapeI32::new();
//! tape.push(&[0xde, 0xad, 0xbe, 0xef])?;
//! tape.push(b"bytes")?;
//!
//! assert_eq!(&tape[1], b"bytes" as &[u8]);
//! # Ok::<(), StringTapeError>(())
//! ```

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

use core::fmt;
use core::marker::PhantomData;
use core::ops::{
    Index, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive, Sub,
};
use core::ptr::{self, NonNull};
use core::slice;

#[cfg(not(feature = "std"))]
use alloc::borrow::Cow;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::borrow::Cow;

use allocator_api2::alloc::{Allocator, Global, Layout};

/// Errors that can occur when working with tape classes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringTapeError {
    /// Data size exceeds offset type maximum (e.g., >2GB for 32-bit offsets).
    OffsetOverflow,
    /// Memory allocation failed.
    AllocationError,
    /// Index out of bounds.
    IndexOutOfBounds,
    /// Invalid UTF-8 sequence.
    Utf8Error(core::str::Utf8Error),
}

impl fmt::Display for StringTapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StringTapeError::OffsetOverflow => write!(f, "offset value too large for offset type"),
            StringTapeError::AllocationError => write!(f, "memory allocation failed"),
            StringTapeError::IndexOutOfBounds => write!(f, "index out of bounds"),
            StringTapeError::Utf8Error(e) => write!(f, "invalid UTF-8: {}", e),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StringTapeError {}

/// A memory-efficient string storage structure compatible with Apache Arrow.
///
/// `CharsTape` stores multiple strings in a contiguous memory layout using offset-based
/// indexing, similar to Apache Arrow's String and LargeString arrays. All string data
/// is stored in a single buffer, with a separate offset array tracking string boundaries.
///
/// # Type Parameters
///
/// * `Offset` - Offset type (`i32`, `i64`, `u32`, `u64`)
/// * `A` - Allocator type (defaults to `Global`)
///
/// # Example
///
/// ```rust
/// use stringtape::{CharsTapeI32, StringTapeError};
///
/// let mut tape = CharsTapeI32::new();
/// tape.push("hello")?;
/// assert_eq!(&tape[0], "hello");
/// # Ok::<(), StringTapeError>(())
/// ```
///
/// Memory layout compatible with Apache Arrow:
/// ```text
/// Data:    [h,e,l,l,o,w,o,r,l,d]
/// Offsets: [0, 5, 10]
/// ```
struct RawTape<Offset: OffsetType, A: Allocator> {
    data: Option<NonNull<[u8]>>,
    offsets: Option<NonNull<[Offset]>>,
    len_bytes: usize,
    len_items: usize,
    allocator: A,
    _phantom: PhantomData<Offset>,
}

/// Named raw parts returned by `as_raw_parts` methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawParts<Offset: OffsetType> {
    /// Pointer to the start of the contiguous data buffer.
    pub data_ptr: *const u8,
    /// Pointer to the start of the offsets buffer.
    pub offsets_ptr: *const Offset,
    /// Number of bytes of valid data in `data_ptr`.
    pub data_len: usize,
    /// Number of items stored (strings/bytes entries).
    pub items_count: usize,
}

/// UTF-8 string view over `RawTape`.
pub struct CharsTape<Offset: OffsetType = i32, A: Allocator = Global> {
    inner: RawTape<Offset, A>,
}

/// Binary bytes view over `RawTape`.
pub struct BytesTape<Offset: OffsetType = i32, A: Allocator = Global> {
    inner: RawTape<Offset, A>,
}

/// Zero-copy read-only view into a RawTape slice.
pub struct RawTapeView<'a, Offset: OffsetType> {
    data: &'a [u8],
    offsets: &'a [Offset],
}

/// UTF-8 string view over `RawTapeView`.
pub struct CharsTapeView<'a, Offset: OffsetType = i32> {
    inner: RawTapeView<'a, Offset>,
}

/// Binary bytes view over `RawTapeView`.
pub struct BytesTapeView<'a, Offset: OffsetType = i32> {
    inner: RawTapeView<'a, Offset>,
}

/// Trait for offset types used in CharsTape.
///
/// Implementations: `i32`/`i64` (Arrow-compatible), `u32`/`u64` (unsigned, no Arrow interop).
pub trait OffsetType: Copy + Default + PartialOrd + Sub<Output = Self> {
    /// Size of the offset type in bytes.
    const SIZE: usize;

    /// Convert a usize value to this offset type.
    ///
    /// Returns `None` if the value is too large to be represented by this offset type.
    fn from_usize(value: usize) -> Option<Self>;

    /// Convert this offset value to usize.
    fn to_usize(self) -> usize;
}

impl OffsetType for i32 {
    const SIZE: usize = 4;

    fn from_usize(value: usize) -> Option<Self> {
        if value <= i32::MAX as usize {
            Some(value as i32)
        } else {
            None
        }
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl OffsetType for i64 {
    const SIZE: usize = 8;

    fn from_usize(value: usize) -> Option<Self> {
        Some(value as i64)
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl OffsetType for u16 {
    const SIZE: usize = 2;

    fn from_usize(value: usize) -> Option<Self> {
        if value <= u16::MAX as usize {
            Some(value as u16)
        } else {
            None
        }
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl OffsetType for u32 {
    const SIZE: usize = 4;

    fn from_usize(value: usize) -> Option<Self> {
        if value <= u32::MAX as usize {
            Some(value as u32)
        } else {
            None
        }
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl OffsetType for u64 {
    const SIZE: usize = 8;

    fn from_usize(value: usize) -> Option<Self> {
        Some(value as u64)
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

/// Trait for length types used in slice collections.
///
/// This trait defines the interface for length types that can be used to represent
/// the length of string cows. Implementations are provided for `u8`, `u16`, `u32`, and `u64`.
pub trait LengthType: Copy + Default + PartialOrd {
    /// Size of the length type in bytes.
    const SIZE: usize;

    /// Convert a usize value to this length type.
    ///
    /// Returns `None` if the value is too large to be represented by this length type.
    fn from_usize(value: usize) -> Option<Self>;

    /// Convert this length value to usize.
    fn to_usize(self) -> usize;
}

impl LengthType for u8 {
    const SIZE: usize = 1;

    fn from_usize(value: usize) -> Option<Self> {
        if value <= u8::MAX as usize {
            Some(value as u8)
        } else {
            None
        }
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl LengthType for u16 {
    const SIZE: usize = 2;

    fn from_usize(value: usize) -> Option<Self> {
        if value <= u16::MAX as usize {
            Some(value as u16)
        } else {
            None
        }
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl LengthType for u32 {
    const SIZE: usize = 4;

    fn from_usize(value: usize) -> Option<Self> {
        if value <= u32::MAX as usize {
            Some(value as u32)
        } else {
            None
        }
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl LengthType for u64 {
    const SIZE: usize = 8;

    fn from_usize(value: usize) -> Option<Self> {
        Some(value as u64)
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl<Offset: OffsetType, A: Allocator> RawTape<Offset, A> {
    /// Creates a new, empty CharsTape with the global allocator.
    ///
    /// This operation is O(1) and does not allocate memory until the first string is pushed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsTapeI32;
    ///
    /// let tape = CharsTapeI32::new();
    /// assert!(tape.is_empty());
    /// assert_eq!(tape.len(), 0);
    /// ```
    pub fn new() -> RawTape<Offset, Global> {
        RawTape::new_in(Global)
    }

    /// Creates a new, empty CharsTape with a custom allocator.
    ///
    /// This operation is O(1) and does not allocate memory until the first string is pushed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsTape;
    /// use allocator_api2::alloc::Global;
    ///
    /// let tape: CharsTape<i32, Global> = CharsTape::new_in(Global);
    /// assert!(tape.is_empty());
    /// assert_eq!(tape.len(), 0);
    /// ```
    pub fn new_in(allocator: A) -> Self {
        Self {
            data: None,
            offsets: None,
            len_bytes: 0,
            len_items: 0,
            allocator,
            _phantom: PhantomData,
        }
    }

    /// Creates a tape with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `data_capacity` - Bytes for string data
    /// * `strings_capacity` - Number of string slots
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::{CharsTapeI32, StringTapeError};
    ///
    /// // Pre-allocate space for ~1KB of string data and 100 strings
    /// let tape = CharsTapeI32::with_capacity(1024, 100)?;
    /// assert_eq!(tape.data_capacity(), 1024);
    /// # Ok::<(), StringTapeError>(())
    /// ```
    pub fn with_capacity(
        data_capacity: usize,
        strings_capacity: usize,
    ) -> Result<RawTape<Offset, Global>, StringTapeError> {
        RawTape::with_capacity_in(data_capacity, strings_capacity, Global)
    }

    /// Creates a new CharsTape with pre-allocated capacity and a custom allocator.
    ///
    /// Pre-allocating capacity can improve performance when you know approximately
    /// how much data you'll be storing.
    ///
    /// # Arguments
    ///
    /// * `data_capacity` - Number of bytes to pre-allocate for string data
    /// * `strings_capacity` - Number of string slots to pre-allocate
    /// * `allocator` - The allocator to use for memory management
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::{CharsTape, StringTapeError};
    /// use allocator_api2::alloc::Global;
    ///
    /// let tape: CharsTape<i32, Global> = CharsTape::with_capacity_in(1024, 100, Global)?;
    /// assert_eq!(tape.data_capacity(), 1024);
    /// # Ok::<(), StringTapeError>(())
    /// ```
    pub fn with_capacity_in(
        data_capacity: usize,
        strings_capacity: usize,
        allocator: A,
    ) -> Result<Self, StringTapeError> {
        let mut tape = Self::new_in(allocator);
        tape.reserve(data_capacity, strings_capacity)?;
        Ok(tape)
    }

    pub fn reserve(
        &mut self,
        additional_bytes: usize,
        additional_strings: usize,
    ) -> Result<(), StringTapeError> {
        if additional_bytes > 0 {
            let current_capacity = self.data_capacity();
            let new_capacity = current_capacity
                .checked_add(additional_bytes)
                .ok_or(StringTapeError::AllocationError)?;
            self.grow_data(new_capacity)?;
        }

        if additional_strings > 0 {
            let current_capacity = self.offsets_capacity();
            let new_capacity = current_capacity
                .checked_add(additional_strings + 1)
                .ok_or(StringTapeError::AllocationError)?;
            self.grow_offsets(new_capacity)?;
        }
        Ok(())
    }

    fn grow_data(&mut self, new_capacity: usize) -> Result<(), StringTapeError> {
        let current_capacity = self.data_capacity();
        if new_capacity <= current_capacity {
            return Ok(());
        }

        let new_layout =
            Layout::array::<u8>(new_capacity).map_err(|_| StringTapeError::AllocationError)?;

        let new_ptr = if let Some(old_ptr) = self.data {
            // Grow existing allocation
            let old_layout = Layout::array::<u8>(current_capacity).unwrap();
            unsafe {
                self.allocator
                    .grow(old_ptr.cast(), old_layout, new_layout)
                    .map_err(|_| StringTapeError::AllocationError)?
            }
        } else {
            // Initial allocation
            self.allocator
                .allocate(new_layout)
                .map_err(|_| StringTapeError::AllocationError)?
        };

        self.data = Some(NonNull::slice_from_raw_parts(new_ptr.cast(), new_capacity));
        Ok(())
    }

    fn grow_offsets(&mut self, new_capacity: usize) -> Result<(), StringTapeError> {
        let current_capacity = self.offsets_capacity();
        if new_capacity <= current_capacity {
            return Ok(());
        }

        let new_layout =
            Layout::array::<Offset>(new_capacity).map_err(|_| StringTapeError::AllocationError)?;

        let new_ptr = if let Some(old_ptr) = self.offsets {
            // Grow existing allocation
            let old_layout = Layout::array::<Offset>(current_capacity).unwrap();
            unsafe {
                self.allocator
                    .grow(old_ptr.cast(), old_layout, new_layout)
                    .map_err(|_| StringTapeError::AllocationError)?
            }
        } else {
            // Initial allocation with first offset = 0
            self.allocator
                .allocate_zeroed(new_layout)
                .map_err(|_| StringTapeError::AllocationError)?
        };

        self.offsets = Some(NonNull::slice_from_raw_parts(new_ptr.cast(), new_capacity));
        Ok(())
    }

    /// Appends bytes to the tape.
    ///
    /// # Errors
    ///
    /// - `OffsetOverflow` if data size exceeds offset type maximum
    /// - `AllocationError` if memory allocation fails
    ///
    /// # Example
    ///
    /// ```rust
    /// # use stringtape::{BytesTapeI32, StringTapeError};
    /// let mut tape = BytesTapeI32::new();
    /// tape.push(b"hello")?;
    /// assert_eq!(tape.len(), 1);
    /// # Ok::<(), StringTapeError>(())
    /// ```
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), StringTapeError> {
        let required_capacity = self
            .len_bytes
            .checked_add(bytes.len())
            .ok_or(StringTapeError::AllocationError)?;

        let current_data_capacity = self.data_capacity();
        if required_capacity > current_data_capacity {
            let new_capacity = (current_data_capacity * 2).max(required_capacity).max(64);
            self.grow_data(new_capacity)?;
        }

        let current_offsets_capacity = self.offsets_capacity();
        if self.len_items + 1 >= current_offsets_capacity {
            let new_capacity = (current_offsets_capacity * 2)
                .max(self.len_items + 2)
                .max(8);
            self.grow_offsets(new_capacity)?;
        }

        // Copy string data
        if let Some(data_ptr) = self.data {
            unsafe {
                ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    data_ptr.as_ptr().cast::<u8>().add(self.len_bytes),
                    bytes.len(),
                );
            }
        }

        self.len_bytes += bytes.len();
        self.len_items += 1;

        // Write new offset
        let offset = Offset::from_usize(self.len_bytes).ok_or(StringTapeError::OffsetOverflow)?;
        if let Some(offsets_ptr) = self.offsets {
            unsafe {
                ptr::write(
                    offsets_ptr.as_ptr().cast::<Offset>().add(self.len_items),
                    offset,
                );
            }
        }

        Ok(())
    }

    /// Returns a reference to the bytes at the given index, or `None` if out of bounds.
    ///
    /// This operation is O(1).
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        if index >= self.len_items {
            return None;
        }

        let (data_ptr, offsets_ptr) = match (self.data, self.offsets) {
            (Some(data), Some(offsets)) => (data, offsets),
            _ => return None,
        };

        unsafe {
            let offsets_ptr = offsets_ptr.as_ptr().cast::<Offset>();
            let start_offset = if index == 0 {
                0
            } else {
                ptr::read(offsets_ptr.add(index)).to_usize()
            };
            let end_offset = ptr::read(offsets_ptr.add(index + 1)).to_usize();

            Some(slice::from_raw_parts(
                data_ptr.as_ptr().cast::<u8>().add(start_offset),
                end_offset - start_offset,
            ))
        }
    }

    /// Returns the number of items in the tape.
    pub fn len(&self) -> usize {
        self.len_items
    }

    /// Returns `true` if the CharsTape contains no strings.
    pub fn is_empty(&self) -> bool {
        self.len_items == 0
    }

    /// Returns the total number of bytes used by string data.
    pub fn data_len(&self) -> usize {
        self.len_bytes
    }

    /// Returns the number of items currently stored (same as `len()`).
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.len_items
    }

    /// Returns the number of bytes allocated for string data.
    pub fn data_capacity(&self) -> usize {
        self.data.map(|ptr| ptr.len()).unwrap_or(0)
    }

    /// Returns the number of offset slots allocated.
    pub fn offsets_capacity(&self) -> usize {
        self.offsets.map(|ptr| ptr.len()).unwrap_or(0)
    }

    /// Removes all items from the tape, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.len_bytes = 0;
        self.len_items = 0;
        if let Some(offsets_ptr) = self.offsets {
            unsafe {
                ptr::write(offsets_ptr.as_ptr().cast::<Offset>(), Offset::default());
            }
        }
    }

    /// Keeps the first `len` items, drops the rest.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len_items {
            return;
        }

        self.len_items = len;
        self.len_bytes = if len == 0 {
            0
        } else if let Some(offsets_ptr) = self.offsets {
            unsafe { ptr::read(offsets_ptr.as_ptr().cast::<Offset>().add(len)).to_usize() }
        } else {
            0
        };
    }

    /// Appends all items from an iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use stringtape::{BytesTapeI32, StringTapeError};
    /// let mut tape = BytesTapeI32::new();
    /// tape.extend([b"hello".as_slice(), b"world".as_slice()])?;
    /// # Ok::<(), StringTapeError>(())
    /// ```
    pub fn extend<I>(&mut self, iter: I) -> Result<(), StringTapeError>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        for s in iter {
            self.push(s.as_ref())?;
        }
        Ok(())
    }

    /// Returns raw pointers for Apache Arrow compatibility.
    ///
    /// Returns `data_ptr`, `offsets_ptr`, `data_len`, `items_count`.
    ///
    /// # Safety
    ///
    /// Pointers valid only while tape is unmodified.
    pub fn as_raw_parts(&self) -> RawParts<Offset> {
        let data_ptr = self
            .data
            .map(|ptr| ptr.as_ptr().cast::<u8>() as *const u8)
            .unwrap_or(ptr::null());
        let offsets_ptr = self
            .offsets
            .map(|ptr| ptr.as_ptr().cast::<Offset>() as *const Offset)
            .unwrap_or(ptr::null());
        RawParts {
            data_ptr,
            offsets_ptr,
            data_len: self.len_bytes,
            items_count: self.len_items,
        }
    }

    /// Returns a slice view of the data buffer.
    ///
    /// This provides a cleaner interface for accessing the underlying data
    /// without dealing with raw pointers.
    pub fn data_slice(&self) -> &[u8] {
        if let Some(data_ptr) = self.data {
            unsafe { core::slice::from_raw_parts(data_ptr.as_ptr().cast::<u8>(), self.len_bytes) }
        } else {
            &[]
        }
    }

    /// Returns a slice view of the offsets buffer.
    ///
    /// This provides a cleaner interface for accessing the underlying offsets
    /// without dealing with raw pointers. The slice contains `len() + 1` elements.
    pub fn offsets_slice(&self) -> &[Offset] {
        if let Some(offsets_ptr) = self.offsets {
            unsafe {
                core::slice::from_raw_parts(
                    offsets_ptr.as_ptr().cast::<Offset>(),
                    self.len_items + 1,
                )
            }
        } else {
            &[]
        }
    }

    /// Returns a reference to the allocator used by this tape.
    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    /// Creates a view of the entire tape.
    pub fn view(&self) -> RawTapeView<'_, Offset> {
        RawTapeView::new(self, 0, self.len_items).unwrap_or(RawTapeView {
            data: &[],
            offsets: &[],
        })
    }

    /// Creates a subview of a continuous slice of this tape.
    pub fn subview(
        &self,
        start: usize,
        end: usize,
    ) -> Result<RawTapeView<'_, Offset>, StringTapeError> {
        RawTapeView::new(self, start, end)
    }
}

impl<Offset: OffsetType, A: Allocator> Drop for RawTape<Offset, A> {
    fn drop(&mut self) {
        if let Some(data_ptr) = self.data {
            let layout = Layout::array::<u8>(data_ptr.len()).unwrap();
            unsafe {
                self.allocator.deallocate(data_ptr.cast(), layout);
            }
        }
        if let Some(offsets_ptr) = self.offsets {
            let layout = Layout::array::<Offset>(offsets_ptr.len()).unwrap();
            unsafe {
                self.allocator.deallocate(offsets_ptr.cast(), layout);
            }
        }
    }
}

unsafe impl<Offset: OffsetType + Send, A: Allocator + Send> Send for RawTape<Offset, A> {}
unsafe impl<Offset: OffsetType + Sync, A: Allocator + Sync> Sync for RawTape<Offset, A> {}

// Index trait implementations for RawTape to support [i..n] syntax
impl<Offset: OffsetType, A: Allocator> Index<Range<usize>> for RawTape<Offset, A> {
    type Output = [u8];

    fn index(&self, range: Range<usize>) -> &Self::Output {
        let view = self
            .subview(range.start, range.end)
            .expect("range out of bounds");
        // Return the underlying data slice
        view.data
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeFrom<usize>> for RawTape<Offset, A> {
    type Output = [u8];

    fn index(&self, range: RangeFrom<usize>) -> &Self::Output {
        let view = self
            .subview(range.start, self.len_items)
            .expect("range out of bounds");
        view.data
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeTo<usize>> for RawTape<Offset, A> {
    type Output = [u8];

    fn index(&self, range: RangeTo<usize>) -> &Self::Output {
        let view = self.subview(0, range.end).expect("range out of bounds");
        view.data
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeFull> for RawTape<Offset, A> {
    type Output = [u8];

    fn index(&self, _range: RangeFull) -> &Self::Output {
        let view = self.view();
        view.data
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeInclusive<usize>> for RawTape<Offset, A> {
    type Output = [u8];

    fn index(&self, range: RangeInclusive<usize>) -> &Self::Output {
        let view = self
            .subview(*range.start(), range.end() + 1)
            .expect("range out of bounds");
        view.data
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeToInclusive<usize>> for RawTape<Offset, A> {
    type Output = [u8];

    fn index(&self, range: RangeToInclusive<usize>) -> &Self::Output {
        let view = self.subview(0, range.end + 1).expect("range out of bounds");
        view.data
    }
}

// ========================
// RawTapeView implementation
// ========================

impl<'a, Offset: OffsetType> RawTapeView<'a, Offset> {
    /// Creates a view into a slice of the RawTape from start to end (exclusive).
    pub(crate) fn new<A: Allocator>(
        tape: &'a RawTape<Offset, A>,
        start: usize,
        end: usize,
    ) -> Result<Self, StringTapeError> {
        if start > end || end > tape.len() {
            return Err(StringTapeError::IndexOutOfBounds);
        }

        let (data_ptr, offsets_ptr) = match (tape.data, tape.offsets) {
            (Some(data), Some(offsets)) => (data, offsets),
            _ => return Err(StringTapeError::IndexOutOfBounds),
        };

        // Keep the data pointer at the beginning of the parent tape to remain Arrow-compatible.
        // Offsets remain absolute (not normalized) and are sliced to the requested range.
        let data = unsafe { slice::from_raw_parts(data_ptr.as_ptr().cast::<u8>(), tape.len_bytes) };

        let offsets = unsafe {
            slice::from_raw_parts(
                offsets_ptr.as_ptr().cast::<Offset>().add(start),
                (end - start) + 1,
            )
        };

        Ok(Self { data, offsets })
    }

    /// Creates a zero-copy view from raw Arrow-compatible parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `data` contains valid bytes for the lifetime `'a`
    /// - `offsets` contains valid offsets with length `items_count + 1`
    /// - All offsets are within bounds of the data slice
    /// - For CharsTapeView, data must be valid UTF-8
    pub unsafe fn from_raw_parts(data: &'a [u8], offsets: &'a [Offset]) -> Self {
        Self { data, offsets }
    }

    /// Returns a reference to the bytes at the given index within this view.
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        if index >= self.len() {
            return None;
        }

        let start_offset = self.offsets[index].to_usize();
        let end_offset = self.offsets[index + 1].to_usize();

        Some(&self.data[start_offset..end_offset])
    }

    /// Returns the number of items in this view.
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Returns `true` if the view contains no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the total number of bytes in this view.
    pub fn data_len(&self) -> usize {
        // Span covered by this view
        self.offsets[self.offsets.len() - 1].to_usize() - self.offsets[0].to_usize()
    }

    /// Creates a sub-view of this view
    pub fn subview(
        &self,
        start: usize,
        end: usize,
    ) -> Result<RawTapeView<'a, Offset>, StringTapeError> {
        if start > end || end > self.len() {
            return Err(StringTapeError::IndexOutOfBounds);
        }

        Ok(RawTapeView {
            // Keep same data pointer, only narrow the offsets slice
            data: self.data,
            offsets: &self.offsets[start..=end],
        })
    }

    /// Returns the raw parts of the view for Apache Arrow compatibility.
    pub fn as_raw_parts(&self) -> RawParts<Offset> {
        // Expose an Arrow-compatible view: data_ptr remains at the tape base,
        // offsets are absolute into that buffer, and data_len reaches the last used byte.
        RawParts {
            data_ptr: self.data.as_ptr(),
            offsets_ptr: self.offsets.as_ptr(),
            data_len: self.offsets[self.offsets.len() - 1].to_usize(),
            items_count: self.len(),
        }
    }
}

impl<'a, Offset: OffsetType> Index<usize> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

// Index trait implementations for RawTapeView to support [i..n] syntax
impl<'a, Offset: OffsetType> Index<Range<usize>> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: Range<usize>) -> &Self::Output {
        let view = self
            .subview(range.start, range.end)
            .expect("range out of bounds");
        let start = view.offsets[0].to_usize();
        let end = view.offsets[view.offsets.len() - 1].to_usize();
        &view.data[start..end]
    }
}

impl<'a, Offset: OffsetType> Index<RangeFrom<usize>> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeFrom<usize>) -> &Self::Output {
        let view = self
            .subview(range.start, self.len())
            .expect("range out of bounds");
        let start = view.offsets[0].to_usize();
        let end = view.offsets[view.offsets.len() - 1].to_usize();
        &view.data[start..end]
    }
}

impl<'a, Offset: OffsetType> Index<RangeTo<usize>> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeTo<usize>) -> &Self::Output {
        let view = self.subview(0, range.end).expect("range out of bounds");
        let start = view.offsets[0].to_usize();
        let end = view.offsets[view.offsets.len() - 1].to_usize();
        &view.data[start..end]
    }
}

impl<'a, Offset: OffsetType> Index<RangeFull> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, _range: RangeFull) -> &Self::Output {
        let start = self.offsets[0].to_usize();
        let end = self.offsets[self.offsets.len() - 1].to_usize();
        &self.data[start..end]
    }
}

impl<'a, Offset: OffsetType> Index<RangeInclusive<usize>> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeInclusive<usize>) -> &Self::Output {
        let view = self
            .subview(*range.start(), range.end() + 1)
            .expect("range out of bounds");
        let start = view.offsets[0].to_usize();
        let end = view.offsets[view.offsets.len() - 1].to_usize();
        &view.data[start..end]
    }
}

impl<'a, Offset: OffsetType> Index<RangeToInclusive<usize>> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeToInclusive<usize>) -> &Self::Output {
        let view = self.subview(0, range.end + 1).expect("range out of bounds");
        let start = view.offsets[0].to_usize();
        let end = view.offsets[view.offsets.len() - 1].to_usize();
        &view.data[start..end]
    }
}

// ========================
// CharsTapeView implementation
// ========================

impl<'a, Offset: OffsetType> CharsTapeView<'a, Offset> {
    /// Creates a zero-copy CharsTapeView from raw Arrow StringArray parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `data` contains valid UTF-8 bytes for the lifetime `'a`
    /// - `offsets` contains valid offsets with appropriate length
    /// - All offsets are within bounds of the data slice
    pub unsafe fn from_raw_parts(data: &'a [u8], offsets: &'a [Offset]) -> Self {
        Self {
            inner: RawTapeView::from_raw_parts(data, offsets),
        }
    }

    /// Returns a reference to the string at the given index, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&str> {
        // Safe because CharsTapeView only comes from CharsTape which validates UTF-8
        self.inner
            .get(index)
            .map(|b| unsafe { core::str::from_utf8_unchecked(b) })
    }

    /// Returns the number of strings in this view.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the view contains no strings.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the total number of bytes in this view.
    pub fn data_len(&self) -> usize {
        self.inner.data_len()
    }

    /// Creates a sub-view of this view
    pub fn subview(
        &self,
        start: usize,
        end: usize,
    ) -> Result<CharsTapeView<'a, Offset>, StringTapeError> {
        Ok(CharsTapeView {
            inner: self.inner.subview(start, end)?,
        })
    }

    /// Returns the raw parts of the view for Apache Arrow compatibility.
    pub fn as_raw_parts(&self) -> RawParts<Offset> {
        self.inner.as_raw_parts()
    }
}

impl<'a, Offset: OffsetType> Index<usize> for CharsTapeView<'a, Offset> {
    type Output = str;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

// ========================
// BytesTapeView implementation
// ========================

impl<'a, Offset: OffsetType> BytesTapeView<'a, Offset> {
    /// Creates a zero-copy BytesTapeView from raw Arrow BinaryArray parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `data` contains valid bytes for the lifetime `'a`
    /// - `offsets` contains valid offsets with appropriate length
    /// - All offsets are within bounds of the data slice
    pub unsafe fn from_raw_parts(data: &'a [u8], offsets: &'a [Offset]) -> Self {
        Self {
            inner: RawTapeView::from_raw_parts(data, offsets),
        }
    }

    /// Returns a reference to the bytes at the given index, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        self.inner.get(index)
    }

    /// Returns the number of items in this view.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the view contains no items.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the total number of bytes in this view.
    pub fn data_len(&self) -> usize {
        self.inner.data_len()
    }

    /// Creates a sub-view of this view
    pub fn subview(
        &self,
        start: usize,
        end: usize,
    ) -> Result<BytesTapeView<'a, Offset>, StringTapeError> {
        Ok(BytesTapeView {
            inner: self.inner.subview(start, end)?,
        })
    }

    /// Returns the raw parts of the view for Apache Arrow compatibility.
    pub fn as_raw_parts(&self) -> RawParts<Offset> {
        self.inner.as_raw_parts()
    }
}

impl<'a, Offset: OffsetType> Index<usize> for BytesTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

// ========================
// CharsTape (UTF-8 view)
// ========================

impl<Offset: OffsetType, A: Allocator> CharsTape<Offset, A> {
    /// Creates a new, empty CharsTape with the global allocator.
    pub fn new() -> CharsTape<Offset, Global> {
        CharsTape {
            inner: RawTape::<Offset, Global>::new(),
        }
    }

    /// Creates a new, empty CharsTape with a custom allocator.
    pub fn new_in(allocator: A) -> Self {
        Self {
            inner: RawTape::<Offset, A>::new_in(allocator),
        }
    }

    /// Creates a new CharsTape with pre-allocated capacity using the global allocator.
    pub fn with_capacity(
        data_capacity: usize,
        strings_capacity: usize,
    ) -> Result<CharsTape<Offset, Global>, StringTapeError> {
        Ok(CharsTape {
            inner: RawTape::<Offset, Global>::with_capacity(data_capacity, strings_capacity)?,
        })
    }

    /// Creates a new CharsTape with pre-allocated capacity and a custom allocator.
    pub fn with_capacity_in(
        data_capacity: usize,
        strings_capacity: usize,
        allocator: A,
    ) -> Result<Self, StringTapeError> {
        Ok(Self {
            inner: RawTape::<Offset, A>::with_capacity_in(
                data_capacity,
                strings_capacity,
                allocator,
            )?,
        })
    }

    /// Adds a string to the end of the CharsTape.
    pub fn push(&mut self, s: &str) -> Result<(), StringTapeError> {
        self.inner.push(s.as_bytes())
    }

    /// Returns a reference to the string at the given index, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&str> {
        // Safe because CharsTape only accepts &str pushes.
        self.inner
            .get(index)
            .map(|b| unsafe { core::str::from_utf8_unchecked(b) })
    }

    /// Returns the number of strings in the CharsTape.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the CharsTape contains no strings.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the total number of bytes used by string data.
    pub fn data_len(&self) -> usize {
        self.inner.data_len()
    }

    /// Returns the number of strings currently stored (same as `len()`).
    pub fn capacity(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of bytes allocated for string data.
    pub fn data_capacity(&self) -> usize {
        self.inner.data_capacity()
    }

    /// Returns the number of offset slots allocated.
    pub fn offsets_capacity(&self) -> usize {
        self.inner.offsets_capacity()
    }

    /// Removes all strings from the CharsTape, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Shortens the CharsTape, keeping the first `len` strings and dropping the rest.
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(len)
    }

    /// Extends the CharsTape with the contents of an iterator.
    pub fn extend<I>(&mut self, iter: I) -> Result<(), StringTapeError>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        for s in iter {
            self.push(s.as_ref())?;
        }
        Ok(())
    }

    /// Returns the raw parts of the CharsTape for Apache Arrow compatibility.
    pub fn as_raw_parts(&self) -> RawParts<Offset> {
        self.inner.as_raw_parts()
    }

    /// Returns a slice view of the data buffer.
    pub fn data_slice(&self) -> &[u8] {
        self.inner.data_slice()
    }

    /// Returns a slice view of the offsets buffer.
    pub fn offsets_slice(&self) -> &[Offset] {
        self.inner.offsets_slice()
    }

    pub fn iter(&self) -> CharsTapeIter<'_, Offset, A> {
        CharsTapeIter {
            tape: self,
            index: 0,
        }
    }

    /// Returns a reference to the allocator used by this CharsTape.
    pub fn allocator(&self) -> &A {
        self.inner.allocator()
    }

    /// Creates a view of the entire CharsTape.
    pub fn view(&self) -> CharsTapeView<'_, Offset> {
        CharsTapeView {
            inner: self.inner.view(),
        }
    }

    /// Creates a subview of a continuous slice of this CharsTape.
    pub fn subview(
        &self,
        start: usize,
        end: usize,
    ) -> Result<CharsTapeView<'_, Offset>, StringTapeError> {
        Ok(CharsTapeView {
            inner: self.inner.subview(start, end)?,
        })
    }

    /// Creates a CharsCows view of the tape.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use stringtape::{CharsTapeI32, CharsCows, StringTapeError};
    /// # use std::borrow::Cow;
    /// let mut tape = CharsTapeI32::new();
    /// tape.extend(["apple", "banana", "cherry"])?;
    ///
    /// let cows: CharsCows<i32, u16> = tape.as_reorderable()?;
    /// assert_eq!(cows.get(0), Some("apple"));
    /// # Ok::<(), StringTapeError>(())
    /// ```
    pub fn as_reorderable<Length: LengthType>(
        &self,
    ) -> Result<CharsCows<'_, Offset, Length>, StringTapeError> {
        CharsCows::from_iter_and_data(self, Cow::Borrowed(self.data_slice()))
    }

    /// Returns data and offsets slices for zero-copy Arrow conversion.
    pub fn arrow_slices(&self) -> (&[u8], &[Offset]) {
        (self.data_slice(), self.offsets_slice())
    }
}

impl<Offset: OffsetType, A: Allocator> Drop for CharsTape<Offset, A> {
    fn drop(&mut self) {
        // Explicit drop of inner to run RawTape's Drop
        // (redundant but keeps intent clear)
    }
}

unsafe impl<Offset: OffsetType + Send, A: Allocator + Send> Send for CharsTape<Offset, A> {}
unsafe impl<Offset: OffsetType + Sync, A: Allocator + Sync> Sync for CharsTape<Offset, A> {}

pub struct CharsTapeIter<'a, Offset: OffsetType, A: Allocator> {
    tape: &'a CharsTape<Offset, A>,
    index: usize,
}

impl<'a, Offset: OffsetType, A: Allocator> Iterator for CharsTapeIter<'a, Offset, A> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.tape.get(self.index);
        if result.is_some() {
            self.index += 1;
        }
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.tape.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, Offset: OffsetType, A: Allocator> ExactSizeIterator for CharsTapeIter<'a, Offset, A> {}

impl<Offset: OffsetType> FromIterator<String> for CharsTape<Offset, Global> {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let mut tape = CharsTape::<Offset, Global>::new();
        for s in iter {
            tape.push(&s)
                .expect("Failed to build CharsTape from iterator");
        }
        tape
    }
}

impl<'a, Offset: OffsetType> FromIterator<&'a str> for CharsTape<Offset, Global> {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        let mut tape = CharsTape::<Offset, Global>::new();
        for s in iter {
            tape.push(s)
                .expect("Failed to build CharsTape from iterator");
        }
        tape
    }
}

impl<Offset: OffsetType, A: Allocator> Index<usize> for CharsTape<Offset, A> {
    type Output = str;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<'a, Offset: OffsetType, A: Allocator> IntoIterator for &'a CharsTape<Offset, A> {
    type Item = &'a str;
    type IntoIter = CharsTapeIter<'a, Offset, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ======================
// BytesTape (bytes view)
// ======================

impl<Offset: OffsetType, A: Allocator> BytesTape<Offset, A> {
    /// Creates a new, empty BytesTape with the global allocator.
    pub fn new() -> BytesTape<Offset, Global> {
        BytesTape {
            inner: RawTape::<Offset, Global>::new(),
        }
    }

    /// Creates a new, empty BytesTape with a custom allocator.
    pub fn new_in(allocator: A) -> Self {
        Self {
            inner: RawTape::<Offset, A>::new_in(allocator),
        }
    }

    /// Creates a new BytesTape with pre-allocated capacity using the global allocator.
    pub fn with_capacity(
        data_capacity: usize,
        items_capacity: usize,
    ) -> Result<BytesTape<Offset, Global>, StringTapeError> {
        Ok(BytesTape {
            inner: RawTape::<Offset, Global>::with_capacity(data_capacity, items_capacity)?,
        })
    }

    /// Creates a new BytesTape with pre-allocated capacity and a custom allocator.
    pub fn with_capacity_in(
        data_capacity: usize,
        items_capacity: usize,
        allocator: A,
    ) -> Result<Self, StringTapeError> {
        Ok(Self {
            inner: RawTape::<Offset, A>::with_capacity_in(
                data_capacity,
                items_capacity,
                allocator,
            )?,
        })
    }

    /// Adds bytes to the end of the tape.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), StringTapeError> {
        self.inner.push(bytes)
    }

    /// Returns a reference to the bytes at the given index, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        self.inner.get(index)
    }

    /// Returns the number of items in the tape.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the tape contains no items.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the total number of bytes used by data.
    pub fn data_len(&self) -> usize {
        self.inner.data_len()
    }

    /// Returns the number of bytes allocated for data.
    pub fn data_capacity(&self) -> usize {
        self.inner.data_capacity()
    }

    /// Returns the number of offset slots allocated.
    pub fn offsets_capacity(&self) -> usize {
        self.inner.offsets_capacity()
    }

    /// Removes all items from the tape, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Shortens the tape, keeping the first `len` items and dropping the rest.
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(len)
    }

    /// Extends the tape with the contents of an iterator of bytes.
    pub fn extend<I>(&mut self, iter: I) -> Result<(), StringTapeError>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        self.inner.extend(iter)
    }

    /// Returns the raw parts of the tape for Apache Arrow compatibility.
    pub fn as_raw_parts(&self) -> RawParts<Offset> {
        self.inner.as_raw_parts()
    }

    /// Returns a slice view of the data buffer.
    pub fn data_slice(&self) -> &[u8] {
        self.inner.data_slice()
    }

    /// Returns a slice view of the offsets buffer.
    pub fn offsets_slice(&self) -> &[Offset] {
        self.inner.offsets_slice()
    }

    /// Returns a reference to the allocator used by this BytesTape.
    pub fn allocator(&self) -> &A {
        self.inner.allocator()
    }

    /// Creates a view of the entire BytesTape.
    pub fn view(&self) -> BytesTapeView<'_, Offset> {
        BytesTapeView {
            inner: self.inner.view(),
        }
    }

    /// Returns an iterator over the byte cows.
    pub fn iter(&self) -> BytesTapeIter<'_, Offset, A> {
        BytesTapeIter {
            tape: self,
            index: 0,
        }
    }

    /// Creates a subview of a continuous slice of this BytesTape.
    pub fn subview(
        &self,
        start: usize,
        end: usize,
    ) -> Result<BytesTapeView<'_, Offset>, StringTapeError> {
        Ok(BytesTapeView {
            inner: self.inner.subview(start, end)?,
        })
    }

    /// Creates a BytesCows view of the tape.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use stringtape::{BytesTapeI32, BytesCows, StringTapeError};
    /// # use std::borrow::Cow;
    /// let mut tape = BytesTapeI32::new();
    /// tape.push(&[1, 2, 3])?;
    /// tape.push(&[4, 5, 6])?;
    /// tape.push(&[7, 8, 9])?;
    ///
    /// let cows: BytesCows<i32, u16> = tape.as_reorderable()?;
    /// assert_eq!(cows.get(0), Some(&[1, 2, 3][..]));
    /// # Ok::<(), StringTapeError>(())
    /// ```
    pub fn as_reorderable<Length: LengthType>(
        &self,
    ) -> Result<BytesCows<'_, Offset, Length>, StringTapeError> {
        BytesCows::from_iter_and_data(self, Cow::Borrowed(self.data_slice()))
    }

    /// Returns data and offsets slices for zero-copy Arrow conversion.
    pub fn arrow_slices(&self) -> (&[u8], &[Offset]) {
        (self.data_slice(), self.offsets_slice())
    }
}

impl<Offset: OffsetType, A: Allocator> Index<usize> for BytesTape<Offset, A> {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

pub struct BytesTapeIter<'a, Offset: OffsetType, A: Allocator> {
    tape: &'a BytesTape<Offset, A>,
    index: usize,
}

impl<'a, Offset: OffsetType, A: Allocator> Iterator for BytesTapeIter<'a, Offset, A> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.tape.get(self.index);
        if result.is_some() {
            self.index += 1;
        }
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.tape.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, Offset: OffsetType, A: Allocator> ExactSizeIterator for BytesTapeIter<'a, Offset, A> {}

impl<'a, Offset: OffsetType, A: Allocator> IntoIterator for &'a BytesTape<Offset, A> {
    type Item = &'a [u8];
    type IntoIter = BytesTapeIter<'a, Offset, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Signed (Arrow-compatible) aliases
pub type CharsTapeI32 = CharsTape<i32, Global>;
pub type CharsTapeI64 = CharsTape<i64, Global>;
pub type BytesTapeI32 = BytesTape<i32, Global>;
pub type BytesTapeI64 = BytesTape<i64, Global>;

pub type CharsTapeViewI32<'a> = CharsTapeView<'a, i32>;
pub type CharsTapeViewI64<'a> = CharsTapeView<'a, i64>;
pub type BytesTapeViewI32<'a> = BytesTapeView<'a, i32>;
pub type BytesTapeViewI64<'a> = BytesTapeView<'a, i64>;

// Unsigned aliases (not zero-copy with Arrow)
pub type CharsTapeU32 = CharsTape<u32, Global>;
pub type CharsTapeU64 = CharsTape<u64, Global>;
pub type BytesTapeU16 = BytesTape<u16, Global>;
pub type BytesTapeU32 = BytesTape<u32, Global>;
pub type BytesTapeU64 = BytesTape<u64, Global>;

pub type CharsTapeViewU32<'a> = CharsTapeView<'a, u32>;
pub type CharsTapeViewU64<'a> = CharsTapeView<'a, u64>;
pub type BytesTapeViewU16<'a> = BytesTapeView<'a, u16>;
pub type BytesTapeViewU32<'a> = BytesTapeView<'a, u32>;
pub type BytesTapeViewU64<'a> = BytesTapeView<'a, u64>;

// Conversion implementations between BytesTape and CharsTape
impl<Offset: OffsetType, A: Allocator> TryFrom<BytesTape<Offset, A>> for CharsTape<Offset, A> {
    type Error = StringTapeError;

    fn try_from(bytes_tape: BytesTape<Offset, A>) -> Result<Self, Self::Error> {
        // Validate that all byte sequences are valid UTF-8
        for i in 0..bytes_tape.len() {
            if let Err(e) = core::str::from_utf8(&bytes_tape[i]) {
                return Err(StringTapeError::Utf8Error(e));
            }
        }

        // Since validation passed, we can safely convert
        // We need to take ownership of the inner RawTape without dropping BytesTape
        let inner = unsafe {
            // Take ownership of the inner RawTape
            let inner = core::ptr::read(&bytes_tape.inner);
            // Prevent BytesTape's destructor from running
            core::mem::forget(bytes_tape);
            inner
        };
        Ok(CharsTape { inner })
    }
}

impl<Offset: OffsetType, A: Allocator> From<CharsTape<Offset, A>> for BytesTape<Offset, A> {
    fn from(chars_tape: CharsTape<Offset, A>) -> Self {
        // CharsTape already contains valid UTF-8, so conversion to BytesTape is infallible
        // We need to take ownership of the inner RawTape without dropping CharsTape
        let inner = unsafe {
            // Take ownership of the inner RawTape
            let inner = core::ptr::read(&chars_tape.inner);
            // Prevent CharsTape's destructor from running
            core::mem::forget(chars_tape);
            inner
        };
        BytesTape { inner }
    }
}

impl<Offset: OffsetType, A: Allocator> BytesTape<Offset, A> {
    pub fn try_into_chars_tape(self) -> Result<CharsTape<Offset, A>, StringTapeError> {
        self.try_into()
    }
}

impl<Offset: OffsetType, A: Allocator> CharsTape<Offset, A> {
    pub fn into_bytes_tape(self) -> BytesTape<Offset, A> {
        self.into()
    }
}

// Conversion implementations between BytesTapeView and CharsTapeView
impl<'a, Offset: OffsetType> TryFrom<BytesTapeView<'a, Offset>> for CharsTapeView<'a, Offset> {
    type Error = StringTapeError;

    fn try_from(bytes_view: BytesTapeView<'a, Offset>) -> Result<Self, Self::Error> {
        // Validate that all byte sequences are valid UTF-8
        for i in 0..bytes_view.len() {
            let bytes = bytes_view.get(i).ok_or(StringTapeError::IndexOutOfBounds)?;
            if core::str::from_utf8(bytes).is_err() {
                return Err(StringTapeError::Utf8Error(
                    core::str::from_utf8(bytes).unwrap_err(),
                ));
            }
        }

        // Since validation passed, construct a CharsTapeView over the same inner view
        Ok(CharsTapeView {
            inner: bytes_view.inner,
        })
    }
}

impl<'a, Offset: OffsetType> From<CharsTapeView<'a, Offset>> for BytesTapeView<'a, Offset> {
    fn from(chars_view: CharsTapeView<'a, Offset>) -> Self {
        // UTF-8 bytes can always be viewed as bytes
        BytesTapeView {
            inner: chars_view.inner,
        }
    }
}

impl<'a, Offset: OffsetType> BytesTapeView<'a, Offset> {
    pub fn try_into_chars_view(self) -> Result<CharsTapeView<'a, Offset>, StringTapeError> {
        self.try_into()
    }
}

impl<'a, Offset: OffsetType> CharsTapeView<'a, Offset> {
    pub fn into_bytes_view(self) -> BytesTapeView<'a, Offset> {
        self.into()
    }
}

impl<Offset: OffsetType> Default for CharsTape<Offset, Global> {
    fn default() -> Self {
        Self::new()
    }
}

// ========================
// CharsCows - Compact slice collection with configurable offset/length types
// ========================

/// Packed entry struct to eliminate padding overhead between offset and length.
///
/// For example, `(u64, u8)` tuple uses 16 bytes (8 + 8 padding), but
/// `PackedEntry<u64, u8>` uses only 9 bytes (8 + 1).
#[repr(C, packed(1))]
#[derive(Copy, Clone, Debug)]
struct PackedEntry<Offset, Length> {
    offset: Offset,
    length: Length,
}

/// A memory-efficient collection of string slices with configurable offset and length types.
///
/// `CharsCows` stores references to string slices in a shared data buffer using compact
/// (offset, length) pairs. This is ideal for large datasets where you want to reference
/// substrings without duplicating the underlying data.
///
/// # Type Parameters
///
/// * `Offset` - The offset type (u8, u16, u32, u64) determining maximum data size
/// * `Length` - The length type (u8, u16, u32, u64) determining maximum slice size
///
/// # Memory Efficiency
///
/// For 500M words (8 bytes avg) from a 4GB file:
/// - `Vec<String>`: ~66 GB (24 bytes per String + heap overhead)
/// - `CharsCows<u32, u16>`: ~7 GB (4+2 bytes per entry + shared 4GB data)
///
/// # Examples
///
/// ```rust
/// use stringtape::{CharsCows, StringTapeError};
/// use std::borrow::Cow;
///
/// let data = "hello world foo bar";
/// let cows = CharsCows::<u32, u16>::from_iter_and_data(
///     data.split_whitespace(),
///     Cow::Borrowed(data.as_bytes())
/// )?;
///
/// assert_eq!(cows.len(), 4);
/// assert_eq!(cows.get(0), Some("hello"));
/// assert_eq!(cows.get(3), Some("bar"));
/// # Ok::<(), StringTapeError>(())
/// ```
#[derive(Debug, Clone)]
pub struct CharsCows<'a, Offset: OffsetType = u32, Length: LengthType = u16> {
    data: Cow<'a, [u8]>,
    entries: Vec<PackedEntry<Offset, Length>>,
}

/// A memory-efficient collection of byte slices with configurable offset and length types.
///
/// Similar to `CharsCows` but for arbitrary binary data without UTF-8 validation.
#[derive(Debug, Clone)]
pub struct BytesCows<'a, Offset: OffsetType = u32, Length: LengthType = u16> {
    data: Cow<'a, [u8]>,
    entries: Vec<PackedEntry<Offset, Length>>,
}

impl<'a, Offset: OffsetType, Length: LengthType> CharsCows<'a, Offset, Length> {
    /// Creates a CharsCows from an iterator of string slices and shared data buffer.
    ///
    /// The slices must be subslices of the data buffer. Offsets and lengths are inferred
    /// from the slice pointers.
    ///
    /// # Arguments
    ///
    /// * `iter` - Iterator yielding string slices that are subslices of `data`
    /// * `data` - Cow-wrapped data buffer (borrowed or owned)
    ///
    /// # Errors
    ///
    /// - `OffsetOverflow` if offset/length exceeds type maximum
    /// - `IndexOutOfBounds` if slice not within data buffer
    ///
    /// # Example
    ///
    /// ```rust
    /// # use stringtape::{CharsCowsU32U8, StringTapeError};
    /// # use std::borrow::Cow;
    /// let data = "hello world";
    /// let cows = CharsCowsU32U8::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// )?;
    /// # Ok::<(), StringTapeError>(())
    /// ```
    pub fn from_iter_and_data<I>(iter: I, data: Cow<'a, [u8]>) -> Result<Self, StringTapeError>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let data_ptr = data.as_ptr() as usize;
        let data_end = data_ptr + data.len();
        let mut entries = Vec::new();

        for s in iter {
            let s_ref = s.as_ref();
            let s_bytes = s_ref.as_bytes();
            let s_ptr = s_bytes.as_ptr() as usize;

            // Calculate offset from base pointer
            if s_ptr < data_ptr || s_ptr > data_end {
                return Err(StringTapeError::IndexOutOfBounds);
            }

            let offset = s_ptr - data_ptr;
            let length = s_bytes.len();

            if offset + length > data.len() {
                return Err(StringTapeError::IndexOutOfBounds);
            }

            let offset_typed = Offset::from_usize(offset).ok_or(StringTapeError::OffsetOverflow)?;
            let length_typed = Length::from_usize(length).ok_or(StringTapeError::OffsetOverflow)?;

            entries.push(PackedEntry {
                offset: offset_typed,
                length: length_typed,
            });
        }

        Ok(Self { data, entries })
    }

    /// Returns a reference to the string at the given index, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&'_ str> {
        self.entries.get(index).map(|entry| {
            // Must copy fields from packed struct (can't take references)
            let start = entry.offset.to_usize();
            let len = entry.length.to_usize();
            // Safety: UTF-8 validated during construction
            // The lifetime of the returned &str is tied to self.data, not self
            unsafe { core::str::from_utf8_unchecked(&self.data[start..start + len]) }
        })
    }

    /// Returns the number of slices in the collection.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the collection contains no cows.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an iterator over the string cows.
    pub fn iter(&self) -> CharsCowsIter<'_, Offset, Length> {
        CharsCowsIter {
            slices: self,
            index: 0,
        }
    }

    /// Returns a reference to the underlying data buffer.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Sorts the slices in-place using the default string comparison.
    ///
    /// This is a stable sort that preserves the order of equal elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsU32U8;
    /// use std::borrow::Cow;
    ///
    /// let data = "zebra apple banana";
    /// let mut cows = CharsCowsU32U8::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// cows.sort();
    /// let sorted: Vec<&str> = cows.iter().collect();
    /// assert_eq!(sorted, vec!["apple", "banana", "zebra"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn sort(&mut self)
    where
        Offset: OffsetType,
        Length: LengthType,
    {
        self.entries.sort_by(|a, b| {
            let str_a = {
                let start = a.offset.to_usize();
                let len = a.length.to_usize();
                unsafe { core::str::from_utf8_unchecked(&self.data[start..start + len]) }
            };
            let str_b = {
                let start = b.offset.to_usize();
                let len = b.length.to_usize();
                unsafe { core::str::from_utf8_unchecked(&self.data[start..start + len]) }
            };
            str_a.cmp(str_b)
        });
    }

    /// Sorts the slices in-place using an unstable sorting algorithm.
    ///
    /// This is faster than stable sort but may not preserve the order of equal elements.
    pub fn sort_unstable(&mut self)
    where
        Offset: OffsetType,
        Length: LengthType,
    {
        self.entries.sort_unstable_by(|a, b| {
            let str_a = {
                let start = a.offset.to_usize();
                let len = a.length.to_usize();
                unsafe { core::str::from_utf8_unchecked(&self.data[start..start + len]) }
            };
            let str_b = {
                let start = b.offset.to_usize();
                let len = b.length.to_usize();
                unsafe { core::str::from_utf8_unchecked(&self.data[start..start + len]) }
            };
            str_a.cmp(str_b)
        });
    }

    /// Sorts the slices in-place using a custom comparison function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsU32U8;
    /// use std::borrow::Cow;
    ///
    /// let data = "aaa bb c";
    /// let mut cows = CharsCowsU32U8::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// // Sort by length, then alphabetically
    /// cows.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
    /// let sorted: Vec<&str> = cows.iter().collect();
    /// assert_eq!(sorted, vec!["c", "bb", "aaa"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn sort_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&str, &str) -> core::cmp::Ordering,
        Offset: OffsetType,
        Length: LengthType,
    {
        self.entries.sort_by(|a, b| {
            let str_a = {
                let start = a.offset.to_usize();
                let len = a.length.to_usize();
                unsafe { core::str::from_utf8_unchecked(&self.data[start..start + len]) }
            };
            let str_b = {
                let start = b.offset.to_usize();
                let len = b.length.to_usize();
                unsafe { core::str::from_utf8_unchecked(&self.data[start..start + len]) }
            };
            compare(str_a, str_b)
        });
    }

    /// Sorts the slices in-place using a key extraction function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsU32U8;
    /// use std::borrow::Cow;
    ///
    /// let data = "aaa bb c";
    /// let mut cows = CharsCowsU32U8::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// // Sort by string length
    /// cows.sort_by_key(|s| s.len());
    /// let sorted: Vec<&str> = cows.iter().collect();
    /// assert_eq!(sorted, vec!["c", "bb", "aaa"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn sort_by_key<K, F>(&mut self, mut f: F)
    where
        F: FnMut(&str) -> K,
        K: Ord,
        Offset: OffsetType,
        Length: LengthType,
    {
        self.entries.sort_by_key(|entry| {
            let start = entry.offset.to_usize();
            let len = entry.length.to_usize();
            let s = unsafe { core::str::from_utf8_unchecked(&self.data[start..start + len]) };
            f(s)
        });
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> BytesCows<'a, Offset, Length> {
    /// Creates BytesCows from iterator of byte slices and shared data buffer.
    ///
    /// Slices must be subslices of the data buffer. Offsets and lengths are inferred
    /// from slice pointers.
    pub fn from_iter_and_data<I>(iter: I, data: Cow<'a, [u8]>) -> Result<Self, StringTapeError>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        let data_ptr = data.as_ptr() as usize;
        let data_end = data_ptr + data.len();
        let mut entries = Vec::new();

        for b in iter {
            let b_ref = b.as_ref();
            let b_ptr = b_ref.as_ptr() as usize;

            if b_ptr < data_ptr || b_ptr > data_end {
                return Err(StringTapeError::IndexOutOfBounds);
            }

            let offset = b_ptr - data_ptr;
            let length = b_ref.len();

            if offset + length > data.len() {
                return Err(StringTapeError::IndexOutOfBounds);
            }

            let offset_typed = Offset::from_usize(offset).ok_or(StringTapeError::OffsetOverflow)?;
            let length_typed = Length::from_usize(length).ok_or(StringTapeError::OffsetOverflow)?;

            entries.push(PackedEntry {
                offset: offset_typed,
                length: length_typed,
            });
        }

        Ok(Self { data, entries })
    }

    /// Creates BytesCows from iterator of (offset, length) pairs and data buffer.
    pub fn from_offsets_and_data<I>(iter: I, data: Cow<'a, [u8]>) -> Result<Self, StringTapeError>
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        let mut entries = Vec::new();

        for (offset, length) in iter {
            let offset_typed = Offset::from_usize(offset).ok_or(StringTapeError::OffsetOverflow)?;
            let length_typed = Length::from_usize(length).ok_or(StringTapeError::OffsetOverflow)?;

            let end = offset
                .checked_add(length)
                .ok_or(StringTapeError::OffsetOverflow)?;
            if end > data.len() {
                return Err(StringTapeError::IndexOutOfBounds);
            }

            entries.push(PackedEntry {
                offset: offset_typed,
                length: length_typed,
            });
        }

        Ok(Self { data, entries })
    }

    /// Returns a reference to the bytes at the given index, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        self.entries.get(index).map(|entry| {
            let start = entry.offset.to_usize();
            let len = entry.length.to_usize();
            &self.data[start..start + len]
        })
    }

    /// Returns the number of slices in the collection.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the collection contains no cows.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an iterator over the byte cows.
    pub fn iter(&self) -> BytesCowsIter<'_, Offset, Length> {
        BytesCowsIter {
            slices: self,
            index: 0,
        }
    }

    /// Returns a reference to the underlying data buffer.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns a zero-copy view of this `BytesCows` as a `CharsCows` if all slices are valid UTF-8.
    ///
    /// This validates that all byte slices contain valid UTF-8, then reinterprets the collection
    /// as strings without copying or moving any data.
    ///
    /// # Errors
    ///
    /// Returns `StringTapeError::Utf8Error` if any slice contains invalid UTF-8.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::BytesCowsU32U8;
    /// use std::borrow::Cow;
    ///
    /// let data = b"hello world";
    /// let bytes = BytesCowsU32U8::from_iter_and_data(
    ///     data.split(|&b| b == b' '),
    ///     Cow::Borrowed(&data[..])
    /// ).unwrap();
    ///
    /// let chars = bytes.as_chars().unwrap();
    /// assert_eq!(chars.get(0), Some("hello"));
    /// assert_eq!(chars.get(1), Some("world"));
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn as_chars(&self) -> Result<CharsCows<'_, Offset, Length>, StringTapeError> {
        // Validate that all slices contain valid UTF-8
        for i in 0..self.len() {
            let slice = self.get(i).ok_or(StringTapeError::IndexOutOfBounds)?;
            core::str::from_utf8(slice).map_err(StringTapeError::Utf8Error)?;
        }

        // Safety: All slices validated as UTF-8
        Ok(CharsCows {
            data: Cow::Borrowed(self.data.as_ref()),
            entries: self.entries.clone(),
        })
    }
}

pub struct CharsCowsIter<'a, Offset: OffsetType, Length: LengthType> {
    slices: &'a CharsCows<'a, Offset, Length>,
    index: usize,
}

impl<'a, Offset: OffsetType, Length: LengthType> Iterator for CharsCowsIter<'a, Offset, Length> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.slices.get(self.index);
        if result.is_some() {
            self.index += 1;
        }
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slices.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> ExactSizeIterator
    for CharsCowsIter<'a, Offset, Length>
{
}

pub struct BytesCowsIter<'a, Offset: OffsetType, Length: LengthType> {
    slices: &'a BytesCows<'a, Offset, Length>,
    index: usize,
}

impl<'a, Offset: OffsetType, Length: LengthType> Iterator for BytesCowsIter<'a, Offset, Length> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.slices.get(self.index);
        if result.is_some() {
            self.index += 1;
        }
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slices.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> ExactSizeIterator
    for BytesCowsIter<'a, Offset, Length>
{
}

impl<'a, Offset: OffsetType, Length: LengthType> Index<usize> for CharsCows<'a, Offset, Length> {
    type Output = str;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> Index<usize> for BytesCows<'a, Offset, Length> {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> IntoIterator
    for &'a CharsCows<'a, Offset, Length>
{
    type Item = &'a str;
    type IntoIter = CharsCowsIter<'a, Offset, Length>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> IntoIterator
    for &'a BytesCows<'a, Offset, Length>
{
    type Item = &'a [u8];
    type IntoIter = BytesCowsIter<'a, Offset, Length>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Conversion implementations between BytesCows and CharsCows
impl<'a, Offset: OffsetType, Length: LengthType> TryFrom<BytesCows<'a, Offset, Length>>
    for CharsCows<'a, Offset, Length>
{
    type Error = StringTapeError;

    fn try_from(bytes_slices: BytesCows<'a, Offset, Length>) -> Result<Self, Self::Error> {
        // Validate that all slices contain valid UTF-8
        for i in 0..bytes_slices.len() {
            let slice = bytes_slices
                .get(i)
                .ok_or(StringTapeError::IndexOutOfBounds)?;
            core::str::from_utf8(slice).map_err(StringTapeError::Utf8Error)?;
        }

        // Safety: All slices validated as UTF-8
        Ok(CharsCows {
            data: bytes_slices.data,
            entries: bytes_slices.entries,
        })
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> From<CharsCows<'a, Offset, Length>>
    for BytesCows<'a, Offset, Length>
{
    fn from(chars_slices: CharsCows<'a, Offset, Length>) -> Self {
        // CharsCows contains valid UTF-8, so conversion to BytesCows is infallible
        BytesCows {
            data: chars_slices.data,
            entries: chars_slices.entries,
        }
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> BytesCows<'a, Offset, Length> {
    pub fn try_into_chars_slices(self) -> Result<CharsCows<'a, Offset, Length>, StringTapeError> {
        self.try_into()
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> CharsCows<'a, Offset, Length> {
    pub fn into_bytes_slices(self) -> BytesCows<'a, Offset, Length> {
        self.into()
    }

    /// Returns a zero-copy view of this `CharsCows` as a `BytesCows`.
    ///
    /// This is a no-cost operation that reinterprets the string collection as bytes
    /// without copying or moving any data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsU32U8;
    /// use std::borrow::Cow;
    ///
    /// let data = "hello world";
    /// let cows = CharsCowsU32U8::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// let bytes = cows.as_bytes();
    /// assert_eq!(bytes.get(0), Some(&b"hello"[..]));
    /// assert_eq!(bytes.get(1), Some(&b"world"[..]));
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn as_bytes(&self) -> BytesCows<'_, Offset, Length> {
        BytesCows {
            data: Cow::Borrowed(self.data.as_ref()),
            entries: self.entries.clone(),
        }
    }
}

// Type aliases for common configurations
pub type CharsCowsU32U16<'a> = CharsCows<'a, u32, u16>;
pub type CharsCowsU32U8<'a> = CharsCows<'a, u32, u8>;
pub type CharsCowsU16U8<'a> = CharsCows<'a, u16, u8>;
pub type CharsCowsU64U32<'a> = CharsCows<'a, u64, u32>;

pub type BytesCowsU32U16<'a> = BytesCows<'a, u32, u16>;
pub type BytesCowsU32U8<'a> = BytesCows<'a, u32, u8>;
pub type BytesCowsU16U8<'a> = BytesCows<'a, u16, u8>;
pub type BytesCowsU64U32<'a> = BytesCows<'a, u64, u32>;

// ========================
// Auto-selecting CharsCows
// ========================

/// Automatically selects the most memory-efficient CharsCows type based on data size.
///
/// Returns an enum that can hold any combination of offset/length types.
pub enum CharsCowsAuto<'a> {
    U32U8(CharsCows<'a, u32, u8>),
    U32U16(CharsCows<'a, u32, u16>),
    U32U32(CharsCows<'a, u32, u32>),
    U64U8(CharsCows<'a, u64, u8>),
    U64U16(CharsCows<'a, u64, u16>),
    U64U32(CharsCows<'a, u64, u32>),
}

impl<'a> CharsCowsAuto<'a> {
    /// Creates the most memory-efficient CharsCows based on data size and max word length.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsAuto;
    /// use std::borrow::Cow;
    ///
    /// let data = "hello world";
    /// let cows = CharsCowsAuto::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// // Automatically picks CharsCows<u32, u8> for small data
    /// assert_eq!(cows.len(), 2);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    /// Creates the most memory-efficient CharsCows using a two-pass strategy.
    ///
    /// First pass scans to find the maximum word length, then second pass builds
    /// with optimal types. Requires `Clone` iterator for memory efficiency.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsAuto;
    /// use std::borrow::Cow;
    ///
    /// let data = "hello world";
    /// let cows = CharsCowsAuto::from_iter_and_data(
    ///     data.split_whitespace(),  // Clone iterator
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// assert_eq!(cows.len(), 2);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn from_iter_and_data<I>(iter: I, data: Cow<'a, [u8]>) -> Result<Self, StringTapeError>
    where
        I: IntoIterator + Clone,
        I::Item: AsRef<str>,
    {
        let data_len = data.len();

        // First pass: find max word length without materializing
        let max_word_len = iter
            .clone()
            .into_iter()
            .map(|s| s.as_ref().len())
            .max()
            .unwrap_or(0);

        // Pick smallest offset type
        let needs_u64_offset = data_len > u32::MAX as usize;

        // Second pass: build with optimal types
        if max_word_len <= u8::MAX as usize {
            if needs_u64_offset {
                Ok(Self::U64U8(CharsCows::from_iter_and_data(iter, data)?))
            } else {
                Ok(Self::U32U8(CharsCows::from_iter_and_data(iter, data)?))
            }
        } else if max_word_len <= u16::MAX as usize {
            if needs_u64_offset {
                Ok(Self::U64U16(CharsCows::from_iter_and_data(iter, data)?))
            } else {
                Ok(Self::U32U16(CharsCows::from_iter_and_data(iter, data)?))
            }
        } else if needs_u64_offset {
            Ok(Self::U64U32(CharsCows::from_iter_and_data(iter, data)?))
        } else {
            Ok(Self::U32U32(CharsCows::from_iter_and_data(iter, data)?))
        }
    }

    /// Returns the number of cows.
    pub fn len(&self) -> usize {
        match self {
            Self::U32U8(s) => s.len(),
            Self::U32U16(s) => s.len(),
            Self::U32U32(s) => s.len(),
            Self::U64U8(s) => s.len(),
            Self::U64U16(s) => s.len(),
            Self::U64U32(s) => s.len(),
        }
    }

    /// Returns `true` if the collection contains no cows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the string at the given index.
    pub fn get(&self, index: usize) -> Option<&str> {
        match self {
            Self::U32U8(s) => s.get(index),
            Self::U32U16(s) => s.get(index),
            Self::U32U32(s) => s.get(index),
            Self::U64U8(s) => s.get(index),
            Self::U64U16(s) => s.get(index),
            Self::U64U32(s) => s.get(index),
        }
    }

    /// Returns the byte size per entry for the selected type combination.
    pub fn bytes_per_entry(&self) -> usize {
        match self {
            Self::U32U8(_) => 5,   // u32(4) + u8(1)
            Self::U32U16(_) => 6,  // u32(4) + u16(2)
            Self::U32U32(_) => 8,  // u32(4) + u32(4)
            Self::U64U8(_) => 9,   // u64(8) + u8(1)
            Self::U64U16(_) => 10, // u64(8) + u16(2)
            Self::U64U32(_) => 12, // u64(8) + u32(4)
        }
    }

    /// Returns a string describing the selected type combination.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U32U8(_) => "CharsCows<u32, u8>",
            Self::U32U16(_) => "CharsCows<u32, u16>",
            Self::U32U32(_) => "CharsCows<u32, u32>",
            Self::U64U8(_) => "CharsCows<u64, u8>",
            Self::U64U16(_) => "CharsCows<u64, u16>",
            Self::U64U32(_) => "CharsCows<u64, u32>",
        }
    }

    /// Returns an iterator over the string cows.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsAuto;
    /// use std::borrow::Cow;
    ///
    /// let data = "hello world foo";
    /// let cows = CharsCowsAuto::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// let words: Vec<&str> = cows.iter().collect();
    /// assert_eq!(words, vec!["hello", "world", "foo"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn iter(&self) -> CharsCowsAutoIter<'_> {
        CharsCowsAutoIter {
            inner: self,
            index: 0,
        }
    }

    /// Sorts the slices in-place using the default string comparison.
    ///
    /// This is a stable sort that preserves the order of equal elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsAuto;
    /// use std::borrow::Cow;
    ///
    /// let data = "zebra apple banana";
    /// let mut cows = CharsCowsAuto::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// cows.sort();
    /// let sorted: Vec<&str> = cows.iter().collect();
    /// assert_eq!(sorted, vec!["apple", "banana", "zebra"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn sort(&mut self) {
        match self {
            Self::U32U8(s) => s.sort(),
            Self::U32U16(s) => s.sort(),
            Self::U32U32(s) => s.sort(),
            Self::U64U8(s) => s.sort(),
            Self::U64U16(s) => s.sort(),
            Self::U64U32(s) => s.sort(),
        }
    }

    /// Sorts the slices in-place using an unstable sorting algorithm.
    ///
    /// This is faster than stable sort but may not preserve the order of equal elements.
    pub fn sort_unstable(&mut self) {
        match self {
            Self::U32U8(s) => s.sort_unstable(),
            Self::U32U16(s) => s.sort_unstable(),
            Self::U32U32(s) => s.sort_unstable(),
            Self::U64U8(s) => s.sort_unstable(),
            Self::U64U16(s) => s.sort_unstable(),
            Self::U64U32(s) => s.sort_unstable(),
        }
    }

    /// Sorts the slices in-place using a custom comparison function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsAuto;
    /// use std::borrow::Cow;
    ///
    /// let data = "aaa bb c";
    /// let mut cows = CharsCowsAuto::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// // Sort by length, then alphabetically
    /// cows.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
    /// let sorted: Vec<&str> = cows.iter().collect();
    /// assert_eq!(sorted, vec!["c", "bb", "aaa"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&str, &str) -> core::cmp::Ordering,
    {
        match self {
            Self::U32U8(s) => s.sort_by(compare),
            Self::U32U16(s) => s.sort_by(compare),
            Self::U32U32(s) => s.sort_by(compare),
            Self::U64U8(s) => s.sort_by(compare),
            Self::U64U16(s) => s.sort_by(compare),
            Self::U64U32(s) => s.sort_by(compare),
        }
    }

    /// Sorts the slices in-place using a key extraction function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsAuto;
    /// use std::borrow::Cow;
    ///
    /// let data = "aaa bb c";
    /// let mut cows = CharsCowsAuto::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// // Sort by string length
    /// cows.sort_by_key(|s| s.len());
    /// let sorted: Vec<&str> = cows.iter().collect();
    /// assert_eq!(sorted, vec!["c", "bb", "aaa"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn sort_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&str) -> K,
        K: Ord,
    {
        match self {
            Self::U32U8(s) => s.sort_by_key(f),
            Self::U32U16(s) => s.sort_by_key(f),
            Self::U32U32(s) => s.sort_by_key(f),
            Self::U64U8(s) => s.sort_by_key(f),
            Self::U64U16(s) => s.sort_by_key(f),
            Self::U64U32(s) => s.sort_by_key(f),
        }
    }

    /// Returns a zero-copy view of this `CharsCowsAuto` as a `BytesCowsAuto`.
    ///
    /// This is a no-cost operation that reinterprets the string collection as bytes
    /// without copying or moving any data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsAuto;
    /// use std::borrow::Cow;
    ///
    /// let data = "hello world";
    /// let cows = CharsCowsAuto::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// let bytes = cows.as_bytes();
    /// assert_eq!(bytes.get(0), Some(&b"hello"[..]));
    /// assert_eq!(bytes.get(1), Some(&b"world"[..]));
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn as_bytes(&self) -> BytesCowsAuto<'_> {
        match self {
            Self::U32U8(s) => BytesCowsAuto::U32U8(s.as_bytes()),
            Self::U32U16(s) => BytesCowsAuto::U32U16(s.as_bytes()),
            Self::U32U32(s) => BytesCowsAuto::U32U32(s.as_bytes()),
            Self::U64U8(s) => BytesCowsAuto::U64U8(s.as_bytes()),
            Self::U64U16(s) => BytesCowsAuto::U64U16(s.as_bytes()),
            Self::U64U32(s) => BytesCowsAuto::U64U32(s.as_bytes()),
        }
    }
}

/// Iterator over CharsCowsAuto string cows.
pub struct CharsCowsAutoIter<'a> {
    inner: &'a CharsCowsAuto<'a>,
    index: usize,
}

impl<'a> Iterator for CharsCowsAutoIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.inner.get(self.index);
        if result.is_some() {
            self.index += 1;
        }
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.inner.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for CharsCowsAutoIter<'a> {}

impl<'a> IntoIterator for &'a CharsCowsAuto<'a> {
    type Item = &'a str;
    type IntoIter = CharsCowsAutoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ========================
// Auto-selecting BytesCows
// ========================

/// Automatically selects the most memory-efficient BytesCows type based on data size.
pub enum BytesCowsAuto<'a> {
    U32U8(BytesCows<'a, u32, u8>),
    U32U16(BytesCows<'a, u32, u16>),
    U32U32(BytesCows<'a, u32, u32>),
    U64U8(BytesCows<'a, u64, u8>),
    U64U16(BytesCows<'a, u64, u16>),
    U64U32(BytesCows<'a, u64, u32>),
}

impl<'a> BytesCowsAuto<'a> {
    /// Creates BytesCowsAuto from iterator of byte cows.
    /// Auto-selects offset and length types based on data size and max slice length.
    pub fn from_iter_and_data<I>(iter: I, data: Cow<'a, [u8]>) -> Result<Self, StringTapeError>
    where
        I: IntoIterator + Clone,
        I::Item: AsRef<[u8]>,
    {
        let data_len = data.len();

        // First pass: find max slice length
        let max_len = iter
            .clone()
            .into_iter()
            .map(|b| b.as_ref().len())
            .max()
            .unwrap_or(0);

        let needs_u64_offset = data_len > u32::MAX as usize;

        // Second pass: build with optimal types
        if max_len <= u8::MAX as usize {
            if needs_u64_offset {
                Ok(Self::U64U8(BytesCows::from_iter_and_data(iter, data)?))
            } else {
                Ok(Self::U32U8(BytesCows::from_iter_and_data(iter, data)?))
            }
        } else if max_len <= u16::MAX as usize {
            if needs_u64_offset {
                Ok(Self::U64U16(BytesCows::from_iter_and_data(iter, data)?))
            } else {
                Ok(Self::U32U16(BytesCows::from_iter_and_data(iter, data)?))
            }
        } else if needs_u64_offset {
            Ok(Self::U64U32(BytesCows::from_iter_and_data(iter, data)?))
        } else {
            Ok(Self::U32U32(BytesCows::from_iter_and_data(iter, data)?))
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U32U8(s) => s.len(),
            Self::U32U16(s) => s.len(),
            Self::U32U32(s) => s.len(),
            Self::U64U8(s) => s.len(),
            Self::U64U16(s) => s.len(),
            Self::U64U32(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&[u8]> {
        match self {
            Self::U32U8(s) => s.get(index),
            Self::U32U16(s) => s.get(index),
            Self::U32U32(s) => s.get(index),
            Self::U64U8(s) => s.get(index),
            Self::U64U16(s) => s.get(index),
            Self::U64U32(s) => s.get(index),
        }
    }

    /// Returns a zero-copy view of this `BytesCowsAuto` as a `CharsCowsAuto` if all slices are valid UTF-8.
    ///
    /// This validates that all byte slices contain valid UTF-8, then reinterprets the collection
    /// as strings without copying or moving any data.
    ///
    /// # Errors
    ///
    /// Returns `StringTapeError::Utf8Error` if any slice contains invalid UTF-8.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::BytesCowsAuto;
    /// use std::borrow::Cow;
    ///
    /// let data = b"hello world";
    /// let bytes = BytesCowsAuto::from_iter_and_data(
    ///     data.split(|&b| b == b' '),
    ///     Cow::Borrowed(&data[..])
    /// ).unwrap();
    ///
    /// let chars = bytes.as_chars().unwrap();
    /// assert_eq!(chars.get(0), Some("hello"));
    /// assert_eq!(chars.get(1), Some("world"));
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn as_chars(&self) -> Result<CharsCowsAuto<'_>, StringTapeError> {
        match self {
            Self::U32U8(s) => Ok(CharsCowsAuto::U32U8(s.as_chars()?)),
            Self::U32U16(s) => Ok(CharsCowsAuto::U32U16(s.as_chars()?)),
            Self::U32U32(s) => Ok(CharsCowsAuto::U32U32(s.as_chars()?)),
            Self::U64U8(s) => Ok(CharsCowsAuto::U64U8(s.as_chars()?)),
            Self::U64U16(s) => Ok(CharsCowsAuto::U64U16(s.as_chars()?)),
            Self::U64U32(s) => Ok(CharsCowsAuto::U64U32(s.as_chars()?)),
        }
    }
}

// ========================
// Auto-selecting CharsTape
// ========================

/// Automatically selects the most memory-efficient CharsTape offset type.
pub enum CharsTapeAuto<A: Allocator = Global> {
    I32(CharsTape<i32, A>),
    U32(CharsTape<u32, A>),
    U64(CharsTape<u64, A>),
}

impl<A: Allocator> CharsTapeAuto<A> {
    /// Creates CharsTapeAuto with custom allocator.
    pub fn new_in(allocator: A) -> Self {
        Self::I32(CharsTape::new_in(allocator))
    }

    pub fn push(&mut self, s: &str) -> Result<(), StringTapeError> {
        match self {
            Self::I32(t) => t.push(s),
            Self::U32(t) => t.push(s),
            Self::U64(t) => t.push(s),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::I32(t) => t.len(),
            Self::U32(t) => t.len(),
            Self::U64(t) => t.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&str> {
        match self {
            Self::I32(t) => t.get(index),
            Self::U32(t) => t.get(index),
            Self::U64(t) => t.get(index),
        }
    }
}

impl Default for CharsTapeAuto<Global> {
    fn default() -> Self {
        Self::new_in(Global)
    }
}

impl<A: Allocator + Clone> CharsTapeAuto<A> {
    /// Creates tape from clonable iterator, auto-selecting offset type (I32/U32/U64) based on total data size.
    /// Two-pass: first calculates size, second builds tape.
    pub fn from_iter_in<'a, I>(iter: I, allocator: A) -> Self
    where
        I: IntoIterator<Item = &'a str> + Clone,
    {
        // First pass: calculate total data size to determine offset type
        let total_size: usize = iter.clone().into_iter().map(|s| s.len()).sum();

        // Choose optimal type based on data size
        if total_size <= i32::MAX as usize {
            let mut tape = CharsTape::new_in(allocator);
            for s in iter {
                tape.push(s).ok();
            }
            Self::I32(tape)
        } else if total_size <= u32::MAX as usize {
            let mut tape = CharsTape::new_in(allocator);
            for s in iter {
                tape.push(s).ok();
            }
            Self::U32(tape)
        } else {
            let mut tape = CharsTape::new_in(allocator);
            for s in iter {
                tape.push(s).ok();
            }
            Self::U64(tape)
        }
    }
}

impl CharsTapeAuto<Global> {
    /// Creates tape from clonable iterator with global allocator.
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<'a, I>(iter: I) -> Self
    where
        I: IntoIterator<Item = &'a str> + Clone,
    {
        Self::from_iter_in(iter, Global)
    }
}

// ========================
// Auto-selecting BytesTape
// ========================

/// Automatically selects the most memory-efficient BytesTape offset type.
pub enum BytesTapeAuto<A: Allocator = Global> {
    U16(BytesTape<u16, A>),
    U32(BytesTape<u32, A>),
    U64(BytesTape<u64, A>),
}

impl<A: Allocator> BytesTapeAuto<A> {
    /// Creates BytesTapeAuto with custom allocator.
    pub fn new_in(allocator: A) -> Self {
        Self::U16(BytesTape::new_in(allocator))
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<(), StringTapeError> {
        match self {
            Self::U16(t) => t.push(bytes),
            Self::U32(t) => t.push(bytes),
            Self::U64(t) => t.push(bytes),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U16(t) => t.len(),
            Self::U32(t) => t.len(),
            Self::U64(t) => t.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&[u8]> {
        match self {
            Self::U16(t) => t.get(index),
            Self::U32(t) => t.get(index),
            Self::U64(t) => t.get(index),
        }
    }
}

impl Default for BytesTapeAuto<Global> {
    fn default() -> Self {
        Self::new_in(Global)
    }
}

impl<A: Allocator + Clone> BytesTapeAuto<A> {
    /// Creates tape from clonable iterator, auto-selecting offset type (U16/U32/U64) based on total data size.
    /// Two-pass: first calculates size, second builds tape.
    pub fn from_iter_in<'a, I>(iter: I, allocator: A) -> Self
    where
        I: IntoIterator<Item = &'a [u8]> + Clone,
    {
        // First pass: calculate total data size to determine offset type
        let total_size: usize = iter.clone().into_iter().map(|b| b.len()).sum();

        // Choose optimal type based on data size
        if total_size <= u16::MAX as usize {
            let mut tape = BytesTape::new_in(allocator);
            for bytes in iter {
                tape.push(bytes).ok();
            }
            Self::U16(tape)
        } else if total_size <= u32::MAX as usize {
            let mut tape = BytesTape::new_in(allocator);
            for bytes in iter {
                tape.push(bytes).ok();
            }
            Self::U32(tape)
        } else {
            let mut tape = BytesTape::new_in(allocator);
            for bytes in iter {
                tape.push(bytes).ok();
            }
            Self::U64(tape)
        }
    }
}

impl BytesTapeAuto<Global> {
    /// Creates tape from clonable iterator with global allocator.
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<'a, I>(iter: I) -> Self
    where
        I: IntoIterator<Item = &'a [u8]> + Clone,
    {
        Self::from_iter_in(iter, Global)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::string::ToString;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    #[test]
    fn basic_operations() {
        let mut tape = CharsTapeI32::new();
        assert!(tape.is_empty());

        tape.push("hello").unwrap();
        tape.push("world").unwrap();
        tape.push("foo").unwrap();

        assert_eq!(tape.len(), 3);
        assert_eq!(tape.get(0), Some("hello"));
        assert_eq!(tape.get(1), Some("world"));
        assert_eq!(tape.get(2), Some("foo"));
        assert_eq!(tape.get(3), None);
    }

    #[test]
    fn unsigned_basic_operations() {
        // u32
        let mut t32 = CharsTapeU32::new();
        t32.push("hello").unwrap();
        t32.push("world").unwrap();
        assert_eq!(t32.len(), 2);
        assert_eq!(t32.get(0), Some("hello"));
        assert_eq!(t32.get(1), Some("world"));

        // u64
        let mut t64 = CharsTapeU64::new();
        t64.extend(["a", "", "bbb"]).unwrap();
        assert_eq!(t64.len(), 3);
        assert_eq!(t64.get(0), Some("a"));
        assert_eq!(t64.get(1), Some(""));
        assert_eq!(t64.get(2), Some("bbb"));
    }

    #[test]
    fn offsets_64bit() {
        let mut tape = CharsTapeI64::new();
        tape.push("test").unwrap();
        assert_eq!(tape.get(0), Some("test"));
    }

    #[test]
    fn iterator_basics() {
        let mut tape = CharsTapeI32::new();
        tape.push("a").unwrap();
        tape.push("b").unwrap();
        tape.push("c").unwrap();

        let strings: Vec<&str> = tape.iter().collect();
        assert_eq!(strings, vec!["a", "b", "c"]);
    }

    #[test]
    fn empty_strings() {
        let mut tape = CharsTapeI32::new();
        tape.push("").unwrap();
        tape.push("non-empty").unwrap();
        tape.push("").unwrap();

        assert_eq!(tape.len(), 3);
        assert_eq!(tape.get(0), Some(""));
        assert_eq!(tape.get(1), Some("non-empty"));
        assert_eq!(tape.get(2), Some(""));
    }

    #[test]
    fn index_trait() {
        let mut tape = CharsTapeI32::new();
        tape.push("hello").unwrap();
        tape.push("world").unwrap();

        assert_eq!(&tape[0], "hello");
        assert_eq!(&tape[1], "world");
    }

    #[test]
    fn into_iterator() {
        let mut tape = CharsTapeI32::new();
        tape.push("a").unwrap();
        tape.push("b").unwrap();
        tape.push("c").unwrap();

        let strings: Vec<&str> = (&tape).into_iter().collect();
        assert_eq!(strings, vec!["a", "b", "c"]);

        // Test for-loop syntax
        let mut result = Vec::new();
        for s in &tape {
            result.push(s);
        }
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn from_iterator() {
        let strings = vec!["hello", "world", "test"];
        let tape: CharsTapeI32 = strings.into_iter().collect();

        assert_eq!(tape.len(), 3);
        assert_eq!(tape.get(0), Some("hello"));
        assert_eq!(tape.get(1), Some("world"));
        assert_eq!(tape.get(2), Some("test"));
    }

    #[test]
    fn from_iterator_unsigned() {
        let strings = vec!["hello", "world", "test"];
        let tape_u32: CharsTapeU32 = strings.clone().into_iter().collect();
        let tape_u64: CharsTapeU64 = strings.clone().into_iter().collect();
        assert_eq!(tape_u32.len(), 3);
        assert_eq!(tape_u64.len(), 3);
        assert_eq!(tape_u32.get(1), Some("world"));
        assert_eq!(tape_u64.get(2), Some("test"));
    }

    #[test]
    fn extend() {
        let mut tape = CharsTapeI32::new();
        tape.push("initial").unwrap();

        let additional = vec!["hello", "world"];
        tape.extend(additional).unwrap();

        assert_eq!(tape.len(), 3);
        assert_eq!(tape.get(0), Some("initial"));
        assert_eq!(tape.get(1), Some("hello"));
        assert_eq!(tape.get(2), Some("world"));
    }

    #[test]
    fn clear_and_truncate() {
        let mut tape = CharsTapeI32::new();
        tape.push("a").unwrap();
        tape.push("b").unwrap();
        tape.push("c").unwrap();

        assert_eq!(tape.len(), 3);

        tape.truncate(2);
        assert_eq!(tape.len(), 2);
        assert_eq!(tape.get(0), Some("a"));
        assert_eq!(tape.get(1), Some("b"));
        assert_eq!(tape.get(2), None);

        tape.clear();
        assert_eq!(tape.len(), 0);
        assert!(tape.is_empty());
    }

    #[test]
    fn unsigned_views_and_subviews() {
        let mut tape = CharsTapeU32::new();
        tape.extend(["0", "1", "22", "333"]).unwrap();
        let view = tape.subview(1, 4).unwrap();
        assert_eq!(view.len(), 3);
        assert_eq!(view.get(0), Some("1"));
        assert_eq!(view.get(2), Some("333"));
        let sub = view.subview(1, 2).unwrap();
        assert_eq!(sub.len(), 1);
        assert_eq!(sub.get(0), Some("22"));
    }

    #[test]
    fn capacity() {
        let tape = CharsTapeI32::with_capacity(100, 10).unwrap();
        assert_eq!(tape.data_capacity(), 100);
        assert_eq!(tape.capacity(), 0); // No strings added yet
    }

    #[test]
    fn custom_allocator() {
        // Using the Global allocator explicitly
        let mut tape: CharsTape<i32, Global> = CharsTape::new_in(Global);

        tape.push("hello").unwrap();
        tape.push("world").unwrap();

        assert_eq!(tape.len(), 2);
        assert_eq!(tape.get(0), Some("hello"));
        assert_eq!(tape.get(1), Some("world"));

        // Verify we can access the allocator
        let _allocator_ref = tape.allocator();
    }

    #[test]
    fn custom_allocator_with_capacity() {
        let tape: CharsTape<i64, Global> = CharsTape::with_capacity_in(256, 50, Global).unwrap();

        assert_eq!(tape.data_capacity(), 256);
        assert!(tape.is_empty());
    }

    #[test]
    fn bytes_tape_basic() {
        let mut tape = BytesTapeI32::new();
        tape.push(&[1, 2, 3]).unwrap();
        tape.push(b"abc").unwrap();

        assert_eq!(tape.len(), 2);
        assert_eq!(&tape[0], &[1u8, 2, 3] as &[u8]);
        assert_eq!(&tape[1], b"abc" as &[u8]);
    }

    #[test]
    fn unsigned_bytes_tape_basic() {
        let mut tape = BytesTapeU64::new();
        tape.push(&[1u8, 2]).unwrap();
        tape.push(&[3u8, 4, 5]).unwrap();
        assert_eq!(tape.len(), 2);
        assert_eq!(&tape[0], &[1u8, 2] as &[u8]);
        assert_eq!(&tape[1], &[3u8, 4, 5] as &[u8]);
    }

    #[test]
    fn chars_tape_view_basic() {
        let mut tape = CharsTapeI32::new();
        tape.push("hello").unwrap();
        tape.push("world").unwrap();
        tape.push("foo").unwrap();
        tape.push("bar").unwrap();

        // Test basic subview creation
        let view = tape.subview(1, 3).unwrap();
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0), Some("world"));
        assert_eq!(view.get(1), Some("foo"));
        assert_eq!(view.get(2), None);

        // Test indexing
        assert_eq!(&view[0], "world");
        assert_eq!(&view[1], "foo");
    }

    #[test]
    fn chars_tape_range_syntax() {
        let mut tape = CharsTapeI32::new();
        tape.push("a").unwrap();
        tape.push("b").unwrap();
        tape.push("c").unwrap();
        tape.push("d").unwrap();

        // Test view() method
        let full_view = tape.view();
        assert_eq!(full_view.len(), 4);
        assert_eq!(full_view.get(0), Some("a"));
        assert_eq!(full_view.get(3), Some("d"));

        // Test subview
        let sub = tape.subview(1, 3).unwrap();
        assert_eq!(sub.len(), 2);
        assert_eq!(sub.get(0), Some("b"));
        assert_eq!(sub.get(1), Some("c"));
    }

    #[test]
    fn chars_tape_view_subslicing() {
        let mut tape = CharsTapeI32::new();
        tape.push("0").unwrap();
        tape.push("1").unwrap();
        tape.push("2").unwrap();
        tape.push("3").unwrap();
        tape.push("4").unwrap();

        // Create initial subview
        let view = tape.subview(1, 4).unwrap(); // ["1", "2", "3"]
        assert_eq!(view.len(), 3);

        // Create sub-view of a view
        let subview = view.subview(1, 2).unwrap(); // ["2"]
        assert_eq!(subview.len(), 1);
        assert_eq!(subview.get(0), Some("2"));

        // Test subviews with different ranges
        let subview_from = view.subview(1, view.len()).unwrap(); // ["2", "3"]
        assert_eq!(subview_from.len(), 2);
        assert_eq!(subview_from.get(0), Some("2"));
        assert_eq!(subview_from.get(1), Some("3"));

        let subview_to = view.subview(0, 2).unwrap(); // ["1", "2"]
        assert_eq!(subview_to.len(), 2);
        assert_eq!(subview_to.get(0), Some("1"));
        assert_eq!(subview_to.get(1), Some("2"));
    }

    #[test]
    fn bytes_tape_view_basic() {
        let mut tape = BytesTapeI32::new();
        tape.push(&[1u8, 2]).unwrap();
        tape.push(&[3u8, 4]).unwrap();
        tape.push(&[5u8, 6]).unwrap();
        tape.push(&[7u8, 8]).unwrap();

        // Test basic subview creation
        let view = tape.subview(1, 3).unwrap();
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0), Some(&[3u8, 4] as &[u8]));
        assert_eq!(view.get(1), Some(&[5u8, 6] as &[u8]));
        assert_eq!(view.get(2), None);

        // Test indexing
        assert_eq!(&view[0], &[3u8, 4] as &[u8]);
        assert_eq!(&view[1], &[5u8, 6] as &[u8]);
    }

    #[test]
    fn view_empty_strings() {
        let mut tape = CharsTapeI32::new();
        tape.push("").unwrap();
        tape.push("non-empty").unwrap();
        tape.push("").unwrap();
        tape.push("another").unwrap();

        let view = tape.subview(0, 3).unwrap();
        assert_eq!(view.len(), 3);
        assert_eq!(view.get(0), Some(""));
        assert_eq!(view.get(1), Some("non-empty"));
        assert_eq!(view.get(2), Some(""));
    }

    #[test]
    fn view_single_item() {
        let mut tape = CharsTapeI32::new();
        tape.push("only").unwrap();

        let view = tape.subview(0, 1).unwrap();
        assert_eq!(view.len(), 1);
        assert_eq!(view.get(0), Some("only"));
    }

    #[test]
    fn view_bounds_checking() {
        let mut tape = CharsTapeI32::new();
        tape.push("a").unwrap();
        tape.push("b").unwrap();

        // Out of bounds subview creation
        assert!(tape.subview(0, 3).is_err());
        assert!(tape.subview(2, 1).is_err());
        assert!(tape.subview(3, 4).is_err());

        // Valid empty subview
        let empty_view = tape.subview(1, 1).unwrap();
        assert_eq!(empty_view.len(), 0);
        assert!(empty_view.is_empty());
    }

    #[test]
    fn view_data_properties() {
        let mut tape = CharsTapeI32::new();
        tape.push("hello").unwrap(); // 5 bytes
        tape.push("world").unwrap(); // 5 bytes
        tape.push("!").unwrap(); // 1 byte

        let view = tape.subview(0, 2).unwrap(); // "hello", "world" = 10 bytes
        assert_eq!(view.data_len(), 10);
        assert!(!view.is_empty());

        let full_view = tape.subview(0, 3).unwrap(); // all = 11 bytes
        assert_eq!(full_view.data_len(), 11);
    }

    #[test]
    fn view_raw_parts() {
        let mut tape = CharsTapeI32::new();
        tape.push("test").unwrap();
        tape.push("data").unwrap();

        let view = tape.subview(0, 2).unwrap();
        let parts = view.as_raw_parts();

        assert!(!parts.data_ptr.is_null());
        assert!(!parts.offsets_ptr.is_null());
        assert_eq!(parts.data_len, 8); // "test" + "data"
        assert_eq!(parts.items_count, 2);
    }

    #[test]
    fn subview_raw_parts_consistency_chars() {
        let mut tape = CharsTapeI32::new();
        tape.extend(["abc", "", "xyz", "pq"]).unwrap();

        // Subview over middle two items: ["", "xyz"]
        let view = tape.subview(1, 3).unwrap();
        let parts = view.as_raw_parts();

        // Offsets len must be items_count + 1 and data_len equals absolute last offset
        unsafe {
            let offsets: &[i32] =
                core::slice::from_raw_parts(parts.offsets_ptr, parts.items_count + 1);
            assert_eq!(offsets.len(), parts.items_count + 1);
            assert!(offsets.windows(2).all(|w| w[0] <= w[1]));
            let last_abs = offsets[offsets.len() - 1] as usize;
            assert_eq!(last_abs, parts.data_len);
        }

        // Also check that element boundaries are respected
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0), Some(""));
        assert_eq!(view.get(1), Some("xyz"));
    }

    #[test]
    fn subview_raw_parts_consistency_bytes() {
        let mut tape = BytesTapeI32::new();
        tape.extend([
            b"a".as_slice(),
            b"".as_slice(),
            b"bc".as_slice(),
            b"def".as_slice(),
        ])
        .unwrap();

        // Subview over last two items: ["bc", "def"]
        let view = tape.subview(2, 4).unwrap();
        let parts = view.as_raw_parts();

        unsafe {
            let offsets: &[i32] =
                core::slice::from_raw_parts(parts.offsets_ptr, parts.items_count + 1);
            assert_eq!(offsets.len(), parts.items_count + 1);
            assert!(offsets.windows(2).all(|w| w[0] <= w[1]));
            let last_abs = offsets[offsets.len() - 1] as usize;
            assert_eq!(last_abs, parts.data_len);
        }

        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0), Some(b"bc" as &[u8]));
        assert_eq!(view.get(1), Some(b"def" as &[u8]));
    }

    #[test]
    fn view_type_aliases() {
        let mut tape = CharsTapeI32::new();
        tape.push("test").unwrap();

        let _view: CharsTapeViewI32 = tape.subview(0, 1).unwrap();

        let mut bytes_tape = BytesTapeI64::new();
        bytes_tape.push(b"test").unwrap();

        let _bytes_view: BytesTapeViewI64 = bytes_tape.subview(0, 1).unwrap();
    }

    #[test]
    fn build_i32_from_other_offset_iterators() {
        let items = ["x", "yy", "", "zzz"];

        // From u32 iterator
        let mut u32t = CharsTapeU32::new();
        u32t.extend(items).unwrap();
        let t_from_u32: CharsTapeI32 = u32t.iter().collect();
        assert_eq!(t_from_u32.len(), items.len());
        assert_eq!(t_from_u32.get(1), Some("yy"));

        // From u64 iterator
        let mut u64t = CharsTapeU64::new();
        u64t.extend(items).unwrap();
        let t_from_u64: CharsTapeI32 = u64t.iter().collect();
        assert_eq!(t_from_u64.len(), items.len());
        assert_eq!(t_from_u64.get(3), Some("zzz"));

        // From i64 iterator
        let mut i64t = CharsTapeI64::new();
        i64t.extend(items).unwrap();
        let t_from_i64: CharsTapeI32 = i64t.iter().collect();
        assert_eq!(t_from_i64.len(), items.len());
        assert_eq!(t_from_i64.get(2), Some(""));
    }

    #[test]
    fn range_indexing_syntax() {
        let mut tape = CharsTapeI32::new();
        tape.push("a").unwrap();
        tape.push("b").unwrap();
        tape.push("c").unwrap();
        tape.push("d").unwrap();

        // While we can't return views with [..] syntax due to lifetime constraints,
        // we can test that the view() and subview() API works correctly

        // Get full view
        let full_view = tape.view();
        assert_eq!(full_view.len(), 4);

        // Get subviews
        let sub = tape.subview(1, 3).unwrap();
        assert_eq!(sub.len(), 2);
        assert_eq!(sub.get(0), Some("b"));
        assert_eq!(sub.get(1), Some("c"));

        // Test subview of subview
        let sub_sub = sub.subview(0, 1).unwrap();
        assert_eq!(sub_sub.len(), 1);
        assert_eq!(sub_sub.get(0), Some("b"));
    }

    #[cfg(test)]
    use arrow::array::{Array, BinaryArray, StringArray};
    #[cfg(test)]
    use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};

    #[test]
    fn charstape_to_arrow_string_array() {
        let mut tape = CharsTapeI32::new();
        tape.extend(["hello", "world", "", "arrow"]).unwrap();

        let (data_slice, offsets_slice) = tape.arrow_slices();
        let data_buffer = Buffer::from_slice_ref(data_slice);
        let offsets_buffer = OffsetBuffer::new(ScalarBuffer::new(
            Buffer::from_slice_ref(offsets_slice),
            0,
            offsets_slice.len(),
        ));
        let arrow_array = StringArray::new(offsets_buffer, data_buffer, None);

        assert_eq!(arrow_array.len(), 4);
        assert_eq!(arrow_array.value(0), "hello");
        assert_eq!(arrow_array.value(2), "");
    }

    #[test]
    fn arrow_string_array_to_charstape_view() {
        let arrow_array = StringArray::from(vec!["foo", "bar", ""]);

        // Zero-copy conversion to CharsTapeView
        let view = unsafe {
            CharsTapeViewI32::from_raw_parts(arrow_array.values(), arrow_array.offsets().as_ref())
        };

        assert_eq!(view.len(), 3);
        assert_eq!(view.get(0), Some("foo"));
        assert_eq!(view.get(1), Some("bar"));
        assert_eq!(view.get(2), Some(""));
    }

    #[test]
    fn arrow_binary_array_to_bytestape_view() {
        let values: Vec<Option<&[u8]>> = vec![
            Some(&[1u8, 2, 3] as &[u8]),
            Some(&[] as &[u8]),
            Some(&[4u8, 5] as &[u8]),
        ];
        let arrow_array = BinaryArray::from(values);

        // Zero-copy conversion to BytesTapeView
        let view = unsafe {
            BytesTapeViewI32::from_raw_parts(arrow_array.values(), arrow_array.offsets().as_ref())
        };

        assert_eq!(view.len(), 3);
        assert_eq!(view.get(0), Some(&[1u8, 2, 3] as &[u8]));
        assert_eq!(view.get(1), Some(&[] as &[u8]));
        assert_eq!(view.get(2), Some(&[4u8, 5] as &[u8]));
    }

    #[test]
    fn zero_copy_roundtrip() {
        // Original data
        let mut tape = CharsTapeI32::new();
        tape.extend(["hello", "", "world"]).unwrap();

        // Convert to Arrow (zero-copy)
        let (data_slice, offsets_slice) = tape.arrow_slices();
        let data_buffer = Buffer::from_slice_ref(data_slice);
        let offsets_buffer = OffsetBuffer::new(ScalarBuffer::new(
            Buffer::from_slice_ref(offsets_slice),
            0,
            offsets_slice.len(),
        ));
        let arrow_array = StringArray::new(offsets_buffer, data_buffer, None);

        // Convert back to CharsTapeView (zero-copy)
        let view = unsafe {
            CharsTapeViewI32::from_raw_parts(arrow_array.values(), arrow_array.offsets().as_ref())
        };

        // Verify data integrity without any copying
        assert_eq!(view.len(), 3);
        assert_eq!(view.get(0), Some("hello"));
        assert_eq!(view.get(1), Some(""));
        assert_eq!(view.get(2), Some("world"));
    }

    #[test]
    fn bytes_to_string_conversion() {
        // Test successful conversion with valid UTF-8
        let mut bytes_tape = BytesTapeI32::new();
        bytes_tape.push(b"hello").unwrap();
        bytes_tape.push(b"world").unwrap();
        bytes_tape.push(b"").unwrap();
        bytes_tape.push(b"rust").unwrap();

        let chars_tape: Result<CharsTapeI32, _> = bytes_tape.try_into();
        assert!(chars_tape.is_ok());

        let chars_tape = chars_tape.unwrap();
        assert_eq!(chars_tape.len(), 4);
        assert_eq!(chars_tape.get(0), Some("hello"));
        assert_eq!(chars_tape.get(1), Some("world"));
        assert_eq!(chars_tape.get(2), Some(""));
        assert_eq!(chars_tape.get(3), Some("rust"));
    }

    #[test]
    fn bytes_to_string_invalid_utf8() {
        // Test conversion failure with invalid UTF-8
        let mut bytes_tape = BytesTapeI32::new();
        bytes_tape.push(b"valid").unwrap();
        bytes_tape.push(&[0xFF, 0xFE]).unwrap(); // Invalid UTF-8 sequence
        bytes_tape.push(b"also valid").unwrap();

        let chars_tape: Result<CharsTapeI32, _> = bytes_tape.try_into();
        assert!(chars_tape.is_err());

        match chars_tape {
            Err(StringTapeError::Utf8Error(_)) => {}
            _ => panic!("Expected Utf8Error"),
        }
    }

    #[test]
    fn string_to_bytes_conversion() {
        // Test infallible conversion from CharsTape to BytesTape
        let mut chars_tape = CharsTapeI32::new();
        chars_tape.push("hello").unwrap();
        chars_tape.push("世界").unwrap(); // Unicode characters
        chars_tape.push("").unwrap();
        chars_tape.push("🦀").unwrap(); // Emoji

        let bytes_tape: BytesTapeI32 = chars_tape.into();
        assert_eq!(bytes_tape.len(), 4);
        assert_eq!(&bytes_tape[0], b"hello");
        assert_eq!(&bytes_tape[1], "世界".as_bytes());
        assert_eq!(&bytes_tape[2], b"");
        assert_eq!(&bytes_tape[3], "🦀".as_bytes());
    }

    #[test]
    fn conversion_convenience_methods() {
        // Test try_into_chars_tape method
        let mut bytes_tape = BytesTapeI32::new();
        bytes_tape.push(b"test").unwrap();
        let string_result = bytes_tape.try_into_chars_tape();
        assert!(string_result.is_ok());
        assert_eq!(string_result.unwrap().get(0), Some("test"));

        // Test into_bytes_tape method
        let mut chars_tape = CharsTapeI32::new();
        chars_tape.push("test").unwrap();
        let bytes_back = chars_tape.into_bytes_tape();
        assert_eq!(&bytes_back[0], b"test");
    }

    #[test]
    fn conversion_round_trip() {
        // Test round-trip conversion preserves data
        let mut original = CharsTapeI32::new();
        original.push("first").unwrap();
        original.push("second").unwrap();
        original.push("third").unwrap();

        // Store expected values before conversion
        let expected = vec!["first", "second", "third"];

        // Convert to BytesTape and back
        let bytes: BytesTapeI32 = original.into();
        let recovered: CharsTapeI32 = bytes.try_into().unwrap();

        assert_eq!(expected.len(), recovered.len());
        for (i, expected_str) in expected.iter().enumerate() {
            assert_eq!(recovered.get(i), Some(*expected_str));
        }
    }

    #[test]
    fn view_to_view_conversions_valid_utf8() {
        // Prepare a CharsTape and obtain its view
        let mut ct = CharsTapeI32::new();
        ct.extend(["abc", "", "世界"]).unwrap();
        let chars_view = ct.view();

        // Chars -> Bytes view conversion is infallible
        let bytes_view: BytesTapeViewI32 = chars_view.into_bytes_view();
        assert_eq!(bytes_view.len(), 3);
        assert_eq!(bytes_view.get(0), Some("abc".as_bytes()));
        assert_eq!(bytes_view.get(1), Some(b"" as &[u8]));
        assert_eq!(bytes_view.get(2), Some("世界".as_bytes()));

        // Bytes -> Chars view conversion is fallible, but should succeed for valid UTF-8
        let chars_back: Result<CharsTapeViewI32, _> = bytes_view.try_into_chars_view();
        assert!(chars_back.is_ok());
        let chars_back = chars_back.unwrap();
        assert_eq!(chars_back.len(), 3);
        assert_eq!(chars_back.get(0), Some("abc"));
        assert_eq!(chars_back.get(1), Some(""));
        assert_eq!(chars_back.get(2), Some("世界"));
    }

    #[test]
    fn view_to_view_bytes_to_chars_invalid_utf8() {
        // Prepare a BytesTape with invalid UTF-8 payload
        let mut bt = BytesTapeI32::new();
        bt.push(b"ok").unwrap();
        bt.push(&[0xFF, 0xFE]).unwrap(); // invalid UTF-8
        let bview = bt.view();

        // Converting to CharsTapeView should fail
        let res: Result<CharsTapeViewI32, _> = bview.try_into_chars_view();
        assert!(res.is_err());
        match res {
            Err(StringTapeError::Utf8Error(_)) => {}
            _ => panic!("Expected Utf8Error"),
        }
    }

    #[test]
    fn chars_slices_basic() {
        let data = "hello world foo bar";
        let cows = CharsCowsU32U16::from_iter_and_data(
            data.split_whitespace(),
            Cow::Borrowed(data.as_bytes()),
        )
        .unwrap();

        assert_eq!(cows.len(), 4);
        assert_eq!(cows.get(0), Some("hello"));
        assert_eq!(cows.get(1), Some("world"));
        assert_eq!(cows.get(2), Some("foo"));
        assert_eq!(cows.get(3), Some("bar"));
        assert_eq!(cows.get(4), None);
    }

    #[test]
    fn chars_slices_index() {
        let data = "abc def";

        let cows = CharsCowsU64U32::from_iter_and_data(
            data.split_whitespace(),
            Cow::Borrowed(data.as_bytes()),
        )
        .unwrap();

        assert_eq!(&cows[0], "abc");
        assert_eq!(&cows[1], "def");
    }

    #[test]
    fn chars_slices_iterator() {
        let data = "a b c";

        let cows = CharsCowsU64U32::from_iter_and_data(
            data.split_whitespace(),
            Cow::Borrowed(data.as_bytes()),
        )
        .unwrap();

        let result: Vec<&str> = cows.iter().collect();
        assert_eq!(result, vec!["a", "b", "c"]);

        // Test for-loop
        let mut count = 0;
        for s in &cows {
            assert_eq!(s.len(), 1);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn chars_slices_arbitrary_order() {
        let data = "0123456789";
        // Create slices in non-sequential order manually
        let s1 = &data[5..7]; // "56"
        let s2 = &data[0..1]; // "0"
        let s3 = &data[9..10]; // "9"
        let s4 = &data[2..5]; // "234"

        let cows =
            CharsCowsU64U32::from_iter_and_data([s1, s2, s3, s4], Cow::Borrowed(data.as_bytes()))
                .unwrap();

        assert_eq!(cows.get(0), Some("56"));
        assert_eq!(cows.get(1), Some("0"));
        assert_eq!(cows.get(2), Some("9"));
        assert_eq!(cows.get(3), Some("234"));
    }

    #[test]
    fn chars_slices_empty_strings() {
        let data = "ab";
        let s1 = &data[0..0]; // empty
        let s2 = &data[1..2]; // "b"
        let s3 = &data[2..2]; // empty

        let cows =
            CharsCowsU64U32::from_iter_and_data([s1, s2, s3], Cow::Borrowed(data.as_bytes()))
                .unwrap();

        assert_eq!(cows.len(), 3);
        assert_eq!(cows.get(0), Some(""));
        assert_eq!(cows.get(1), Some("b"));
        assert_eq!(cows.get(2), Some(""));
    }

    #[test]
    fn chars_slices_overflow_checks() {
        let data_vec = vec![b'x'; 300];
        let data = core::str::from_utf8(&data_vec).unwrap();

        // u8 length overflow - 256 bytes exceeds u8::MAX
        let long_slice = &data[0..256];
        let result = CharsCowsU32U8::from_iter_and_data(
            core::iter::once(long_slice),
            Cow::Borrowed(data.as_bytes()),
        );
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), StringTapeError::OffsetOverflow);

        // Valid with u16 length
        let result = CharsCowsU32U16::from_iter_and_data(
            core::iter::once(long_slice),
            Cow::Borrowed(data.as_bytes()),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn chars_slices_bounds_check() {
        let data = String::from("hello");
        let other_data = String::from("world");

        // Slice from different string - should fail
        let result = CharsCowsU64U32::from_iter_and_data(
            core::iter::once(other_data.as_str()),
            Cow::Borrowed(data.as_bytes()),
        );
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), StringTapeError::IndexOutOfBounds);

        // Valid slice from same string
        let result = CharsCowsU64U32::from_iter_and_data(
            core::iter::once(data.as_str()),
            Cow::Borrowed(data.as_bytes()),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn slices_conversions() {
        let data = "hello world";
        let chars = CharsCowsU32U8::from_iter_and_data(
            data.split_whitespace(),
            Cow::Borrowed(data.as_bytes()),
        )
        .unwrap();

        // CharsCows -> BytesCows
        let bytes: BytesCowsU32U8 = chars.into();
        assert_eq!(bytes.get(0), Some(b"hello" as &[u8]));
        assert_eq!(bytes.get(1), Some(b"world" as &[u8]));

        // N -> CharsCows
        let chars_back: CharsCowsU32U8 = bytes.try_into().unwrap();
        assert_eq!(chars_back.get(0), Some("hello"));
        assert_eq!(chars_back.get(1), Some("world"));
    }

    #[test]
    fn slices_type_aliases() {
        let data = "test";

        let _s1: CharsCowsU32U16 =
            CharsCows::from_iter_and_data(core::iter::once(data), Cow::Borrowed(data.as_bytes()))
                .unwrap();
        let _s2: CharsCowsU32U8 =
            CharsCows::from_iter_and_data(core::iter::once(data), Cow::Borrowed(data.as_bytes()))
                .unwrap();
        let _s3: CharsCowsU16U8 =
            CharsCows::from_iter_and_data(core::iter::once(data), Cow::Borrowed(data.as_bytes()))
                .unwrap();
        let _s4: CharsCowsU64U32 =
            CharsCows::from_iter_and_data(core::iter::once(data), Cow::Borrowed(data.as_bytes()))
                .unwrap();
    }

    #[test]
    fn chars_slices_auto_sorted() {
        let data = "zebra apple banana cherry";
        let mut cows = CharsCowsAuto::from_iter_and_data(
            data.split_whitespace(),
            Cow::Borrowed(data.as_bytes()),
        )
        .unwrap();

        // Sort in-place using standard Rust patterns
        cows.sort();

        let sorted: Vec<&str> = cows.iter().collect();
        assert_eq!(sorted, vec!["apple", "banana", "cherry", "zebra"]);
    }

    #[test]
    fn chars_slices_auto_to_vec_string() {
        let data = "hello world foo";
        let cows = CharsCowsAuto::from_iter_and_data(
            data.split_whitespace(),
            Cow::Borrowed(data.as_bytes()),
        )
        .unwrap();

        // Convert to Vec<String> using iterator
        let vec_string: Vec<String> = cows.iter().map(|s| s.to_string()).collect();

        assert_eq!(vec_string, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn chars_slices_auto_filter_map() {
        let data = "hello world foo bar";
        let cows = CharsCowsAuto::from_iter_and_data(
            data.split_whitespace(),
            Cow::Borrowed(data.as_bytes()),
        )
        .unwrap();

        // Filter and uppercase using iterator - common Vec<String> pattern
        let result: Vec<String> = cows
            .iter()
            .filter_map(|word| {
                if word.len() > 3 {
                    Some(word.to_uppercase())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(result, vec!["HELLO", "WORLD"]);
    }

    #[test]
    fn chars_slices_auto_type_selection() {
        // Small data -> u32 offset, u8 length
        let small = "hi";
        let s1 = CharsCowsAuto::from_iter_and_data(
            core::iter::once(small),
            Cow::Borrowed(small.as_bytes()),
        )
        .unwrap();
        assert!(matches!(s1, CharsCowsAuto::U32U8(_)));
        assert_eq!(s1.bytes_per_entry(), 5);

        // Long word -> u32 offset, u16 length
        let long_word = "a".repeat(300);
        let s2 = CharsCowsAuto::from_iter_and_data(
            core::iter::once(long_word.as_str()),
            Cow::Borrowed(long_word.as_bytes()),
        )
        .unwrap();
        assert!(matches!(s2, CharsCowsAuto::U32U16(_)));
        assert_eq!(s2.bytes_per_entry(), 6);
    }
}

// ========================
// Examples
// ========================

#[cfg(all(feature = "std", not(test)))]
pub mod examples {
    use super::*;
    use std::env;
    use std::fs;

    pub fn bench_vec_string() -> std::io::Result<()> {
        let path = env::args().nth(1).expect("Usage: bench_vec_string <file>");

        eprintln!("[Vec<String>] Loading file: {}", path);
        let content = fs::read_to_string(&path)?;
        eprintln!("[Vec<String>] File size: {} bytes", content.len());

        eprintln!("[Vec<String>] Collecting words...");
        let words: Vec<String> = content.split_whitespace().map(|s| s.to_string()).collect();

        eprintln!("[Vec<String>] Collected {} words", words.len());

        // Keep alive to measure peak
        std::thread::sleep(std::time::Duration::from_millis(1000));
        Ok(())
    }

    pub fn bench_vec_slice() -> std::io::Result<()> {
        let path = env::args().nth(1).expect("Usage: bench_vec_slice <file>");

        eprintln!("[Vec<&[u8]>] Loading file: {}", path);
        let content = fs::read_to_string(&path)?;
        eprintln!("[Vec<&[u8]>] File size: {} bytes", content.len());

        eprintln!("[Vec<&[u8]>] Collecting words...");
        let words: Vec<&[u8]> = content.split_whitespace().map(|s| s.as_bytes()).collect();

        eprintln!("[Vec<&[u8]>] Collected {} words", words.len());

        // Keep alive to measure peak
        std::thread::sleep(std::time::Duration::from_millis(1000));
        Ok(())
    }

    pub fn bench_chars_slices() -> Result<(), Box<dyn std::error::Error>> {
        let path = env::args()
            .nth(1)
            .expect("Usage: bench_chars_slices <file>");

        eprintln!("[CharsCows] Loading file: {}", path);
        let content = fs::read_to_string(&path)?;
        eprintln!("[CharsCows] File size: {} bytes", content.len());

        eprintln!("[CharsCows] Building CharsCows from words...");
        // Use u64 offset for files >4GB, u32 length for words up to 4GB
        let cows = CharsCowsAuto::from_iter_and_data(
            content.split_whitespace(),
            Cow::Borrowed(content.as_bytes()),
        )?;

        eprintln!("[CharsCows] Collected {} words", cows.len());

        // Keep alive to measure peak
        std::thread::sleep(std::time::Duration::from_millis(1000));
        Ok(())
    }

    pub fn bench_chars_tape() -> Result<(), Box<dyn std::error::Error>> {
        let path = env::args().nth(1).expect("Usage: bench_chars_tape <file>");

        eprintln!("[CharsTape] Loading file: {}", path);
        let content = fs::read_to_string(&path)?;
        eprintln!("[CharsTape] File size: {} bytes", content.len());

        eprintln!("[CharsTape] Building CharsTape from words...");
        // Use from_iter for automatic type selection based on total data size
        let tape = CharsTapeAuto::from_iter(content.split_whitespace());

        eprintln!("[CharsTape] Collected {} words", tape.len());

        // Keep alive to measure peak
        std::thread::sleep(std::time::Duration::from_millis(1000));
        Ok(())
    }
}

// ========================
// Binary entry points
// ========================

#[cfg(all(feature = "std", not(test)))]
#[allow(dead_code)] // Only used when building binaries, not when checking lib
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let exe_path = std::env::current_exe()?;
    let exe_name = exe_path.file_name().and_then(|n| n.to_str()).unwrap_or("");

    match exe_name {
        "bench_vec_string" => examples::bench_vec_string()?,
        "bench_vec_slice" => examples::bench_vec_slice()?,
        "bench_chars_slices" => examples::bench_chars_slices()?,
        "bench_chars_tape" => examples::bench_chars_tape()?,
        _ => {
            eprintln!("Unknown binary: {}", exe_name);
            eprintln!("Available: bench_vec_string, bench_vec_slice, bench_chars_slices, bench_chars_tape");
            std::process::exit(1);
        }
    }

    Ok(())
}
