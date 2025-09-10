#![cfg_attr(not(feature = "std"), no_std)]

//! # StringTape
//!
//! Memory-efficient string and bytes storage compatible with Apache Arrow.
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
//! It also supports binary data via `BytesTape`:
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
use alloc::string::String;

use allocator_api2::alloc::{Allocator, Global, Layout};

/// Errors that can occur when working with tape classes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringTapeError {
    /// The string data size exceeds the maximum value representable by the offset type.
    ///
    /// This can happen when using 32-bit offsets (`CharsTapeI32`) and the total data
    /// exceeds 2GB, or when memory allocation fails.
    OffsetOverflow,
    /// Memory allocation failed.
    AllocationError,
    /// Index is out of bounds for the current number of strings.
    IndexOutOfBounds,
    /// Invalid UTF-8 sequence encountered.
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
/// * `Offset` - The offset type used for indexing (`i32` for CharsTapeI32, `i64` for CharsTapeI64)
/// * `A` - The allocator type (must implement `Allocator`). Defaults to `Global`.
///
/// # Examples
///
/// ```rust
/// use stringtape::{CharsTapeI32, StringTapeError};
///
/// // Create a new CharsTape with i32 offsets and global allocator
/// let mut tape = CharsTapeI32::new();
/// tape.push("hello")?;
/// tape.push("world")?;
///
/// assert_eq!(tape.len(), 2);
/// assert_eq!(&tape[0], "hello");
/// assert_eq!(tape.get(1), Some("world"));
/// # Ok::<(), StringTapeError>(())
/// ```
///
/// # Custom Allocators
///
/// ```rust,ignore
/// use stringtape::CharsTape;
/// use allocator_api2::alloc::{Allocator, Global};
///
/// // Use with the global allocator explicitly
/// let tape: CharsTape<i32, Global> = CharsTape::new_in(Global);
/// ```
///
/// # Memory Layout
///
/// The memory layout is compatible with Apache Arrow:
/// ```text
/// Data buffer:    [h,e,l,l,o,w,o,r,l,d]
/// Offset buffer:  [0, 5, 10]
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

/// A view into a continuous slice of a RawTape.
///
/// This provides a zero-copy view that implements the same read-only interface
/// as RawTape but cannot modify the underlying data.
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
/// This trait defines the interface for offset types that can be used to index
/// into the string data buffer. Implementations are provided for `i32` and `i64`
/// to match Apache Arrow's String and LargeString array types, and for `u32` and
/// `u64` when unsigned offsets are desired (note: Arrow interop is i32/i64-only).
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

    /// Creates a new CharsTape with pre-allocated capacity using the global allocator.
    ///
    /// Pre-allocating capacity can improve performance when you know approximately
    /// how much data you'll be storing.
    ///
    /// # Arguments
    ///
    /// * `data_capacity` - Number of bytes to pre-allocate for string data
    /// * `strings_capacity` - Number of string slots to pre-allocate
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
            let ptr = self
                .allocator
                .allocate_zeroed(new_layout)
                .map_err(|_| StringTapeError::AllocationError)?;
            ptr
        };

        self.offsets = Some(NonNull::slice_from_raw_parts(new_ptr.cast(), new_capacity));
        Ok(())
    }

    /// Adds a raw bytes slice to the end of the tape.
    ///
    /// # Errors
    ///
    /// Returns `StringTapeError::OffsetOverflow` if adding this slice would cause
    /// the total data size to exceed the maximum value representable by the offset type.
    ///
    /// Returns `StringTapeError::AllocationError` if memory allocation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::{BytesTapeI32, StringTapeError};
    ///
    /// let mut tape = BytesTapeI32::new();
    /// tape.push(b"hello")?;
    /// tape.push(&[1, 2, 3])?;
    /// assert_eq!(tape.len(), 2);
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

    /// Shortens the tape, keeping the first `len` items and dropping the rest.
    ///
    /// If `len` is greater than the current length, this has no effect.
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

    /// Extends the tape with the contents of an iterator of byte slices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::{BytesTapeI32, StringTapeError};
    ///
    /// let mut tape = BytesTapeI32::new();
    /// tape.extend([b"hello".as_slice(), b"world".as_slice()])?;
    /// assert_eq!(tape.len(), 2);
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

    /// Returns the raw parts of the tape for Apache Arrow compatibility.
    ///
    /// Returns named fields:
    /// - `data_ptr`: Data buffer pointer
    /// - `offsets_ptr`: Offsets buffer pointer
    /// - `data_len`: Data length in bytes
    /// - `items_count`: Number of items
    ///
    /// # Safety
    ///
    /// The returned pointers are valid only as long as the CharsTape is not modified.
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

    /// Helper to get offset at index
    fn get_offset(&self, index: usize) -> Result<Offset, StringTapeError> {
        if index > self.len_items {
            return Err(StringTapeError::IndexOutOfBounds);
        }

        if let Some(offsets_ptr) = self.offsets {
            unsafe { Ok(*offsets_ptr.as_ptr().cast::<Offset>().add(index)) }
        } else {
            Ok(Offset::default())
        }
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

        let base_offset = if start == 0 {
            0
        } else {
            tape.get_offset(start)?.to_usize()
        };

        let end_offset = tape.get_offset(end)?.to_usize();

        let data = unsafe {
            slice::from_raw_parts(
                data_ptr.as_ptr().cast::<u8>().add(base_offset),
                end_offset - base_offset,
            )
        };

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

        let start_offset = if index == 0 {
            0
        } else {
            (self.offsets[index] - self.offsets[0]).to_usize()
        };
        let end_offset = (self.offsets[index + 1] - self.offsets[0]).to_usize();

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
        self.data.len()
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

        let base_in_data = if start == 0 {
            0
        } else {
            (self.offsets[start] - self.offsets[0]).to_usize()
        };
        let end_in_data = (self.offsets[end] - self.offsets[0]).to_usize();

        Ok(RawTapeView {
            data: &self.data[base_in_data..end_in_data],
            offsets: &self.offsets[start..=end],
        })
    }

    /// Returns the raw parts of the view for Apache Arrow compatibility.
    pub fn as_raw_parts(&self) -> RawParts<Offset> {
        RawParts {
            data_ptr: self.data.as_ptr(),
            offsets_ptr: self.offsets.as_ptr(),
            data_len: self.data_len(),
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
        view.data
    }
}

impl<'a, Offset: OffsetType> Index<RangeFrom<usize>> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeFrom<usize>) -> &Self::Output {
        let view = self
            .subview(range.start, self.len())
            .expect("range out of bounds");
        view.data
    }
}

impl<'a, Offset: OffsetType> Index<RangeTo<usize>> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeTo<usize>) -> &Self::Output {
        let view = self.subview(0, range.end).expect("range out of bounds");
        view.data
    }
}

impl<'a, Offset: OffsetType> Index<RangeFull> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, _range: RangeFull) -> &Self::Output {
        self.data
    }
}

impl<'a, Offset: OffsetType> Index<RangeInclusive<usize>> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeInclusive<usize>) -> &Self::Output {
        let view = self
            .subview(*range.start(), range.end() + 1)
            .expect("range out of bounds");
        view.data
    }
}

impl<'a, Offset: OffsetType> Index<RangeToInclusive<usize>> for RawTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeToInclusive<usize>) -> &Self::Output {
        let view = self.subview(0, range.end + 1).expect("range out of bounds");
        view.data
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
pub type BytesTapeU32 = BytesTape<u32, Global>;
pub type BytesTapeU64 = BytesTape<u64, Global>;

pub type CharsTapeViewU32<'a> = CharsTapeView<'a, u32>;
pub type CharsTapeViewU64<'a> = CharsTapeView<'a, u64>;
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

impl<Offset: OffsetType> Default for CharsTape<Offset, Global> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
