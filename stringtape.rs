#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("README.md")]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

use core::alloc::{Allocator, Layout};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::ops::{
    Bound, Index, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};
use core::ptr::{self, NonNull};
use core::slice;

#[cfg(not(feature = "std"))]
use alloc::borrow::Cow;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::borrow::Cow;

use alloc::alloc::Global;

/// Why a tape or cows operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StringTapeError {
    /// The allocator refused a request, or the size overflowed `isize`.
    Allocation,
    /// The data outgrew what the `Offset` type addresses, such as 2 GiB for `i32`.
    OffsetOverflow,
    /// One item is longer than the cows `Length` type holds, such as 255 bytes for `u8`.
    LengthOverflow,
    /// An item given to a cows constructor lies outside its data buffer.
    #[non_exhaustive]
    OutsideData {
        /// Position of the item in the input.
        index: usize,
    },
    /// An item is not valid UTF-8.
    #[non_exhaustive]
    InvalidUtf8 {
        /// Position of the item.
        index: usize,
        /// Where within that item decoding failed.
        error: core::str::Utf8Error,
    },
    /// A reorder argument is not a permutation of `0..len()`.
    NotAPermutation,
}

impl fmt::Display for StringTapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocation => f.write_str("memory allocation failed"),
            Self::OffsetOverflow => f.write_str("data too large for the offset type"),
            Self::LengthOverflow => f.write_str("item too long for the length type"),
            Self::OutsideData { index } => write!(f, "item {index} lies outside the data buffer"),
            Self::InvalidUtf8 { index, .. } => write!(f, "item {index} is not valid UTF-8"),
            Self::NotAPermutation => f.write_str("order is not a permutation of the collection"),
        }
    }
}

impl core::error::Error for StringTapeError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::InvalidUtf8 { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Checks that every item is UTF-8, naming the first one that is not.
fn validate_utf8<'a>(items: impl IntoIterator<Item = &'a [u8]>) -> Result<(), StringTapeError> {
    for (index, item) in items.into_iter().enumerate() {
        core::str::from_utf8(item)
            .map_err(|error| StringTapeError::InvalidUtf8 { index, error })?;
    }
    Ok(())
}

/// An empty `Vec` with room for `capacity` items, or `Allocation` if the allocator refuses.
fn vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, StringTapeError> {
    let mut vec = Vec::new();
    vec.try_reserve_exact(capacity)
        .map_err(|_| StringTapeError::Allocation)?;
    Ok(vec)
}

/// The storage behind [`CharsTape`] and [`BytesTape`]: items back to back, plus offsets.
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

/// Strings stored back to back in one buffer, laid out like Apache Arrow's `String` arrays.
///
/// A second buffer holds `len() + 1` offsets, one where each string starts plus one past the
/// last, so `i32` and `i64` offsets are zero-copy compatible with `String` and `LargeString`.
///
/// ```rust
/// use stringtape::{CharsTapeI32, StringTapeError};
///
/// let mut tape = CharsTapeI32::new();
/// tape.push("hello")?;
/// tape.push("world")?;
/// assert_eq!(tape.get(0), Some("hello"));
/// assert_eq!(tape.data(), b"helloworld");
/// assert_eq!(tape.offsets(), [0, 5, 10]);
/// # Ok::<(), StringTapeError>(())
/// ```
pub struct CharsTape<Offset: OffsetType = i32, A: Allocator = Global> {
    inner: RawTape<Offset, A>,
}

/// Byte strings stored back to back in one buffer, laid out like Apache Arrow's `Binary` arrays.
pub struct BytesTape<Offset: OffsetType = i32, A: Allocator = Global> {
    inner: RawTape<Offset, A>,
}

/// Zero-copy read-only view into a RawTape slice.
#[derive(Clone, Copy)]
struct RawTapeView<'a, Offset: OffsetType> {
    data: &'a [u8],
    offsets: &'a [Offset],
}

/// A borrowed range of a [`CharsTape`], or of strings in an Arrow `String` array.
#[derive(Clone, Copy)]
pub struct CharsTapeView<'a, Offset: OffsetType = i32> {
    inner: RawTapeView<'a, Offset>,
}

/// A borrowed range of a [`BytesTape`], or of byte strings in an Arrow `Binary` array.
#[derive(Clone, Copy)]
pub struct BytesTapeView<'a, Offset: OffsetType = i32> {
    inner: RawTapeView<'a, Offset>,
}

mod sealed {
    /// Limits [`OffsetType`](super::OffsetType) and [`LengthType`](super::LengthType) to these.
    pub trait Sealed: Sized + 'static {
        /// The offsets of a tape that has not allocated yet: a lone zero.
        const ZERO: &'static [Self];
    }

    impl Sealed for u8 {
        const ZERO: &'static [Self] = &[0];
    }

    impl Sealed for u16 {
        const ZERO: &'static [Self] = &[0];
    }

    impl Sealed for u32 {
        const ZERO: &'static [Self] = &[0];
    }

    impl Sealed for u64 {
        const ZERO: &'static [Self] = &[0];
    }

    impl Sealed for i32 {
        const ZERO: &'static [Self] = &[0];
    }

    impl Sealed for i64 {
        const ZERO: &'static [Self] = &[0];
    }
}

/// The integer a tape stores its byte offsets in; sealed.
///
/// `i32` and `i64` match Apache Arrow's `String`/`Binary` and `LargeString`/`LargeBinary`,
/// while `u16`, `u32` and `u64` trade that compatibility for range.
pub trait OffsetType: sealed::Sealed + Copy {
    /// Converts a byte position, or returns `None` if it does not fit.
    fn from_usize(value: usize) -> Option<Self>;

    /// Converts back to a byte position.
    fn to_usize(self) -> usize;
}

impl OffsetType for i32 {
    fn from_usize(value: usize) -> Option<Self> {
        Self::try_from(value).ok()
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl OffsetType for i64 {
    fn from_usize(value: usize) -> Option<Self> {
        Self::try_from(value).ok()
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl OffsetType for u16 {
    fn from_usize(value: usize) -> Option<Self> {
        Self::try_from(value).ok()
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl OffsetType for u32 {
    fn from_usize(value: usize) -> Option<Self> {
        Self::try_from(value).ok()
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl OffsetType for u64 {
    fn from_usize(value: usize) -> Option<Self> {
        Self::try_from(value).ok()
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

/// The integer a cows collection stores each item's length in: `u8`, `u16`, `u32` or `u64`; sealed.
pub trait LengthType: sealed::Sealed + Copy {
    /// Converts a byte count, or returns `None` if it does not fit.
    fn from_usize(value: usize) -> Option<Self>;

    /// Converts back to a byte count.
    fn to_usize(self) -> usize;
}

impl LengthType for u8 {
    fn from_usize(value: usize) -> Option<Self> {
        Self::try_from(value).ok()
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl LengthType for u16 {
    fn from_usize(value: usize) -> Option<Self> {
        Self::try_from(value).ok()
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl LengthType for u32 {
    fn from_usize(value: usize) -> Option<Self> {
        Self::try_from(value).ok()
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl LengthType for u64 {
    fn from_usize(value: usize) -> Option<Self> {
        Self::try_from(value).ok()
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

/// Reordering shared by every cows collection.
///
/// Implemented by `CharsCows`, `BytesCows` and their `Auto` wrappers. Every operation
/// moves only the compact `(offset, length)` entries, never the payload.
///
/// The trait is what keeps the four types in step: a method added to one of them and
/// forgotten on another fails to compile rather than going unnoticed.
pub trait Reorderable {
    /// The element type yielded by the collection: `str` or `[u8]`.
    type Item: ?Sized + Ord;

    /// Sorts the elements in their natural order, in place and without allocating.
    fn sort(&mut self);

    /// Sorts the elements with a comparison function, keeping equal elements in order.
    fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&Self::Item, &Self::Item) -> core::cmp::Ordering;

    /// Sorts the elements by a key extracted from each one, keeping equal keys in order.
    fn sort_by_key<K, F>(&mut self, key_of: F)
    where
        F: FnMut(&Self::Item) -> K,
        K: Ord;

    /// Reorders the elements to match `order`, a permutation of `0..len()`.
    fn permute(&mut self, order: &[usize]) -> Result<(), StringTapeError>;
}

impl<Offset: OffsetType, A: Allocator> RawTape<Offset, A> {
    /// Creates a new, empty CharsTape with a custom allocator.
    ///
    /// This operation is O(1) and does not allocate memory until the first string is pushed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsTape;
    /// use std::alloc::Global;
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
    /// use std::alloc::Global;
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
        let data_capacity = self
            .len_bytes
            .checked_add(additional_bytes)
            .ok_or(StringTapeError::Allocation)?;
        self.grow_data(data_capacity)?;

        if additional_strings > 0 {
            let offsets_capacity = (self.len_items + 1)
                .checked_add(additional_strings)
                .ok_or(StringTapeError::Allocation)?;
            self.grow_offsets(offsets_capacity)?;
        }
        Ok(())
    }

    pub fn shrink_to_fit(&mut self) -> Result<(), StringTapeError> {
        let data_len = self.data_len();
        let offsets_len = self.len() + 1;

        if data_len < self.data_capacity() {
            self.shrink_data(data_len)?;
        }

        if offsets_len < self.offsets_capacity() {
            self.shrink_offsets(offsets_len)?;
        }

        Ok(())
    }

    fn grow_data(&mut self, new_capacity: usize) -> Result<(), StringTapeError> {
        let current_capacity = self.data_capacity();
        if new_capacity <= current_capacity {
            return Ok(());
        }

        let new_layout =
            Layout::array::<u8>(new_capacity).map_err(|_| StringTapeError::Allocation)?;

        let new_ptr = if let Some(old_ptr) = self.data {
            // Grow existing allocation
            let old_layout =
                Layout::array::<u8>(current_capacity).map_err(|_| StringTapeError::Allocation)?;
            unsafe {
                self.allocator
                    .grow(old_ptr.cast(), old_layout, new_layout)
                    .map_err(|_| StringTapeError::Allocation)?
            }
        } else {
            // Initial allocation
            self.allocator
                .allocate(new_layout)
                .map_err(|_| StringTapeError::Allocation)?
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
            Layout::array::<Offset>(new_capacity).map_err(|_| StringTapeError::Allocation)?;

        let new_ptr = if let Some(old_ptr) = self.offsets {
            // Grow existing allocation
            let old_layout = Layout::array::<Offset>(current_capacity)
                .map_err(|_| StringTapeError::Allocation)?;
            unsafe {
                self.allocator
                    .grow(old_ptr.cast(), old_layout, new_layout)
                    .map_err(|_| StringTapeError::Allocation)?
            }
        } else {
            // Initial allocation with first offset = 0
            self.allocator
                .allocate_zeroed(new_layout)
                .map_err(|_| StringTapeError::Allocation)?
        };

        self.offsets = Some(NonNull::slice_from_raw_parts(new_ptr.cast(), new_capacity));
        Ok(())
    }

    fn shrink_data(&mut self, new_capacity: usize) -> Result<(), StringTapeError> {
        let current_capacity = self.data_capacity();
        if new_capacity >= current_capacity || new_capacity == 0 {
            return Ok(());
        }

        if let Some(old_ptr) = self.data {
            let old_layout =
                Layout::array::<u8>(current_capacity).map_err(|_| StringTapeError::Allocation)?;
            let new_layout =
                Layout::array::<u8>(new_capacity).map_err(|_| StringTapeError::Allocation)?;

            let new_ptr = unsafe {
                self.allocator
                    .shrink(old_ptr.cast(), old_layout, new_layout)
                    .map_err(|_| StringTapeError::Allocation)?
            };

            self.data = Some(NonNull::slice_from_raw_parts(new_ptr.cast(), new_capacity));
        }

        Ok(())
    }

    fn shrink_offsets(&mut self, new_capacity: usize) -> Result<(), StringTapeError> {
        let current_capacity = self.offsets_capacity();
        if new_capacity >= current_capacity || new_capacity == 0 {
            return Ok(());
        }

        if let Some(old_ptr) = self.offsets {
            let old_layout = Layout::array::<Offset>(current_capacity)
                .map_err(|_| StringTapeError::Allocation)?;
            let new_layout =
                Layout::array::<Offset>(new_capacity).map_err(|_| StringTapeError::Allocation)?;

            let new_ptr = unsafe {
                self.allocator
                    .shrink(old_ptr.cast(), old_layout, new_layout)
                    .map_err(|_| StringTapeError::Allocation)?
            };

            self.offsets = Some(NonNull::slice_from_raw_parts(new_ptr.cast(), new_capacity));
        }

        Ok(())
    }

    /// Appends bytes to the tape.
    ///
    /// # Errors
    ///
    /// - `OffsetOverflow` if data size exceeds offset type maximum
    /// - `Allocation` if memory allocation fails
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
        let len_bytes = self
            .len_bytes
            .checked_add(bytes.len())
            .ok_or(StringTapeError::Allocation)?;
        // Converted before anything grows or moves, so a rejected push leaves the tape as it was.
        let offset = Offset::from_usize(len_bytes).ok_or(StringTapeError::OffsetOverflow)?;

        let current_data_capacity = self.data_capacity();
        if len_bytes > current_data_capacity {
            let new_capacity = (current_data_capacity * 2).max(len_bytes).max(64);
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

        if let Some(offsets_ptr) = self.offsets {
            unsafe {
                offsets_ptr
                    .cast::<Offset>()
                    .add(self.len_items + 1)
                    .write(offset);
            }
        }

        self.len_bytes = len_bytes;
        self.len_items += 1;
        Ok(())
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
    }

    /// Keeps the first `len` items, drops the rest.
    pub fn truncate(&mut self, len: usize) {
        if len < self.len_items {
            self.len_bytes = self.offsets()[len].to_usize();
            self.len_items = len;
        }
    }

    /// Removes the last item and returns it, or `None` if the tape is empty.
    pub fn pop(&mut self) -> Option<&[u8]> {
        let last = self.len_items.checked_sub(1)?;
        let item: *const [u8] = self.view().get(last)?;
        self.truncate(last);
        // Truncating keeps the bytes in place until the next push, which this borrow blocks.
        Some(unsafe { &*item })
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

    /// Returns the bytes of every item, back to back.
    pub fn data(&self) -> &[u8] {
        if let Some(data_ptr) = self.data {
            unsafe { core::slice::from_raw_parts(data_ptr.as_ptr().cast::<u8>(), self.len_bytes) }
        } else {
            &[]
        }
    }

    /// Returns the bytes of every item for in-place edits that keep each item's length.
    pub fn data_mut(&mut self) -> &mut [u8] {
        if let Some(data_ptr) = self.data {
            unsafe {
                core::slice::from_raw_parts_mut(data_ptr.as_ptr().cast::<u8>(), self.len_bytes)
            }
        } else {
            &mut []
        }
    }

    /// Returns the `len() + 1` offsets: where each item starts, then where the last one ends.
    pub fn offsets(&self) -> &[Offset] {
        match self.offsets {
            Some(offsets) => unsafe {
                slice::from_raw_parts(offsets.as_ptr().cast::<Offset>(), self.len_items + 1)
            },
            None => Offset::ZERO,
        }
    }

    /// Returns a reference to the allocator used by this tape.
    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    /// Creates a view of the entire tape.
    pub fn view(&self) -> RawTapeView<'_, Offset> {
        RawTapeView {
            data: self.data(),
            offsets: self.offsets(),
        }
    }
}

impl<Offset: OffsetType, A: Allocator> Drop for RawTape<Offset, A> {
    fn drop(&mut self) {
        // The layouts were valid when these blocks were allocated, so rebuilding them cannot fail.
        if let Some(data_ptr) = self.data {
            if let Ok(layout) = Layout::array::<u8>(data_ptr.len()) {
                unsafe { self.allocator.deallocate(data_ptr.cast(), layout) };
            }
        }
        if let Some(offsets_ptr) = self.offsets {
            if let Ok(layout) = Layout::array::<Offset>(offsets_ptr.len()) {
                unsafe { self.allocator.deallocate(offsets_ptr.cast(), layout) };
            }
        }
    }
}

unsafe impl<Offset: OffsetType + Send, A: Allocator + Send> Send for RawTape<Offset, A> {}
unsafe impl<Offset: OffsetType + Sync, A: Allocator + Sync> Sync for RawTape<Offset, A> {}

impl<'a, Offset: OffsetType> RawTapeView<'a, Offset> {
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
    pub fn get(&self, index: usize) -> Option<&'a [u8]> {
        let start = self.offsets.get(index)?.to_usize();
        let end = self.offsets.get(index + 1)?.to_usize();
        self.data.get(start..end)
    }

    /// Returns a reference to the item at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &'a [u8] {
        let start = self.offsets.get_unchecked(index).to_usize();
        let end = self.offsets.get_unchecked(index + 1).to_usize();
        self.data.get_unchecked(start..end)
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
        self.spanned_bytes().len()
    }

    /// Narrows the view to the items in `range`, or returns `None` if it reaches past the end.
    pub fn subview<R: RangeBounds<usize>>(&self, range: R) -> Option<Self> {
        let start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start.checked_add(1)?,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&end) => end.checked_add(1)?,
            Bound::Excluded(&end) => end,
            Bound::Unbounded => self.len(),
        };
        if start > end {
            return None;
        }
        // Keep the data pointer at the beginning of the parent tape to remain Arrow-compatible.
        Some(Self {
            data: self.data,
            offsets: self.offsets.get(start..=end)?,
        })
    }

    /// Returns the bytes of every item in this view.
    ///
    /// `data` spans the whole parent tape so the view stays Arrow-compatible, so the
    /// span has to be recovered from the first and last offsets.
    fn spanned_bytes(&self) -> &'a [u8] {
        match (self.offsets.first(), self.offsets.last()) {
            (Some(start), Some(end)) => &self.data[start.to_usize()..end.to_usize()],
            _ => &[],
        }
    }

    /// Returns the raw parts of the view for Apache Arrow compatibility.
    pub fn as_raw_parts(&self) -> RawParts<Offset> {
        // Expose an Arrow-compatible view: data_ptr remains at the tape base,
        // offsets are absolute into that buffer, and data_len reaches the last used byte.
        RawParts {
            data_ptr: self.data.as_ptr(),
            offsets_ptr: self.offsets.as_ptr(),
            data_len: self.offsets.last().map_or(0, |last| last.to_usize()),
            items_count: self.len(),
        }
    }
}

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
    pub fn get(&self, index: usize) -> Option<&'a str> {
        // Safe because CharsTapeView only comes from CharsTape which validates UTF-8
        self.inner
            .get(index)
            .map(|b| unsafe { core::str::from_utf8_unchecked(b) })
    }

    /// Returns a reference to the string at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &'a str {
        core::str::from_utf8_unchecked(self.inner.get_unchecked(index))
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

    /// Narrows the view to the strings in `range`, or returns `None` if it reaches past the end.
    pub fn subview<R: RangeBounds<usize>>(&self, range: R) -> Option<Self> {
        Some(Self {
            inner: self.inner.subview(range)?,
        })
    }

    /// Returns the raw parts of the view for Apache Arrow compatibility.
    pub fn as_raw_parts(&self) -> RawParts<Offset> {
        self.inner.as_raw_parts()
    }

    /// Returns the text of every string in this view, back to back.
    pub fn as_str(&self) -> &'a str {
        // Item boundaries are character boundaries, so the span is valid UTF-8.
        unsafe { core::str::from_utf8_unchecked(self.inner.spanned_bytes()) }
    }

    /// Returns an iterator over the strings, which borrow the tape rather than the view.
    pub fn iter(&self) -> CharsTapeViewIter<'a, Offset> {
        CharsTapeViewIter {
            view: *self,
            front: 0,
            back: self.len(),
        }
    }
}

/// Iterator over CharsTapeView strings.
#[derive(Clone, Copy)]
pub struct CharsTapeViewIter<'a, Offset: OffsetType> {
    view: CharsTapeView<'a, Offset>,
    front: usize,
    back: usize,
}

impl<'a, Offset: OffsetType> Iterator for CharsTapeViewIter<'a, Offset> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let idx = self.front;
            self.front += 1;
            Some(unsafe { self.view.get_unchecked(idx) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }
}

impl<'a, Offset: OffsetType> DoubleEndedIterator for CharsTapeViewIter<'a, Offset> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(unsafe { self.view.get_unchecked(self.back) })
    }
}

impl<'a, Offset: OffsetType> ExactSizeIterator for CharsTapeViewIter<'a, Offset> {}

impl<'a, Offset: OffsetType> FusedIterator for CharsTapeViewIter<'a, Offset> {}

impl<'a, Offset: OffsetType> fmt::Debug for CharsTapeViewIter<'a, Offset> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharsTapeViewIter")
            .field("front", &self.front)
            .field("back", &self.back)
            .field("remaining", &self.back.saturating_sub(self.front))
            .finish()
    }
}

impl<'a, Offset: OffsetType> IntoIterator for &CharsTapeView<'a, Offset> {
    type Item = &'a str;
    type IntoIter = CharsTapeViewIter<'a, Offset>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, Offset: OffsetType> fmt::Debug for CharsTapeView<'a, Offset> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharsTapeView")
            .field("len", &self.len())
            .field("data_len", &self.data_len())
            .finish()
    }
}

impl<'a, Offset: OffsetType> PartialEq for CharsTapeView<'a, Offset> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<'a, Offset: OffsetType> Eq for CharsTapeView<'a, Offset> {}

impl<'a, Offset: OffsetType> Hash for CharsTapeView<'a, Offset> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        self.iter().for_each(|item| item.hash(state));
    }
}

impl<'a, Offset: OffsetType> PartialOrd for CharsTapeView<'a, Offset> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, Offset: OffsetType> Ord for CharsTapeView<'a, Offset> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

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
    pub fn get(&self, index: usize) -> Option<&'a [u8]> {
        self.inner.get(index)
    }

    /// Returns a reference to the bytes at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &'a [u8] {
        self.inner.get_unchecked(index)
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

    /// Narrows the view to the items in `range`, or returns `None` if it reaches past the end.
    pub fn subview<R: RangeBounds<usize>>(&self, range: R) -> Option<Self> {
        Some(Self {
            inner: self.inner.subview(range)?,
        })
    }

    /// Returns the raw parts of the view for Apache Arrow compatibility.
    pub fn as_raw_parts(&self) -> RawParts<Offset> {
        self.inner.as_raw_parts()
    }

    /// Returns the bytes of every item in this view, back to back.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.inner.spanned_bytes()
    }

    /// Returns an iterator over the items, which borrow the tape rather than the view.
    pub fn iter(&self) -> BytesTapeViewIter<'a, Offset> {
        BytesTapeViewIter {
            view: *self,
            front: 0,
            back: self.len(),
        }
    }
}

/// Iterator over BytesTapeView byte slices.
#[derive(Clone, Copy)]
pub struct BytesTapeViewIter<'a, Offset: OffsetType> {
    view: BytesTapeView<'a, Offset>,
    front: usize,
    back: usize,
}

impl<'a, Offset: OffsetType> Iterator for BytesTapeViewIter<'a, Offset> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let idx = self.front;
            self.front += 1;
            Some(unsafe { self.view.get_unchecked(idx) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }
}

impl<'a, Offset: OffsetType> DoubleEndedIterator for BytesTapeViewIter<'a, Offset> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(unsafe { self.view.get_unchecked(self.back) })
    }
}

impl<'a, Offset: OffsetType> ExactSizeIterator for BytesTapeViewIter<'a, Offset> {}

impl<'a, Offset: OffsetType> FusedIterator for BytesTapeViewIter<'a, Offset> {}

impl<'a, Offset: OffsetType> fmt::Debug for BytesTapeViewIter<'a, Offset> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesTapeViewIter")
            .field("front", &self.front)
            .field("back", &self.back)
            .field("remaining", &self.back.saturating_sub(self.front))
            .finish()
    }
}

impl<'a, Offset: OffsetType> IntoIterator for &BytesTapeView<'a, Offset> {
    type Item = &'a [u8];
    type IntoIter = BytesTapeViewIter<'a, Offset>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, Offset: OffsetType> fmt::Debug for BytesTapeView<'a, Offset> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesTapeView")
            .field("len", &self.len())
            .field("data_len", &self.data_len())
            .finish()
    }
}

impl<'a, Offset: OffsetType> PartialEq for BytesTapeView<'a, Offset> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<'a, Offset: OffsetType> Eq for BytesTapeView<'a, Offset> {}

impl<'a, Offset: OffsetType> Hash for BytesTapeView<'a, Offset> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        self.iter().for_each(|item| item.hash(state));
    }
}

impl<'a, Offset: OffsetType> PartialOrd for BytesTapeView<'a, Offset> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, Offset: OffsetType> Ord for BytesTapeView<'a, Offset> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<Offset: OffsetType> CharsTape<Offset, Global> {
    /// Creates an empty tape in the global allocator; nothing is allocated until the first push.
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates an empty tape in the global allocator with room for the given bytes and strings.
    pub fn with_capacity(
        data_capacity: usize,
        strings_capacity: usize,
    ) -> Result<Self, StringTapeError> {
        Self::with_capacity_in(data_capacity, strings_capacity, Global)
    }
}

impl<Offset: OffsetType, A: Allocator> CharsTape<Offset, A> {
    /// Creates a new, empty CharsTape with a custom allocator.
    pub fn new_in(allocator: A) -> Self {
        Self {
            inner: RawTape::new_in(allocator),
        }
    }

    /// Creates a new CharsTape with pre-allocated capacity and a custom allocator.
    pub fn with_capacity_in(
        data_capacity: usize,
        strings_capacity: usize,
        allocator: A,
    ) -> Result<Self, StringTapeError> {
        Ok(Self {
            inner: RawTape::with_capacity_in(data_capacity, strings_capacity, allocator)?,
        })
    }

    /// Appends a string, leaving the tape unchanged if that fails.
    pub fn push(&mut self, s: &str) -> Result<(), StringTapeError> {
        self.inner.push(s.as_bytes())
    }

    /// Returns a reference to the string at the given index, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&str> {
        self.view().get(index)
    }

    /// Returns a reference to the string at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &str {
        self.view().get_unchecked(index)
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

    /// Returns the number of strings that can be stored without reallocating.
    ///
    /// One less than the offset slot count, which holds `len() + 1` entries.
    pub fn capacity(&self) -> usize {
        self.inner.offsets_capacity().saturating_sub(1)
    }

    /// Reserves room for `additional_bytes` more bytes and `additional_strings` more
    /// strings, sizing a cleared tape for the next batch in one step.
    pub fn reserve(
        &mut self,
        additional_bytes: usize,
        additional_strings: usize,
    ) -> Result<(), StringTapeError> {
        self.inner.reserve(additional_bytes, additional_strings)
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

    /// Returns a reference to the first string, or `None` if empty.
    pub fn first(&self) -> Option<&str> {
        self.get(0)
    }

    /// Returns a reference to the last string, or `None` if empty.
    pub fn last(&self) -> Option<&str> {
        self.get(self.len().checked_sub(1)?)
    }

    /// Removes the last string and returns it, or `None` if empty.
    pub fn pop(&mut self) -> Option<&str> {
        // Safe because CharsTape only accepts &str pushes.
        self.inner
            .pop()
            .map(|bytes| unsafe { core::str::from_utf8_unchecked(bytes) })
    }

    /// Returns true if the tape contains the given string.
    pub fn contains(&self, item: &str) -> bool {
        self.iter().any(|s| s == item)
    }

    /// Shrinks allocated capacity to fit current data.
    pub fn shrink_to_fit(&mut self) -> Result<(), StringTapeError> {
        self.inner.shrink_to_fit()
    }

    /// Appends every string an iterator yields, stopping at the first that fails.
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
        self.view().as_raw_parts()
    }

    /// Returns the bytes of every string, back to back.
    ///
    /// There is no mutable counterpart: writing bytes could break the UTF-8 invariant.
    /// Convert with [`BytesTape::from`] when raw mutation is the goal.
    pub fn data(&self) -> &[u8] {
        self.inner.data()
    }

    /// Returns the `len() + 1` offsets: where each string starts, then where the last one ends.
    pub fn offsets(&self) -> &[Offset] {
        self.inner.offsets()
    }

    /// Returns an iterator over the strings.
    pub fn iter(&self) -> CharsTapeIter<'_, Offset, A> {
        CharsTapeIter {
            tape: self,
            front: 0,
            back: self.len(),
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

    /// Views the strings in `range`, or returns `None` if it reaches past the end.
    pub fn subview<R: RangeBounds<usize>>(&self, range: R) -> Option<CharsTapeView<'_, Offset>> {
        self.view().subview(range)
    }

    /// Copies the strings into a tape with wider offsets, in a clone of the allocator.
    fn widen<Wider: OffsetType>(&self) -> Result<CharsTape<Wider, A>, StringTapeError>
    where
        A: Clone,
    {
        let mut wider =
            CharsTape::with_capacity_in(self.data_len(), self.len(), self.allocator().clone())?;
        wider.extend(self)?;
        Ok(wider)
    }
}

impl<Offset: OffsetType, A: Allocator> fmt::Debug for CharsTape<Offset, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharsTape")
            .field("len", &self.len())
            .field("data_len", &self.data_len())
            .finish()
    }
}

impl<Offset: OffsetType, A: Allocator> PartialEq for CharsTape<Offset, A> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<Offset: OffsetType, A: Allocator> Eq for CharsTape<Offset, A> {}

impl<Offset: OffsetType, A: Allocator> Hash for CharsTape<Offset, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        self.iter().for_each(|item| item.hash(state));
    }
}

impl<Offset: OffsetType, A: Allocator> PartialOrd for CharsTape<Offset, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Offset: OffsetType, A: Allocator> Ord for CharsTape<Offset, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<Offset: OffsetType> Default for CharsTape<Offset, Global> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over the strings of a [`CharsTape`].
#[derive(Clone, Copy)]
pub struct CharsTapeIter<'a, Offset: OffsetType, A: Allocator> {
    tape: &'a CharsTape<Offset, A>,
    front: usize,
    back: usize,
}

impl<'a, Offset: OffsetType, A: Allocator> Iterator for CharsTapeIter<'a, Offset, A> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let idx = self.front;
            self.front += 1;
            Some(unsafe { self.tape.get_unchecked(idx) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }
}

impl<'a, Offset: OffsetType, A: Allocator> DoubleEndedIterator for CharsTapeIter<'a, Offset, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(unsafe { self.tape.get_unchecked(self.back) })
    }
}

impl<'a, Offset: OffsetType, A: Allocator> ExactSizeIterator for CharsTapeIter<'a, Offset, A> {}

impl<'a, Offset: OffsetType, A: Allocator> FusedIterator for CharsTapeIter<'a, Offset, A> {}

impl<'a, Offset: OffsetType, A: Allocator> fmt::Debug for CharsTapeIter<'a, Offset, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharsTapeIter")
            .field("front", &self.front)
            .field("back", &self.back)
            .field("remaining", &self.back.saturating_sub(self.front))
            .finish()
    }
}

impl<'a, Offset: OffsetType, A: Allocator> IntoIterator for &'a CharsTape<Offset, A> {
    type Item = &'a str;
    type IntoIter = CharsTapeIter<'a, Offset, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<Offset: OffsetType> BytesTape<Offset, Global> {
    /// Creates an empty tape in the global allocator; nothing is allocated until the first push.
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates an empty tape in the global allocator with room for the given bytes and items.
    pub fn with_capacity(
        data_capacity: usize,
        items_capacity: usize,
    ) -> Result<Self, StringTapeError> {
        Self::with_capacity_in(data_capacity, items_capacity, Global)
    }
}

impl<Offset: OffsetType, A: Allocator> BytesTape<Offset, A> {
    /// Creates a new, empty BytesTape with a custom allocator.
    pub fn new_in(allocator: A) -> Self {
        Self {
            inner: RawTape::new_in(allocator),
        }
    }

    /// Creates a new BytesTape with pre-allocated capacity and a custom allocator.
    pub fn with_capacity_in(
        data_capacity: usize,
        items_capacity: usize,
        allocator: A,
    ) -> Result<Self, StringTapeError> {
        Ok(Self {
            inner: RawTape::with_capacity_in(data_capacity, items_capacity, allocator)?,
        })
    }

    /// Appends bytes, leaving the tape unchanged if that fails.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), StringTapeError> {
        self.inner.push(bytes)
    }

    /// Returns a reference to the bytes at the given index, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        self.view().get(index)
    }

    /// Returns a reference to the bytes at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &[u8] {
        self.view().get_unchecked(index)
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

    /// Returns the number of items that can be stored without reallocating.
    ///
    /// One less than the offset slot count, which holds `len() + 1` entries.
    pub fn capacity(&self) -> usize {
        self.inner.offsets_capacity().saturating_sub(1)
    }

    /// Reserves room for `additional_bytes` more bytes and `additional_items` more items,
    /// sizing a cleared tape for the next batch in one step.
    pub fn reserve(
        &mut self,
        additional_bytes: usize,
        additional_items: usize,
    ) -> Result<(), StringTapeError> {
        self.inner.reserve(additional_bytes, additional_items)
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

    /// Returns a reference to the first item, or `None` if empty.
    pub fn first(&self) -> Option<&[u8]> {
        self.get(0)
    }

    /// Returns a reference to the last item, or `None` if empty.
    pub fn last(&self) -> Option<&[u8]> {
        self.get(self.len().checked_sub(1)?)
    }

    /// Removes the last item and returns it, or `None` if empty.
    pub fn pop(&mut self) -> Option<&[u8]> {
        self.inner.pop()
    }

    /// Returns true if the tape contains the given byte slice.
    pub fn contains(&self, item: &[u8]) -> bool {
        self.iter().any(|bytes| bytes == item)
    }

    /// Shrinks allocated capacity to fit current data.
    pub fn shrink_to_fit(&mut self) -> Result<(), StringTapeError> {
        self.inner.shrink_to_fit()
    }

    /// Appends every item an iterator yields, stopping at the first that fails.
    pub fn extend<I>(&mut self, iter: I) -> Result<(), StringTapeError>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        self.inner.extend(iter)
    }

    /// Returns the raw parts of the tape for Apache Arrow compatibility.
    pub fn as_raw_parts(&self) -> RawParts<Offset> {
        self.view().as_raw_parts()
    }

    /// Returns the bytes of every item, back to back.
    pub fn data(&self) -> &[u8] {
        self.inner.data()
    }

    /// Returns the bytes of every item for in-place edits that keep each item's length.
    ///
    /// Offsets are untouched, so this suits length-preserving transforms applied across
    /// the whole tape at once rather than item by item.
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.inner.data_mut()
    }

    /// Returns the `len() + 1` offsets: where each item starts, then where the last one ends.
    pub fn offsets(&self) -> &[Offset] {
        self.inner.offsets()
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

    /// Returns an iterator over the items.
    pub fn iter(&self) -> BytesTapeIter<'_, Offset, A> {
        BytesTapeIter {
            tape: self,
            front: 0,
            back: self.len(),
        }
    }

    /// Views the items in `range`, or returns `None` if it reaches past the end.
    pub fn subview<R: RangeBounds<usize>>(&self, range: R) -> Option<BytesTapeView<'_, Offset>> {
        self.view().subview(range)
    }

    /// Copies the items into a tape with wider offsets, in a clone of the allocator.
    fn widen<Wider: OffsetType>(&self) -> Result<BytesTape<Wider, A>, StringTapeError>
    where
        A: Clone,
    {
        let mut wider =
            BytesTape::with_capacity_in(self.data_len(), self.len(), self.allocator().clone())?;
        wider.extend(self)?;
        Ok(wider)
    }
}

impl<Offset: OffsetType, A: Allocator> fmt::Debug for BytesTape<Offset, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesTape")
            .field("len", &self.len())
            .field("data_len", &self.data_len())
            .finish()
    }
}

impl<Offset: OffsetType, A: Allocator> PartialEq for BytesTape<Offset, A> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<Offset: OffsetType, A: Allocator> Eq for BytesTape<Offset, A> {}

impl<Offset: OffsetType, A: Allocator> Hash for BytesTape<Offset, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        self.iter().for_each(|item| item.hash(state));
    }
}

impl<Offset: OffsetType, A: Allocator> PartialOrd for BytesTape<Offset, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Offset: OffsetType, A: Allocator> Ord for BytesTape<Offset, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<Offset: OffsetType> Default for BytesTape<Offset, Global> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over the items of a [`BytesTape`].
#[derive(Clone, Copy)]
pub struct BytesTapeIter<'a, Offset: OffsetType, A: Allocator> {
    tape: &'a BytesTape<Offset, A>,
    front: usize,
    back: usize,
}

impl<'a, Offset: OffsetType, A: Allocator> Iterator for BytesTapeIter<'a, Offset, A> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let idx = self.front;
            self.front += 1;
            Some(unsafe { self.tape.get_unchecked(idx) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }
}

impl<'a, Offset: OffsetType, A: Allocator> DoubleEndedIterator for BytesTapeIter<'a, Offset, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(unsafe { self.tape.get_unchecked(self.back) })
    }
}

impl<'a, Offset: OffsetType, A: Allocator> ExactSizeIterator for BytesTapeIter<'a, Offset, A> {}

impl<'a, Offset: OffsetType, A: Allocator> FusedIterator for BytesTapeIter<'a, Offset, A> {}

impl<'a, Offset: OffsetType, A: Allocator> fmt::Debug for BytesTapeIter<'a, Offset, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesTapeIter")
            .field("front", &self.front)
            .field("back", &self.back)
            .field("remaining", &self.back.saturating_sub(self.front))
            .finish()
    }
}

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

impl<Offset: OffsetType, A: Allocator> TryFrom<BytesTape<Offset, A>> for CharsTape<Offset, A> {
    type Error = StringTapeError;

    fn try_from(bytes_tape: BytesTape<Offset, A>) -> Result<Self, Self::Error> {
        validate_utf8(&bytes_tape)?;
        Ok(CharsTape {
            inner: bytes_tape.inner,
        })
    }
}

impl<Offset: OffsetType, A: Allocator> From<CharsTape<Offset, A>> for BytesTape<Offset, A> {
    fn from(chars_tape: CharsTape<Offset, A>) -> Self {
        BytesTape {
            inner: chars_tape.inner,
        }
    }
}

impl<'a, Offset: OffsetType> TryFrom<BytesTapeView<'a, Offset>> for CharsTapeView<'a, Offset> {
    type Error = StringTapeError;

    fn try_from(bytes_view: BytesTapeView<'a, Offset>) -> Result<Self, Self::Error> {
        validate_utf8(&bytes_view)?;
        Ok(CharsTapeView {
            inner: bytes_view.inner,
        })
    }
}

impl<'a, Offset: OffsetType> From<CharsTapeView<'a, Offset>> for BytesTapeView<'a, Offset> {
    fn from(chars_view: CharsTapeView<'a, Offset>) -> Self {
        BytesTapeView {
            inner: chars_view.inner,
        }
    }
}

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

impl<Offset: OffsetType, Length: LengthType> PackedEntry<Offset, Length> {
    /// Returns the half-open byte range this entry points at.
    ///
    /// Taken by value because the struct is packed, so its fields cannot be borrowed.
    #[inline]
    fn range(self) -> Range<usize> {
        let start = self.offset.to_usize();
        start..start + self.length.to_usize()
    }
}

/// Returns the bytes one entry points at.
#[inline]
fn slice_at<Offset: OffsetType, Length: LengthType>(
    data: &[u8],
    entry: PackedEntry<Offset, Length>,
) -> &[u8] {
    &data[entry.range()]
}

/// Returns the string one entry points at.
///
/// # Safety
///
/// The entry must span validated UTF-8, which every constructor of [`CharsCows`]
/// guarantees.
#[inline]
unsafe fn str_at<Offset: OffsetType, Length: LengthType>(
    data: &[u8],
    entry: PackedEntry<Offset, Length>,
) -> &str {
    core::str::from_utf8_unchecked(slice_at(data, entry))
}

/// Locates `item` inside `data` as an `(offset, length)` pair.
///
/// A slice starting before `data` wraps to an offset past its end, which
/// [`entries_from_offsets`] then rejects like any other span outside the buffer.
fn span_in(data: &[u8], item: &[u8]) -> (usize, usize) {
    let offset = (item.as_ptr() as usize).wrapping_sub(data.as_ptr() as usize);
    (offset, item.len())
}

/// Builds entries from explicit `(offset, length)` pairs, bounded by `data_len`.
fn entries_from_offsets<Offset: OffsetType, Length: LengthType, I>(
    iter: I,
    data_len: usize,
) -> Result<Vec<PackedEntry<Offset, Length>>, StringTapeError>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    let mut entries = Vec::new();
    for (index, (offset, length)) in iter.into_iter().enumerate() {
        if offset.checked_add(length).is_none_or(|end| end > data_len) {
            return Err(StringTapeError::OutsideData { index });
        }
        entries
            .try_reserve(1)
            .map_err(|_| StringTapeError::Allocation)?;
        entries.push(PackedEntry {
            offset: Offset::from_usize(offset).ok_or(StringTapeError::OffsetOverflow)?,
            length: Length::from_usize(length).ok_or(StringTapeError::LengthOverflow)?,
        });
    }
    Ok(entries)
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
pub struct CharsCows<'a, Offset: OffsetType = u32, Length: LengthType = u16> {
    data: Cow<'a, [u8]>,
    entries: Vec<PackedEntry<Offset, Length>>,
}

/// A memory-efficient collection of byte slices with configurable offset and length types.
///
/// Similar to `CharsCows` but for arbitrary binary data without UTF-8 validation.
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
    /// - `OutsideData` if a slice is not within the data buffer
    /// - `OffsetOverflow` or `LengthOverflow` if a slice's position or length does not fit
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
        let spans = iter
            .into_iter()
            .map(|item| span_in(&data, item.as_ref().as_bytes()));
        let entries = entries_from_offsets(spans, data.len())?;
        Ok(Self { data, entries })
    }

    /// Creates a CharsCows from an iterator of `(offset, length)` pairs and a data buffer.
    ///
    /// Each span is validated as UTF-8, since every accessor on this type assumes it.
    ///
    /// # Errors
    ///
    /// - `OutsideData` if a span reaches past the data buffer
    /// - `OffsetOverflow` or `LengthOverflow` if a span's position or length does not fit
    /// - `InvalidUtf8` if a span is not valid UTF-8
    pub fn from_offsets_and_data<I>(iter: I, data: Cow<'a, [u8]>) -> Result<Self, StringTapeError>
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        BytesCows::from_offsets_and_data(iter, data)?.try_into()
    }

    /// Returns a reference to the string at the given index, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&'_ str> {
        // Safety: UTF-8 validated during construction
        self.entries
            .get(index)
            .map(|entry| unsafe { str_at(&self.data, *entry) })
    }

    /// Returns a reference to the string at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &str {
        str_at(&self.data, *self.entries.get_unchecked(index))
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
            front: 0,
            back: self.len(),
        }
    }

    /// Returns a reference to the underlying data buffer.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Sorts the slices in their natural order, in place and without allocating.
    ///
    /// Equal slices hold the same bytes, so this matches a stable sort's result.
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
    pub fn sort(&mut self) {
        let data = &self.data;
        self.entries
            .sort_unstable_by(|first, second| slice_at(data, *first).cmp(slice_at(data, *second)));
    }

    /// Sorts the slices in-place with a comparison function, keeping equal slices in order.
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
    {
        let data = &self.data;
        self.entries.sort_by(|first, second| unsafe {
            compare(str_at(data, *first), str_at(data, *second))
        });
    }

    /// Sorts the slices in-place by a key extracted from each one, keeping equal keys in order.
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
    pub fn sort_by_key<K, F>(&mut self, mut key_of: F)
    where
        F: FnMut(&str) -> K,
        K: Ord,
    {
        let data = &self.data;
        self.entries
            .sort_by_key(|entry| key_of(unsafe { str_at(data, *entry) }));
    }

    /// Reorders the slices to match `order`, a permutation of `0..len()`.
    ///
    /// Pairs with an external solver: a SIMD argument sort such as
    /// `stringzilla::sz::argsort_by` computes the order, and this applies it. Only the
    /// compact entries move.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::CharsCowsU32U8;
    /// use std::borrow::Cow;
    ///
    /// let data = "alpha beta gamma";
    /// let mut cows = CharsCowsU32U8::from_iter_and_data(
    ///     data.split_whitespace(),
    ///     Cow::Borrowed(data.as_bytes())
    /// ).unwrap();
    ///
    /// cows.permute(&[2, 0, 1]).unwrap();
    /// let reordered: Vec<&str> = cows.iter().collect();
    /// assert_eq!(reordered, vec!["gamma", "alpha", "beta"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn permute(&mut self, order: &[usize]) -> Result<(), StringTapeError> {
        permute_entries(&mut self.entries, order)
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
        let spans = iter.into_iter().map(|item| span_in(&data, item.as_ref()));
        let entries = entries_from_offsets(spans, data.len())?;
        Ok(Self { data, entries })
    }

    /// Creates BytesCows from iterator of (offset, length) pairs and data buffer.
    pub fn from_offsets_and_data<I>(iter: I, data: Cow<'a, [u8]>) -> Result<Self, StringTapeError>
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        let entries = entries_from_offsets(iter, data.len())?;
        Ok(Self { data, entries })
    }

    /// Returns a reference to the bytes at the given index, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        self.entries
            .get(index)
            .map(|entry| slice_at(&self.data, *entry))
    }

    /// Returns a reference to the bytes at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &[u8] {
        let entry = *self.entries.get_unchecked(index);
        self.data.get_unchecked(entry.range())
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
            front: 0,
            back: self.len(),
        }
    }

    /// Returns a reference to the underlying data buffer.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Sorts the slices in byte-lexicographic order, in place and without allocating.
    ///
    /// Only the compact entries move; the payload buffer is never touched.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::BytesCowsU32U16;
    /// use std::borrow::Cow;
    ///
    /// let data = b"zebra apple banana";
    /// let mut cows = BytesCowsU32U16::from_iter_and_data(
    ///     data.split(|&b| b == b' '),
    ///     Cow::Borrowed(&data[..])
    /// ).unwrap();
    ///
    /// cows.sort();
    /// let sorted: Vec<&[u8]> = cows.iter().collect();
    /// assert_eq!(sorted, vec![&b"apple"[..], b"banana", b"zebra"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn sort(&mut self) {
        let data = &self.data;
        self.entries
            .sort_unstable_by(|first, second| slice_at(data, *first).cmp(slice_at(data, *second)));
    }

    /// Sorts the slices in-place with a comparison function, keeping equal slices in order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::BytesCowsU32U16;
    /// use std::borrow::Cow;
    ///
    /// let data = b"aaa bb c";
    /// let mut cows = BytesCowsU32U16::from_iter_and_data(
    ///     data.split(|&b| b == b' '),
    ///     Cow::Borrowed(&data[..])
    /// ).unwrap();
    ///
    /// cows.sort_by(|first, second| first.len().cmp(&second.len()));
    /// let sorted: Vec<&[u8]> = cows.iter().collect();
    /// assert_eq!(sorted, vec![&b"c"[..], b"bb", b"aaa"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn sort_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&[u8], &[u8]) -> core::cmp::Ordering,
    {
        let data = &self.data;
        self.entries
            .sort_by(|first, second| compare(slice_at(data, *first), slice_at(data, *second)));
    }

    /// Sorts the slices in-place by a key extracted from each one, keeping equal keys in order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::BytesCowsU32U16;
    /// use std::borrow::Cow;
    ///
    /// let data = b"aaa bb c";
    /// let mut cows = BytesCowsU32U16::from_iter_and_data(
    ///     data.split(|&b| b == b' '),
    ///     Cow::Borrowed(&data[..])
    /// ).unwrap();
    ///
    /// cows.sort_by_key(|slice| slice.len());
    /// let sorted: Vec<&[u8]> = cows.iter().collect();
    /// assert_eq!(sorted, vec![&b"c"[..], b"bb", b"aaa"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn sort_by_key<K, F>(&mut self, mut key_of: F)
    where
        F: FnMut(&[u8]) -> K,
        K: Ord,
    {
        let data = &self.data;
        self.entries
            .sort_by_key(|entry| key_of(slice_at(data, *entry)));
    }

    /// Reorders the slices to match `order`, a permutation of `0..len()`.
    ///
    /// Pairs with an external solver: a SIMD argument sort such as
    /// `stringzilla::sz::argsort_by` computes the order, and this applies it. Only the
    /// compact entries move.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::BytesCowsU32U16;
    /// use std::borrow::Cow;
    ///
    /// let data = b"alpha beta gamma";
    /// let mut cows = BytesCowsU32U16::from_iter_and_data(
    ///     data.split(|&b| b == b' '),
    ///     Cow::Borrowed(&data[..])
    /// ).unwrap();
    ///
    /// cows.permute(&[2, 0, 1]).unwrap();
    /// let reordered: Vec<&[u8]> = cows.iter().collect();
    /// assert_eq!(reordered, vec![&b"gamma"[..], b"alpha", b"beta"]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn permute(&mut self, order: &[usize]) -> Result<(), StringTapeError> {
        permute_entries(&mut self.entries, order)
    }
}

/// Applies `order` to `entries`, rejecting anything that is not a permutation.
///
/// A repeated index would silently duplicate one slice and drop another, so the check is
/// unconditional — its bitmap is smaller than the entry buffer already being allocated.
fn permute_entries<Offset: OffsetType, Length: LengthType>(
    entries: &mut Vec<PackedEntry<Offset, Length>>,
    order: &[usize],
) -> Result<(), StringTapeError> {
    if order.len() != entries.len() {
        return Err(StringTapeError::NotAPermutation);
    }

    let mut seen = vec_with_capacity(order.len())?;
    seen.resize(order.len(), false);
    let mut permuted = vec_with_capacity(order.len())?;
    for &index in order {
        let slot = seen
            .get_mut(index)
            .ok_or(StringTapeError::NotAPermutation)?;
        if core::mem::replace(slot, true) {
            return Err(StringTapeError::NotAPermutation);
        }
        permuted.push(entries[index]);
    }
    *entries = permuted;
    Ok(())
}

/// Iterator over the strings of a [`CharsCows`].
#[derive(Clone, Copy)]
pub struct CharsCowsIter<'a, Offset: OffsetType, Length: LengthType> {
    slices: &'a CharsCows<'a, Offset, Length>,
    front: usize,
    back: usize,
}

impl<'a, Offset: OffsetType, Length: LengthType> Iterator for CharsCowsIter<'a, Offset, Length> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let idx = self.front;
            self.front += 1;
            Some(unsafe { self.slices.get_unchecked(idx) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> DoubleEndedIterator
    for CharsCowsIter<'a, Offset, Length>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(unsafe { self.slices.get_unchecked(self.back) })
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> ExactSizeIterator
    for CharsCowsIter<'a, Offset, Length>
{
}

impl<'a, Offset: OffsetType, Length: LengthType> FusedIterator
    for CharsCowsIter<'a, Offset, Length>
{
}

impl<'a, Offset: OffsetType, Length: LengthType> fmt::Debug for CharsCowsIter<'a, Offset, Length> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharsCowsIter")
            .field("front", &self.front)
            .field("back", &self.back)
            .field("remaining", &self.back.saturating_sub(self.front))
            .finish()
    }
}

/// Iterator over the items of a [`BytesCows`].
#[derive(Clone, Copy)]
pub struct BytesCowsIter<'a, Offset: OffsetType, Length: LengthType> {
    slices: &'a BytesCows<'a, Offset, Length>,
    front: usize,
    back: usize,
}

impl<'a, Offset: OffsetType, Length: LengthType> Iterator for BytesCowsIter<'a, Offset, Length> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let idx = self.front;
            self.front += 1;
            Some(unsafe { self.slices.get_unchecked(idx) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> DoubleEndedIterator
    for BytesCowsIter<'a, Offset, Length>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(unsafe { self.slices.get_unchecked(self.back) })
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> ExactSizeIterator
    for BytesCowsIter<'a, Offset, Length>
{
}

impl<'a, Offset: OffsetType, Length: LengthType> FusedIterator
    for BytesCowsIter<'a, Offset, Length>
{
}

impl<'a, Offset: OffsetType, Length: LengthType> fmt::Debug for BytesCowsIter<'a, Offset, Length> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesCowsIter")
            .field("front", &self.front)
            .field("back", &self.back)
            .field("remaining", &self.back.saturating_sub(self.front))
            .finish()
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> Reorderable for CharsCows<'a, Offset, Length> {
    type Item = str;

    fn sort(&mut self) {
        CharsCows::sort(self)
    }

    fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&str, &str) -> core::cmp::Ordering,
    {
        CharsCows::sort_by(self, compare)
    }

    fn sort_by_key<K, F>(&mut self, key_of: F)
    where
        F: FnMut(&str) -> K,
        K: Ord,
    {
        CharsCows::sort_by_key(self, key_of)
    }

    fn permute(&mut self, order: &[usize]) -> Result<(), StringTapeError> {
        CharsCows::permute(self, order)
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> Reorderable for BytesCows<'a, Offset, Length> {
    type Item = [u8];

    fn sort(&mut self) {
        BytesCows::sort(self)
    }

    fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&[u8], &[u8]) -> core::cmp::Ordering,
    {
        BytesCows::sort_by(self, compare)
    }

    fn sort_by_key<K, F>(&mut self, key_of: F)
    where
        F: FnMut(&[u8]) -> K,
        K: Ord,
    {
        BytesCows::sort_by_key(self, key_of)
    }

    fn permute(&mut self, order: &[usize]) -> Result<(), StringTapeError> {
        BytesCows::permute(self, order)
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

impl<'a, Offset: OffsetType, Length: LengthType> fmt::Debug for CharsCows<'a, Offset, Length> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharsCows")
            .field("len", &self.len())
            .finish()
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> PartialEq for CharsCows<'a, Offset, Length> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> Eq for CharsCows<'a, Offset, Length> {}

impl<'a, Offset: OffsetType, Length: LengthType> Hash for CharsCows<'a, Offset, Length> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        self.iter().for_each(|item| item.hash(state));
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> PartialOrd for CharsCows<'a, Offset, Length> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> Ord for CharsCows<'a, Offset, Length> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> fmt::Debug for BytesCows<'a, Offset, Length> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesCows")
            .field("len", &self.len())
            .finish()
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> PartialEq for BytesCows<'a, Offset, Length> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> Eq for BytesCows<'a, Offset, Length> {}

impl<'a, Offset: OffsetType, Length: LengthType> Hash for BytesCows<'a, Offset, Length> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        self.iter().for_each(|item| item.hash(state));
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> PartialOrd for BytesCows<'a, Offset, Length> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> Ord for BytesCows<'a, Offset, Length> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> TryFrom<BytesCows<'a, Offset, Length>>
    for CharsCows<'a, Offset, Length>
{
    type Error = StringTapeError;

    fn try_from(bytes: BytesCows<'a, Offset, Length>) -> Result<Self, Self::Error> {
        validate_utf8(&bytes)?;
        Ok(CharsCows {
            data: bytes.data,
            entries: bytes.entries,
        })
    }
}

impl<'a, Offset: OffsetType, Length: LengthType> From<CharsCows<'a, Offset, Length>>
    for BytesCows<'a, Offset, Length>
{
    fn from(chars: CharsCows<'a, Offset, Length>) -> Self {
        BytesCows {
            data: chars.data,
            entries: chars.entries,
        }
    }
}

/// Borrows the buffer and copies the entries, once every item is validated as UTF-8.
impl<'b, Offset: OffsetType, Length: LengthType> TryFrom<&'b BytesCows<'_, Offset, Length>>
    for CharsCows<'b, Offset, Length>
{
    type Error = StringTapeError;

    fn try_from(bytes: &'b BytesCows<'_, Offset, Length>) -> Result<Self, Self::Error> {
        validate_utf8(bytes)?;
        let mut entries = vec_with_capacity(bytes.entries.len())?;
        entries.extend_from_slice(&bytes.entries);
        Ok(CharsCows {
            data: Cow::Borrowed(bytes.data()),
            entries,
        })
    }
}

/// Borrows the buffer and copies the entries.
///
/// ```rust
/// use stringtape::{BytesCowsU32U8, CharsCowsU32U8, StringTapeError};
/// use std::borrow::Cow;
///
/// let data = "hello world";
/// let chars =
///     CharsCowsU32U8::from_iter_and_data(data.split(' '), Cow::Borrowed(data.as_bytes()))?;
/// let bytes = BytesCowsU32U8::try_from(&chars)?;
/// assert_eq!(bytes.get(1), Some(&b"world"[..]));
/// # Ok::<(), StringTapeError>(())
/// ```
impl<'b, Offset: OffsetType, Length: LengthType> TryFrom<&'b CharsCows<'_, Offset, Length>>
    for BytesCows<'b, Offset, Length>
{
    type Error = StringTapeError;

    fn try_from(chars: &'b CharsCows<'_, Offset, Length>) -> Result<Self, Self::Error> {
        let mut entries = vec_with_capacity(chars.entries.len())?;
        entries.extend_from_slice(&chars.entries);
        Ok(BytesCows {
            data: Cow::Borrowed(chars.data()),
            entries,
        })
    }
}

/// Indexes the tape's strings in place, so they can be reordered without moving their bytes.
///
/// ```rust
/// use stringtape::{CharsCows, CharsTapeI32, StringTapeError};
///
/// let mut tape = CharsTapeI32::new();
/// tape.extend(["banana", "apple", "cherry"])?;
///
/// let mut cows = CharsCows::<i32, u16>::try_from(&tape)?;
/// cows.sort();
/// assert_eq!(cows.get(0), Some("apple"));
/// # Ok::<(), StringTapeError>(())
/// ```
impl<'a, Offset: OffsetType, Length: LengthType, A: Allocator> TryFrom<&'a CharsTape<Offset, A>>
    for CharsCows<'a, Offset, Length>
{
    type Error = StringTapeError;

    fn try_from(tape: &'a CharsTape<Offset, A>) -> Result<Self, Self::Error> {
        Self::from_iter_and_data(tape, Cow::Borrowed(tape.data()))
    }
}

/// Indexes the tape's items in place, so they can be reordered without moving their bytes.
///
/// ```rust
/// use stringtape::{BytesCows, BytesTapeI32, StringTapeError};
///
/// let mut tape = BytesTapeI32::new();
/// tape.extend([&[7, 8, 9][..], &[1, 2, 3]])?;
///
/// let mut cows = BytesCows::<i32, u16>::try_from(&tape)?;
/// cows.sort();
/// assert_eq!(cows.get(0), Some(&[1, 2, 3][..]));
/// # Ok::<(), StringTapeError>(())
/// ```
impl<'a, Offset: OffsetType, Length: LengthType, A: Allocator> TryFrom<&'a BytesTape<Offset, A>>
    for BytesCows<'a, Offset, Length>
{
    type Error = StringTapeError;

    fn try_from(tape: &'a BytesTape<Offset, A>) -> Result<Self, Self::Error> {
        Self::from_iter_and_data(tape, Cow::Borrowed(tape.data()))
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
    /// Two passes: the first, over a clone of the iterator, finds the longest word, and the
    /// second builds entries of the narrowest types that fit.
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
    /// assert_eq!(cows.type_name(), "CharsCows<u32, u8>");
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
        let needs_u64_offset = data_len > u32::MAX as usize || max_word_len > u32::MAX as usize;

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

    /// Returns a reference to the string at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &str {
        match self {
            Self::U32U8(s) => s.get_unchecked(index),
            Self::U32U16(s) => s.get_unchecked(index),
            Self::U32U32(s) => s.get_unchecked(index),
            Self::U64U8(s) => s.get_unchecked(index),
            Self::U64U16(s) => s.get_unchecked(index),
            Self::U64U32(s) => s.get_unchecked(index),
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
            front: 0,
            back: self.len(),
        }
    }

    /// Sorts the slices in their natural order, in place and without allocating.
    ///
    /// Equal slices hold the same bytes, so this matches a stable sort's result.
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

    /// Sorts the slices in-place with a comparison function, keeping equal slices in order.
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

    /// Sorts the slices in-place by a key extracted from each one, keeping equal keys in order.
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

    /// Reorders the slices to match `order`, a permutation of `0..len()`.
    pub fn permute(&mut self, order: &[usize]) -> Result<(), StringTapeError> {
        match self {
            Self::U32U8(s) => s.permute(order),
            Self::U32U16(s) => s.permute(order),
            Self::U32U32(s) => s.permute(order),
            Self::U64U8(s) => s.permute(order),
            Self::U64U16(s) => s.permute(order),
            Self::U64U32(s) => s.permute(order),
        }
    }

    /// Returns a reference to the underlying data buffer.
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
    /// assert_eq!(cows.data(), b"hello world");
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn data(&self) -> &[u8] {
        match self {
            Self::U32U8(s) => s.data(),
            Self::U32U16(s) => s.data(),
            Self::U32U32(s) => s.data(),
            Self::U64U8(s) => s.data(),
            Self::U64U16(s) => s.data(),
            Self::U64U32(s) => s.data(),
        }
    }
}

/// Iterator over CharsCowsAuto string cows.
#[derive(Clone, Copy)]
pub struct CharsCowsAutoIter<'a> {
    inner: &'a CharsCowsAuto<'a>,
    front: usize,
    back: usize,
}

impl<'a> Iterator for CharsCowsAutoIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let idx = self.front;
            self.front += 1;
            Some(unsafe { self.inner.get_unchecked(idx) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }
}

impl<'a> DoubleEndedIterator for CharsCowsAutoIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(unsafe { self.inner.get_unchecked(self.back) })
    }
}

impl<'a> ExactSizeIterator for CharsCowsAutoIter<'a> {}

impl<'a> FusedIterator for CharsCowsAutoIter<'a> {}

impl<'a> fmt::Debug for CharsCowsAutoIter<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharsCowsAutoIter")
            .field("front", &self.front)
            .field("back", &self.back)
            .field("remaining", &self.back.saturating_sub(self.front))
            .finish()
    }
}

impl<'a> IntoIterator for &'a CharsCowsAuto<'a> {
    type Item = &'a str;
    type IntoIter = CharsCowsAutoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> fmt::Debug for CharsCowsAuto<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharsCowsAuto")
            .field("len", &self.len())
            .finish()
    }
}

impl<'a> PartialEq for CharsCowsAuto<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<'a> Eq for CharsCowsAuto<'a> {}

impl<'a> Hash for CharsCowsAuto<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        self.iter().for_each(|item| item.hash(state));
    }
}

impl<'a> PartialOrd for CharsCowsAuto<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for CharsCowsAuto<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

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

        let needs_u64_offset = data_len > u32::MAX as usize || max_len > u32::MAX as usize;

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

    /// Returns a reference to the bytes at the given index.
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

    /// Returns a reference to the bytes at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &[u8] {
        match self {
            Self::U32U8(s) => s.get_unchecked(index),
            Self::U32U16(s) => s.get_unchecked(index),
            Self::U32U32(s) => s.get_unchecked(index),
            Self::U64U8(s) => s.get_unchecked(index),
            Self::U64U16(s) => s.get_unchecked(index),
            Self::U64U32(s) => s.get_unchecked(index),
        }
    }

    /// Returns a reference to the underlying data buffer.
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
    /// assert_eq!(bytes.data(), b"hello world");
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn data(&self) -> &[u8] {
        match self {
            Self::U32U8(s) => s.data(),
            Self::U32U16(s) => s.data(),
            Self::U32U32(s) => s.data(),
            Self::U64U8(s) => s.data(),
            Self::U64U16(s) => s.data(),
            Self::U64U32(s) => s.data(),
        }
    }

    /// Returns the bytes each entry occupies for the selected variant.
    pub fn bytes_per_entry(&self) -> usize {
        match self {
            Self::U32U8(_) => 4 + 1,
            Self::U32U16(_) => 4 + 2,
            Self::U32U32(_) => 4 + 4,
            Self::U64U8(_) => 8 + 1,
            Self::U64U16(_) => 8 + 2,
            Self::U64U32(_) => 8 + 4,
        }
    }

    /// Returns the name of the selected variant.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U32U8(_) => "BytesCows<u32, u8>",
            Self::U32U16(_) => "BytesCows<u32, u16>",
            Self::U32U32(_) => "BytesCows<u32, u32>",
            Self::U64U8(_) => "BytesCows<u64, u8>",
            Self::U64U16(_) => "BytesCows<u64, u16>",
            Self::U64U32(_) => "BytesCows<u64, u32>",
        }
    }

    /// Sorts the slices in byte-lexicographic order, in place and without allocating.
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

    /// Sorts the slices in-place with a comparison function, keeping equal slices in order.
    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&[u8], &[u8]) -> core::cmp::Ordering,
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

    /// Sorts the slices in-place by a key extracted from each one, keeping equal keys in order.
    pub fn sort_by_key<K, F>(&mut self, key_of: F)
    where
        F: FnMut(&[u8]) -> K,
        K: Ord,
    {
        match self {
            Self::U32U8(s) => s.sort_by_key(key_of),
            Self::U32U16(s) => s.sort_by_key(key_of),
            Self::U32U32(s) => s.sort_by_key(key_of),
            Self::U64U8(s) => s.sort_by_key(key_of),
            Self::U64U16(s) => s.sort_by_key(key_of),
            Self::U64U32(s) => s.sort_by_key(key_of),
        }
    }

    /// Reorders the slices to match `order`, a permutation of `0..len()`.
    pub fn permute(&mut self, order: &[usize]) -> Result<(), StringTapeError> {
        match self {
            Self::U32U8(s) => s.permute(order),
            Self::U32U16(s) => s.permute(order),
            Self::U32U32(s) => s.permute(order),
            Self::U64U8(s) => s.permute(order),
            Self::U64U16(s) => s.permute(order),
            Self::U64U32(s) => s.permute(order),
        }
    }

    /// Returns an iterator over the byte cows.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::BytesCowsAuto;
    /// use std::borrow::Cow;
    ///
    /// let data = b"hello world foo";
    /// let bytes = BytesCowsAuto::from_iter_and_data(
    ///     data.split(|&b| b == b' '),
    ///     Cow::Borrowed(&data[..])
    /// ).unwrap();
    ///
    /// let slices: Vec<&[u8]> = bytes.iter().collect();
    /// assert_eq!(slices, vec![&b"hello"[..], &b"world"[..], &b"foo"[..]]);
    /// # Ok::<(), stringtape::StringTapeError>(())
    /// ```
    pub fn iter(&self) -> BytesCowsAutoIter<'_> {
        BytesCowsAutoIter {
            inner: self,
            front: 0,
            back: self.len(),
        }
    }
}

/// Iterator over BytesCowsAuto byte cows.
#[derive(Clone, Copy)]
pub struct BytesCowsAutoIter<'a> {
    inner: &'a BytesCowsAuto<'a>,
    front: usize,
    back: usize,
}

impl<'a> Iterator for BytesCowsAutoIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let idx = self.front;
            self.front += 1;
            Some(unsafe { self.inner.get_unchecked(idx) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }
}

impl<'a> DoubleEndedIterator for BytesCowsAutoIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(unsafe { self.inner.get_unchecked(self.back) })
    }
}

impl<'a> ExactSizeIterator for BytesCowsAutoIter<'a> {}

impl<'a> FusedIterator for BytesCowsAutoIter<'a> {}

impl<'a> fmt::Debug for BytesCowsAutoIter<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesCowsAutoIter")
            .field("front", &self.front)
            .field("back", &self.back)
            .field("remaining", &self.back.saturating_sub(self.front))
            .finish()
    }
}

impl<'a> IntoIterator for &'a BytesCowsAuto<'a> {
    type Item = &'a [u8];
    type IntoIter = BytesCowsAutoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> Reorderable for CharsCowsAuto<'a> {
    type Item = str;

    fn sort(&mut self) {
        CharsCowsAuto::sort(self)
    }

    fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&str, &str) -> core::cmp::Ordering,
    {
        CharsCowsAuto::sort_by(self, compare)
    }

    fn sort_by_key<K, F>(&mut self, key_of: F)
    where
        F: FnMut(&str) -> K,
        K: Ord,
    {
        CharsCowsAuto::sort_by_key(self, key_of)
    }

    fn permute(&mut self, order: &[usize]) -> Result<(), StringTapeError> {
        CharsCowsAuto::permute(self, order)
    }
}

impl<'a> Reorderable for BytesCowsAuto<'a> {
    type Item = [u8];

    fn sort(&mut self) {
        BytesCowsAuto::sort(self)
    }

    fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&[u8], &[u8]) -> core::cmp::Ordering,
    {
        BytesCowsAuto::sort_by(self, compare)
    }

    fn sort_by_key<K, F>(&mut self, key_of: F)
    where
        F: FnMut(&[u8]) -> K,
        K: Ord,
    {
        BytesCowsAuto::sort_by_key(self, key_of)
    }

    fn permute(&mut self, order: &[usize]) -> Result<(), StringTapeError> {
        BytesCowsAuto::permute(self, order)
    }
}

impl<'a> fmt::Debug for BytesCowsAuto<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesCowsAuto")
            .field("len", &self.len())
            .finish()
    }
}

impl<'a> PartialEq for BytesCowsAuto<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<'a> Eq for BytesCowsAuto<'a> {}

impl<'a> Hash for BytesCowsAuto<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        self.iter().for_each(|item| item.hash(state));
    }
}

impl<'a> PartialOrd for BytesCowsAuto<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for BytesCowsAuto<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

/// Borrows the buffer and copies the entries.
impl<'b> TryFrom<&'b CharsCowsAuto<'_>> for BytesCowsAuto<'b> {
    type Error = StringTapeError;

    fn try_from(chars: &'b CharsCowsAuto<'_>) -> Result<Self, Self::Error> {
        Ok(match chars {
            CharsCowsAuto::U32U8(cows) => Self::U32U8(cows.try_into()?),
            CharsCowsAuto::U32U16(cows) => Self::U32U16(cows.try_into()?),
            CharsCowsAuto::U32U32(cows) => Self::U32U32(cows.try_into()?),
            CharsCowsAuto::U64U8(cows) => Self::U64U8(cows.try_into()?),
            CharsCowsAuto::U64U16(cows) => Self::U64U16(cows.try_into()?),
            CharsCowsAuto::U64U32(cows) => Self::U64U32(cows.try_into()?),
        })
    }
}

/// Borrows the buffer and copies the entries, once every item is validated as UTF-8.
///
/// ```rust
/// use stringtape::{BytesCowsAuto, CharsCowsAuto, StringTapeError};
/// use std::borrow::Cow;
///
/// let data = b"hello world";
/// let bytes =
///     BytesCowsAuto::from_iter_and_data(data.split(|&b| b == b' '), Cow::Borrowed(&data[..]))?;
/// let chars = CharsCowsAuto::try_from(&bytes)?;
/// assert_eq!(chars.get(1), Some("world"));
/// # Ok::<(), StringTapeError>(())
/// ```
impl<'b> TryFrom<&'b BytesCowsAuto<'_>> for CharsCowsAuto<'b> {
    type Error = StringTapeError;

    fn try_from(bytes: &'b BytesCowsAuto<'_>) -> Result<Self, Self::Error> {
        Ok(match bytes {
            BytesCowsAuto::U32U8(cows) => Self::U32U8(cows.try_into()?),
            BytesCowsAuto::U32U16(cows) => Self::U32U16(cows.try_into()?),
            BytesCowsAuto::U32U32(cows) => Self::U32U32(cows.try_into()?),
            BytesCowsAuto::U64U8(cows) => Self::U64U8(cows.try_into()?),
            BytesCowsAuto::U64U16(cows) => Self::U64U16(cows.try_into()?),
            BytesCowsAuto::U64U32(cows) => Self::U64U32(cows.try_into()?),
        })
    }
}

/// A [`CharsTape`] that starts with `i32` offsets and widens them to `u32`, then `u64`.
pub enum CharsTapeAuto<A: Allocator = Global> {
    I32(CharsTape<i32, A>),
    U32(CharsTape<u32, A>),
    U64(CharsTape<u64, A>),
}

impl<A: Allocator + Clone> CharsTapeAuto<A> {
    /// Appends a string, widening the offsets when the data outgrows them.
    ///
    /// Widening copies every string into a new tape, in a clone of the allocator.
    pub fn push(&mut self, s: &str) -> Result<(), StringTapeError> {
        loop {
            let pushed = match self {
                Self::I32(t) => t.push(s),
                Self::U32(t) => t.push(s),
                Self::U64(t) => t.push(s),
            };
            if pushed != Err(StringTapeError::OffsetOverflow) {
                return pushed;
            }
            *self = match self {
                Self::I32(t) => Self::U32(t.widen()?),
                Self::U32(t) => Self::U64(t.widen()?),
                Self::U64(_) => return pushed,
            };
        }
    }

    /// Appends every string an iterator yields, stopping at the first that fails.
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
}

impl<A: Allocator> CharsTapeAuto<A> {
    /// Creates CharsTapeAuto with custom allocator.
    pub fn new_in(allocator: A) -> Self {
        Self::I32(CharsTape::new_in(allocator))
    }

    /// Creates a tape with pre-allocated capacity, using a custom allocator.
    pub fn with_capacity_in(
        data_capacity: usize,
        strings_capacity: usize,
        allocator: A,
    ) -> Result<Self, StringTapeError> {
        Ok(Self::I32(CharsTape::with_capacity_in(
            data_capacity,
            strings_capacity,
            allocator,
        )?))
    }

    /// Returns the number of strings.
    pub fn len(&self) -> usize {
        match self {
            Self::I32(t) => t.len(),
            Self::U32(t) => t.len(),
            Self::U64(t) => t.len(),
        }
    }

    /// Returns `true` if the tape contains no strings.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the string at `index`, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&str> {
        match self {
            Self::I32(t) => t.get(index),
            Self::U32(t) => t.get(index),
            Self::U64(t) => t.get(index),
        }
    }

    /// Returns the string at `index` without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `index < self.len()`.
    pub unsafe fn get_unchecked(&self, index: usize) -> &str {
        match self {
            Self::I32(t) => t.get_unchecked(index),
            Self::U32(t) => t.get_unchecked(index),
            Self::U64(t) => t.get_unchecked(index),
        }
    }

    /// Returns the first string, or `None` if empty.
    pub fn first(&self) -> Option<&str> {
        self.get(0)
    }

    /// Returns the last string, or `None` if empty.
    pub fn last(&self) -> Option<&str> {
        match self {
            Self::I32(t) => t.last(),
            Self::U32(t) => t.last(),
            Self::U64(t) => t.last(),
        }
    }

    /// Removes the last string and returns it, or `None` if empty.
    pub fn pop(&mut self) -> Option<&str> {
        match self {
            Self::I32(t) => t.pop(),
            Self::U32(t) => t.pop(),
            Self::U64(t) => t.pop(),
        }
    }

    /// Returns true if the tape contains the given string.
    pub fn contains(&self, item: &str) -> bool {
        match self {
            Self::I32(t) => t.contains(item),
            Self::U32(t) => t.contains(item),
            Self::U64(t) => t.contains(item),
        }
    }

    /// Returns the total number of bytes used by string data.
    pub fn data_len(&self) -> usize {
        match self {
            Self::I32(t) => t.data_len(),
            Self::U32(t) => t.data_len(),
            Self::U64(t) => t.data_len(),
        }
    }

    /// Returns the number of strings that can be stored without reallocating.
    pub fn capacity(&self) -> usize {
        match self {
            Self::I32(t) => t.capacity(),
            Self::U32(t) => t.capacity(),
            Self::U64(t) => t.capacity(),
        }
    }

    /// Returns the number of bytes allocated for string data.
    pub fn data_capacity(&self) -> usize {
        match self {
            Self::I32(t) => t.data_capacity(),
            Self::U32(t) => t.data_capacity(),
            Self::U64(t) => t.data_capacity(),
        }
    }

    /// Returns the number of offset slots allocated.
    pub fn offsets_capacity(&self) -> usize {
        match self {
            Self::I32(t) => t.offsets_capacity(),
            Self::U32(t) => t.offsets_capacity(),
            Self::U64(t) => t.offsets_capacity(),
        }
    }

    /// Reserves room for more bytes and more strings.
    pub fn reserve(
        &mut self,
        additional_bytes: usize,
        additional_strings: usize,
    ) -> Result<(), StringTapeError> {
        match self {
            Self::I32(t) => t.reserve(additional_bytes, additional_strings),
            Self::U32(t) => t.reserve(additional_bytes, additional_strings),
            Self::U64(t) => t.reserve(additional_bytes, additional_strings),
        }
    }

    /// Removes all strings, keeping allocated capacity.
    pub fn clear(&mut self) {
        match self {
            Self::I32(t) => t.clear(),
            Self::U32(t) => t.clear(),
            Self::U64(t) => t.clear(),
        }
    }

    /// Keeps the first `len` strings and drops the rest.
    pub fn truncate(&mut self, len: usize) {
        match self {
            Self::I32(t) => t.truncate(len),
            Self::U32(t) => t.truncate(len),
            Self::U64(t) => t.truncate(len),
        }
    }

    /// Releases capacity beyond what is currently used.
    pub fn shrink_to_fit(&mut self) -> Result<(), StringTapeError> {
        match self {
            Self::I32(t) => t.shrink_to_fit(),
            Self::U32(t) => t.shrink_to_fit(),
            Self::U64(t) => t.shrink_to_fit(),
        }
    }

    /// Returns the bytes of every string, back to back.
    pub fn data(&self) -> &[u8] {
        match self {
            Self::I32(t) => t.data(),
            Self::U32(t) => t.data(),
            Self::U64(t) => t.data(),
        }
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &A {
        match self {
            Self::I32(t) => t.allocator(),
            Self::U32(t) => t.allocator(),
            Self::U64(t) => t.allocator(),
        }
    }

    /// Returns the name of the selected variant.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::I32(_) => "CharsTape<i32>",
            Self::U32(_) => "CharsTape<u32>",
            Self::U64(_) => "CharsTape<u64>",
        }
    }

    /// Returns an iterator over the strings.
    pub fn iter(&self) -> CharsTapeAutoIter<'_, A> {
        CharsTapeAutoIter {
            tape: self,
            front: 0,
            back: self.len(),
        }
    }
}

/// Iterator over the strings of a [`CharsTapeAuto`].
#[derive(Clone, Copy)]
pub struct CharsTapeAutoIter<'a, A: Allocator> {
    tape: &'a CharsTapeAuto<A>,
    front: usize,
    back: usize,
}

impl<'a, A: Allocator> Iterator for CharsTapeAutoIter<'a, A> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let index = self.front;
            self.front += 1;
            Some(unsafe { self.tape.get_unchecked(index) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }
}

impl<'a, A: Allocator> DoubleEndedIterator for CharsTapeAutoIter<'a, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;
            Some(unsafe { self.tape.get_unchecked(self.back) })
        } else {
            None
        }
    }
}

impl<'a, A: Allocator> ExactSizeIterator for CharsTapeAutoIter<'a, A> {}

impl<'a, A: Allocator> FusedIterator for CharsTapeAutoIter<'a, A> {}

impl<'a, A: Allocator> fmt::Debug for CharsTapeAutoIter<'a, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharsTapeAutoIter")
            .field("front", &self.front)
            .field("back", &self.back)
            .field("remaining", &self.back.saturating_sub(self.front))
            .finish()
    }
}

impl<'a, A: Allocator> IntoIterator for &'a CharsTapeAuto<A> {
    type Item = &'a str;
    type IntoIter = CharsTapeAutoIter<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl CharsTapeAuto<Global> {
    /// Creates an empty CharsTapeAuto with the global allocator.
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates a tape with pre-allocated capacity, using the global allocator.
    pub fn with_capacity(
        data_capacity: usize,
        strings_capacity: usize,
    ) -> Result<Self, StringTapeError> {
        Self::with_capacity_in(data_capacity, strings_capacity, Global)
    }
}

impl Default for CharsTapeAuto<Global> {
    fn default() -> Self {
        Self::new_in(Global)
    }
}

impl<A: Allocator> fmt::Debug for CharsTapeAuto<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharsTapeAuto")
            .field("len", &self.len())
            .finish()
    }
}

impl<A: Allocator> PartialEq for CharsTapeAuto<A> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<A: Allocator> Eq for CharsTapeAuto<A> {}

impl<A: Allocator> Hash for CharsTapeAuto<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        self.iter().for_each(|item| item.hash(state));
    }
}

impl<A: Allocator> PartialOrd for CharsTapeAuto<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: Allocator> Ord for CharsTapeAuto<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

/// A [`BytesTape`] that starts with `u16` offsets and widens them to `u32`, then `u64`.
pub enum BytesTapeAuto<A: Allocator = Global> {
    U16(BytesTape<u16, A>),
    U32(BytesTape<u32, A>),
    U64(BytesTape<u64, A>),
}

impl<A: Allocator + Clone> BytesTapeAuto<A> {
    /// Appends an item, widening the offsets when the data outgrows them.
    ///
    /// Widening copies every item into a new tape, in a clone of the allocator.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), StringTapeError> {
        loop {
            let pushed = match self {
                Self::U16(t) => t.push(bytes),
                Self::U32(t) => t.push(bytes),
                Self::U64(t) => t.push(bytes),
            };
            if pushed != Err(StringTapeError::OffsetOverflow) {
                return pushed;
            }
            *self = match self {
                Self::U16(t) => Self::U32(t.widen()?),
                Self::U32(t) => Self::U64(t.widen()?),
                Self::U64(_) => return pushed,
            };
        }
    }

    /// Appends every item an iterator yields, stopping at the first that fails.
    pub fn extend<I>(&mut self, iter: I) -> Result<(), StringTapeError>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        for bytes in iter {
            self.push(bytes.as_ref())?;
        }
        Ok(())
    }
}

impl<A: Allocator> BytesTapeAuto<A> {
    /// Creates BytesTapeAuto with custom allocator.
    pub fn new_in(allocator: A) -> Self {
        Self::U16(BytesTape::new_in(allocator))
    }

    /// Creates a tape with pre-allocated capacity, using a custom allocator.
    pub fn with_capacity_in(
        data_capacity: usize,
        items_capacity: usize,
        allocator: A,
    ) -> Result<Self, StringTapeError> {
        Ok(Self::U16(BytesTape::with_capacity_in(
            data_capacity,
            items_capacity,
            allocator,
        )?))
    }

    /// Returns the number of items.
    pub fn len(&self) -> usize {
        match self {
            Self::U16(t) => t.len(),
            Self::U32(t) => t.len(),
            Self::U64(t) => t.len(),
        }
    }

    /// Returns `true` if the tape contains no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the item at `index`, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        match self {
            Self::U16(t) => t.get(index),
            Self::U32(t) => t.get(index),
            Self::U64(t) => t.get(index),
        }
    }

    /// Returns the item at `index` without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `index < self.len()`.
    pub unsafe fn get_unchecked(&self, index: usize) -> &[u8] {
        match self {
            Self::U16(t) => t.get_unchecked(index),
            Self::U32(t) => t.get_unchecked(index),
            Self::U64(t) => t.get_unchecked(index),
        }
    }

    /// Returns the first item, or `None` if empty.
    pub fn first(&self) -> Option<&[u8]> {
        self.get(0)
    }

    /// Returns the last item, or `None` if empty.
    pub fn last(&self) -> Option<&[u8]> {
        match self {
            Self::U16(t) => t.last(),
            Self::U32(t) => t.last(),
            Self::U64(t) => t.last(),
        }
    }

    /// Removes the last item and returns it, or `None` if empty.
    pub fn pop(&mut self) -> Option<&[u8]> {
        match self {
            Self::U16(t) => t.pop(),
            Self::U32(t) => t.pop(),
            Self::U64(t) => t.pop(),
        }
    }

    /// Returns true if the tape contains the given byte slice.
    pub fn contains(&self, item: &[u8]) -> bool {
        match self {
            Self::U16(t) => t.contains(item),
            Self::U32(t) => t.contains(item),
            Self::U64(t) => t.contains(item),
        }
    }

    /// Returns the total number of bytes used by item data.
    pub fn data_len(&self) -> usize {
        match self {
            Self::U16(t) => t.data_len(),
            Self::U32(t) => t.data_len(),
            Self::U64(t) => t.data_len(),
        }
    }

    /// Returns the number of items that can be stored without reallocating.
    pub fn capacity(&self) -> usize {
        match self {
            Self::U16(t) => t.capacity(),
            Self::U32(t) => t.capacity(),
            Self::U64(t) => t.capacity(),
        }
    }

    /// Returns the number of bytes allocated for data.
    pub fn data_capacity(&self) -> usize {
        match self {
            Self::U16(t) => t.data_capacity(),
            Self::U32(t) => t.data_capacity(),
            Self::U64(t) => t.data_capacity(),
        }
    }

    /// Returns the number of offset slots allocated.
    pub fn offsets_capacity(&self) -> usize {
        match self {
            Self::U16(t) => t.offsets_capacity(),
            Self::U32(t) => t.offsets_capacity(),
            Self::U64(t) => t.offsets_capacity(),
        }
    }

    /// Reserves room for more bytes and more items.
    pub fn reserve(
        &mut self,
        additional_bytes: usize,
        additional_items: usize,
    ) -> Result<(), StringTapeError> {
        match self {
            Self::U16(t) => t.reserve(additional_bytes, additional_items),
            Self::U32(t) => t.reserve(additional_bytes, additional_items),
            Self::U64(t) => t.reserve(additional_bytes, additional_items),
        }
    }

    /// Removes all items, keeping allocated capacity.
    pub fn clear(&mut self) {
        match self {
            Self::U16(t) => t.clear(),
            Self::U32(t) => t.clear(),
            Self::U64(t) => t.clear(),
        }
    }

    /// Keeps the first `len` items and drops the rest.
    pub fn truncate(&mut self, len: usize) {
        match self {
            Self::U16(t) => t.truncate(len),
            Self::U32(t) => t.truncate(len),
            Self::U64(t) => t.truncate(len),
        }
    }

    /// Releases capacity beyond what is currently used.
    pub fn shrink_to_fit(&mut self) -> Result<(), StringTapeError> {
        match self {
            Self::U16(t) => t.shrink_to_fit(),
            Self::U32(t) => t.shrink_to_fit(),
            Self::U64(t) => t.shrink_to_fit(),
        }
    }

    /// Returns the bytes of every item, back to back.
    pub fn data(&self) -> &[u8] {
        match self {
            Self::U16(t) => t.data(),
            Self::U32(t) => t.data(),
            Self::U64(t) => t.data(),
        }
    }

    /// Returns the bytes of every item for in-place edits that keep each item's length.
    pub fn data_mut(&mut self) -> &mut [u8] {
        match self {
            Self::U16(t) => t.data_mut(),
            Self::U32(t) => t.data_mut(),
            Self::U64(t) => t.data_mut(),
        }
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &A {
        match self {
            Self::U16(t) => t.allocator(),
            Self::U32(t) => t.allocator(),
            Self::U64(t) => t.allocator(),
        }
    }

    /// Returns the name of the selected variant.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U16(_) => "BytesTape<u16>",
            Self::U32(_) => "BytesTape<u32>",
            Self::U64(_) => "BytesTape<u64>",
        }
    }

    /// Returns an iterator over the items.
    pub fn iter(&self) -> BytesTapeAutoIter<'_, A> {
        BytesTapeAutoIter {
            tape: self,
            front: 0,
            back: self.len(),
        }
    }
}

/// Iterator over the items of a [`BytesTapeAuto`].
#[derive(Clone, Copy)]
pub struct BytesTapeAutoIter<'a, A: Allocator> {
    tape: &'a BytesTapeAuto<A>,
    front: usize,
    back: usize,
}

impl<'a, A: Allocator> Iterator for BytesTapeAutoIter<'a, A> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let index = self.front;
            self.front += 1;
            Some(unsafe { self.tape.get_unchecked(index) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }
}

impl<'a, A: Allocator> DoubleEndedIterator for BytesTapeAutoIter<'a, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;
            Some(unsafe { self.tape.get_unchecked(self.back) })
        } else {
            None
        }
    }
}

impl<'a, A: Allocator> ExactSizeIterator for BytesTapeAutoIter<'a, A> {}

impl<'a, A: Allocator> FusedIterator for BytesTapeAutoIter<'a, A> {}

impl<'a, A: Allocator> fmt::Debug for BytesTapeAutoIter<'a, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesTapeAutoIter")
            .field("front", &self.front)
            .field("back", &self.back)
            .field("remaining", &self.back.saturating_sub(self.front))
            .finish()
    }
}

impl<'a, A: Allocator> IntoIterator for &'a BytesTapeAuto<A> {
    type Item = &'a [u8];
    type IntoIter = BytesTapeAutoIter<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl BytesTapeAuto<Global> {
    /// Creates an empty BytesTapeAuto with the global allocator.
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates a tape with pre-allocated capacity, using the global allocator.
    pub fn with_capacity(
        data_capacity: usize,
        items_capacity: usize,
    ) -> Result<Self, StringTapeError> {
        Self::with_capacity_in(data_capacity, items_capacity, Global)
    }
}

impl Default for BytesTapeAuto<Global> {
    fn default() -> Self {
        Self::new_in(Global)
    }
}

impl<A: Allocator> fmt::Debug for BytesTapeAuto<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesTapeAuto")
            .field("len", &self.len())
            .finish()
    }
}

impl<A: Allocator> PartialEq for BytesTapeAuto<A> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<A: Allocator> Eq for BytesTapeAuto<A> {}

impl<A: Allocator> Hash for BytesTapeAuto<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        self.iter().for_each(|item| item.hash(state));
    }
}

impl<A: Allocator> PartialOrd for BytesTapeAuto<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: Allocator> Ord for BytesTapeAuto<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<Offset: OffsetType, A: Allocator> Index<Range<usize>> for CharsTape<Offset, A> {
    type Output = str;

    fn index(&self, range: Range<usize>) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeFrom<usize>> for CharsTape<Offset, A> {
    type Output = str;

    fn index(&self, range: RangeFrom<usize>) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeTo<usize>> for CharsTape<Offset, A> {
    type Output = str;

    fn index(&self, range: RangeTo<usize>) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeFull> for CharsTape<Offset, A> {
    type Output = str;

    fn index(&self, range: RangeFull) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeInclusive<usize>> for CharsTape<Offset, A> {
    type Output = str;

    fn index(&self, range: RangeInclusive<usize>) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeToInclusive<usize>> for CharsTape<Offset, A> {
    type Output = str;

    fn index(&self, range: RangeToInclusive<usize>) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<Offset: OffsetType, A: Allocator> Index<Range<usize>> for BytesTape<Offset, A> {
    type Output = [u8];

    fn index(&self, range: Range<usize>) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeFrom<usize>> for BytesTape<Offset, A> {
    type Output = [u8];

    fn index(&self, range: RangeFrom<usize>) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeTo<usize>> for BytesTape<Offset, A> {
    type Output = [u8];

    fn index(&self, range: RangeTo<usize>) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeFull> for BytesTape<Offset, A> {
    type Output = [u8];

    fn index(&self, range: RangeFull) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeInclusive<usize>> for BytesTape<Offset, A> {
    type Output = [u8];

    fn index(&self, range: RangeInclusive<usize>) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

impl<Offset: OffsetType, A: Allocator> Index<RangeToInclusive<usize>> for BytesTape<Offset, A> {
    type Output = [u8];

    fn index(&self, range: RangeToInclusive<usize>) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

impl<'a, Offset: OffsetType> Index<Range<usize>> for CharsTapeView<'a, Offset> {
    type Output = str;

    fn index(&self, range: Range<usize>) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<'a, Offset: OffsetType> Index<RangeFrom<usize>> for CharsTapeView<'a, Offset> {
    type Output = str;

    fn index(&self, range: RangeFrom<usize>) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<'a, Offset: OffsetType> Index<RangeTo<usize>> for CharsTapeView<'a, Offset> {
    type Output = str;

    fn index(&self, range: RangeTo<usize>) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<'a, Offset: OffsetType> Index<RangeFull> for CharsTapeView<'a, Offset> {
    type Output = str;

    fn index(&self, range: RangeFull) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<'a, Offset: OffsetType> Index<RangeInclusive<usize>> for CharsTapeView<'a, Offset> {
    type Output = str;

    fn index(&self, range: RangeInclusive<usize>) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<'a, Offset: OffsetType> Index<RangeToInclusive<usize>> for CharsTapeView<'a, Offset> {
    type Output = str;

    fn index(&self, range: RangeToInclusive<usize>) -> &str {
        self.subview(range).expect("range out of bounds").as_str()
    }
}

impl<'a, Offset: OffsetType> Index<Range<usize>> for BytesTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: Range<usize>) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

impl<'a, Offset: OffsetType> Index<RangeFrom<usize>> for BytesTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeFrom<usize>) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

impl<'a, Offset: OffsetType> Index<RangeTo<usize>> for BytesTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeTo<usize>) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

impl<'a, Offset: OffsetType> Index<RangeFull> for BytesTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeFull) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

impl<'a, Offset: OffsetType> Index<RangeInclusive<usize>> for BytesTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeInclusive<usize>) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

impl<'a, Offset: OffsetType> Index<RangeToInclusive<usize>> for BytesTapeView<'a, Offset> {
    type Output = [u8];

    fn index(&self, range: RangeToInclusive<usize>) -> &[u8] {
        self.subview(range).expect("range out of bounds").as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::string::{String, ToString};
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

        assert_eq!(tape.pop(), Some("foo"));
        tape.push("bar").unwrap();
        assert_eq!(tape.last(), Some("bar"));
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
        assert_eq!(tape.iter().collect::<Vec<_>>(), [""]);
        assert_eq!(tape.view().len(), 1);

        tape.push("non-empty").unwrap();
        tape.push("").unwrap();

        assert_eq!(tape.len(), 3);
        assert_eq!(tape.get(0), Some(""));
        assert_eq!(tape.get(1), Some("non-empty"));
        assert_eq!(tape.get(2), Some(""));
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
        let view = tape.subview(1..4).unwrap();
        assert_eq!(view.len(), 3);
        assert_eq!(view.get(0), Some("1"));
        assert_eq!(view.get(2), Some("333"));
        let sub = view.subview(1..2).unwrap();
        assert_eq!(sub.len(), 1);
        assert_eq!(sub.get(0), Some("22"));
    }

    #[test]
    fn capacity() {
        let mut tape = CharsTapeI32::with_capacity(100, 10).unwrap();
        assert_eq!(tape.data_capacity(), 100);
        tape.reserve(100, 0).unwrap();
        assert_eq!(tape.data_capacity(), 100);
        assert_eq!(
            tape.reserve(0, usize::MAX),
            Err(StringTapeError::Allocation)
        );
        // Capacity is what was requested, independent of how many strings are stored.
        assert_eq!(tape.len(), 0);
        assert!(tape.capacity() >= 10, "capacity was {}", tape.capacity());

        let requested = tape.capacity();
        tape.push("hello").unwrap();
        assert_eq!(tape.len(), 1);
        assert_eq!(tape.capacity(), requested);
    }

    #[test]
    fn bytes_tape_exposes_mutable_data() {
        let mut tape = BytesTapeU32::new();
        tape.push(b"acgt").unwrap();
        tape.push(b"tgca").unwrap();

        for byte in tape.data_mut() {
            byte.make_ascii_uppercase();
        }

        assert_eq!(tape.get(0), Some(&b"ACGT"[..]));
        assert_eq!(tape.get(1), Some(&b"TGCA"[..]));
    }

    #[test]
    fn mutable_data_of_an_empty_tape_is_empty() {
        let mut tape = BytesTapeU32::new();
        assert!(tape.data_mut().is_empty());
    }

    /// A cleared tape can be re-sized for the next batch, which was impossible while
    /// `reserve` lived only on the private inner type.
    #[test]
    fn reserve_grows_a_cleared_tape() {
        let mut tape = BytesTapeU32::new();
        tape.push(b"small").unwrap();
        tape.clear();

        tape.reserve(4096, 64).unwrap();
        let data_capacity = tape.data_capacity();
        assert!(data_capacity >= 4096);

        for _ in 0..64 {
            tape.push(b"item").unwrap();
        }
        assert_eq!(tape.len(), 64);
        assert_eq!(
            tape.data_capacity(),
            data_capacity,
            "pushes within the reserved size should not reallocate"
        );
    }

    #[test]
    fn reserve_works_on_chars_tapes_too() {
        let mut tape = CharsTape::<u32>::new();
        tape.reserve(1024, 16).unwrap();
        assert!(tape.data_capacity() >= 1024);
        assert!(tape.capacity() >= 16);
    }

    /// Capacity must describe the allocation, not the number of stored items.
    #[test]
    fn capacity_reports_capacity_not_length() {
        let mut tape = BytesTape::<u32>::with_capacity(128, 8).unwrap();
        tape.push(b"one").unwrap();
        assert_eq!(tape.len(), 1);
        assert!(tape.capacity() >= 8, "capacity was {}", tape.capacity());

        let mut chars = CharsTape::<u32>::with_capacity(128, 8).unwrap();
        chars.push("one").unwrap();
        assert_eq!(chars.len(), 1);
        assert!(chars.capacity() >= 8, "capacity was {}", chars.capacity());
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
        assert_eq!(tape.get(0), Some(&[1u8, 2, 3][..]));
        assert_eq!(tape.get(1), Some(&b"abc"[..]));
    }

    #[test]
    fn unsigned_bytes_tape_basic() {
        let mut tape = BytesTapeU64::new();
        tape.push(&[1u8, 2]).unwrap();
        tape.push(&[3u8, 4, 5]).unwrap();
        assert_eq!(tape.len(), 2);
        assert_eq!(tape.get(0), Some(&[1u8, 2][..]));
        assert_eq!(tape.get(1), Some(&[3u8, 4, 5][..]));
    }

    #[test]
    fn overflowing_push_leaves_the_tape_unchanged() {
        let mut tape = BytesTapeU16::new();
        tape.push(&[0; 40_000]).unwrap();
        assert_eq!(
            tape.push(&[0; 40_000]),
            Err(StringTapeError::OffsetOverflow)
        );
        assert_eq!(tape.len(), 1);
        assert_eq!(tape.data_len(), 40_000);
    }

    #[test]
    fn chars_tape_view_basic() {
        let mut tape = CharsTapeI32::new();
        tape.push("hello").unwrap();
        tape.push("world").unwrap();
        tape.push("foo").unwrap();
        tape.push("bar").unwrap();

        // Test basic subview creation
        let view = tape.subview(1..3).unwrap();
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0), Some("world"));
        assert_eq!(view.get(1), Some("foo"));
        assert_eq!(view.get(2), None);
    }

    #[test]
    fn chars_tape_range_syntax() {
        let mut tape = CharsTapeI32::new();
        tape.push("a").unwrap();
        tape.push("b").unwrap();
        tape.push("c").unwrap();
        tape.push("d").unwrap();

        // Items borrow the tape, so they outlive the view they came from
        let first = tape.view().get(0);
        assert_eq!(first, Some("a"));

        // Test view() method
        let full_view = tape.view();
        assert_eq!(full_view.len(), 4);
        assert_eq!(full_view.get(0), Some("a"));
        assert_eq!(full_view.get(3), Some("d"));

        // Test subview
        let sub = tape.subview(1..3).unwrap();
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
        let view = tape.subview(1..4).unwrap(); // ["1", "2", "3"]
        assert_eq!(view.len(), 3);

        // Create sub-view of a view
        let subview = view.subview(1..2).unwrap(); // ["2"]
        assert_eq!(subview.len(), 1);
        assert_eq!(subview.get(0), Some("2"));

        // Test subviews with different ranges
        let subview_from = view.subview(1..).unwrap(); // ["2", "3"]
        assert_eq!(subview_from.len(), 2);
        assert_eq!(subview_from.get(0), Some("2"));
        assert_eq!(subview_from.get(1), Some("3"));

        let subview_to = view.subview(..2).unwrap(); // ["1", "2"]
        assert_eq!(subview_to.len(), 2);
        assert_eq!(subview_to.get(0), Some("1"));
        assert_eq!(subview_to.get(1), Some("2"));

        assert_eq!(view.subview(..=1), Some(subview_to));
        assert_eq!(view.subview(..), Some(view));
        assert_eq!(view.subview(..=3), None);
    }

    #[test]
    fn bytes_tape_view_basic() {
        let mut tape = BytesTapeI32::new();
        tape.push(&[1u8, 2]).unwrap();
        tape.push(&[3u8, 4]).unwrap();
        tape.push(&[5u8, 6]).unwrap();
        tape.push(&[7u8, 8]).unwrap();

        // Test basic subview creation
        let view = tape.subview(1..3).unwrap();
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0), Some(&[3u8, 4] as &[u8]));
        assert_eq!(view.get(1), Some(&[5u8, 6] as &[u8]));
        assert_eq!(view.get(2), None);
    }

    #[test]
    fn view_empty_strings() {
        let mut tape = CharsTapeI32::new();
        tape.push("").unwrap();
        tape.push("non-empty").unwrap();
        tape.push("").unwrap();
        tape.push("another").unwrap();

        let view = tape.subview(0..3).unwrap();
        assert_eq!(view.len(), 3);
        assert_eq!(view.get(0), Some(""));
        assert_eq!(view.get(1), Some("non-empty"));
        assert_eq!(view.get(2), Some(""));
    }

    #[test]
    fn view_single_item() {
        let mut tape = CharsTapeI32::new();
        tape.push("only").unwrap();

        let view = tape.subview(0..1).unwrap();
        assert_eq!(view.len(), 1);
        assert_eq!(view.get(0), Some("only"));
    }

    #[test]
    fn view_bounds_checking() {
        let mut tape = CharsTapeI32::new();
        tape.push("a").unwrap();
        tape.push("b").unwrap();

        // Out of bounds subview creation
        let (start, end) = (2, 1);
        assert!(tape.subview(0..3).is_none());
        assert!(tape.subview(start..end).is_none());
        assert!(tape.subview(3..4).is_none());

        // Valid empty subview
        let empty_view = tape.subview(1..1).unwrap();
        assert_eq!(empty_view.len(), 0);
        assert!(empty_view.is_empty());
    }

    #[test]
    fn view_data_properties() {
        let mut tape = CharsTapeI32::new();
        assert_eq!(tape.view().data_len(), 0);
        assert_eq!(tape.offsets(), [0]);
        tape.push("hello").unwrap(); // 5 bytes
        tape.push("world").unwrap(); // 5 bytes
        tape.push("!").unwrap(); // 1 byte

        let view = tape.subview(0..2).unwrap(); // "hello", "world" = 10 bytes
        assert_eq!(view.data_len(), 10);
        assert!(!view.is_empty());

        let full_view = tape.subview(0..3).unwrap(); // all = 11 bytes
        assert_eq!(full_view.data_len(), 11);
    }

    #[test]
    fn view_raw_parts() {
        let mut tape = CharsTapeI32::new();
        tape.push("test").unwrap();
        tape.push("data").unwrap();

        let view = tape.subview(0..2).unwrap();
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
        let view = tape.subview(1..3).unwrap();
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
        let view = tape.subview(2..4).unwrap();
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

        let _view: CharsTapeViewI32 = tape.subview(0..1).unwrap();

        let mut bytes_tape = BytesTapeI64::new();
        bytes_tape.push(b"test").unwrap();

        let _bytes_view: BytesTapeViewI64 = bytes_tape.subview(0..1).unwrap();
    }

    #[test]
    fn build_i32_from_other_offset_iterators() {
        let items = ["x", "yy", "", "zzz"];

        // From u32 iterator
        let mut u32t = CharsTapeU32::new();
        u32t.extend(items).unwrap();
        let mut t_from_u32 = CharsTapeI32::new();
        t_from_u32.extend(&u32t).unwrap();
        assert_eq!(t_from_u32.len(), items.len());
        assert_eq!(t_from_u32.get(1), Some("yy"));

        // From u64 iterator
        let mut u64t = CharsTapeU64::new();
        u64t.extend(items).unwrap();
        let mut t_from_u64 = CharsTapeI32::new();
        t_from_u64.extend(&u64t).unwrap();
        assert_eq!(t_from_u64.len(), items.len());
        assert_eq!(t_from_u64.get(3), Some("zzz"));

        // From i64 iterator
        let mut i64t = CharsTapeI64::new();
        i64t.extend(items).unwrap();
        let mut t_from_i64 = CharsTapeI32::new();
        t_from_i64.extend(&i64t).unwrap();
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

        // Get full view
        let full_view = tape.view();
        assert_eq!(full_view.len(), 4);

        // Get subviews
        let sub = tape.subview(1..3).unwrap();
        assert_eq!(sub.len(), 2);
        assert_eq!(sub.get(0), Some("b"));
        assert_eq!(sub.get(1), Some("c"));

        // Test subview of subview
        let sub_sub = sub.subview(0..1).unwrap();
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

        let data_buffer = Buffer::from_slice_ref(tape.data());
        let offsets_buffer = OffsetBuffer::new(ScalarBuffer::new(
            Buffer::from_slice_ref(tape.offsets()),
            0,
            tape.offsets().len(),
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
        let data_buffer = Buffer::from_slice_ref(tape.data());
        let offsets_buffer = OffsetBuffer::new(ScalarBuffer::new(
            Buffer::from_slice_ref(tape.offsets()),
            0,
            tape.offsets().len(),
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

        let error = CharsTapeI32::try_from(bytes_tape).unwrap_err();
        assert!(matches!(
            error,
            StringTapeError::InvalidUtf8 { index: 1, .. }
        ));
        assert!(core::error::Error::source(&error).is_some());
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
        assert_eq!(bytes_tape.get(0), Some(&b"hello"[..]));
        assert_eq!(bytes_tape.get(1), Some("世界".as_bytes()));
        assert_eq!(bytes_tape.get(2), Some(&b""[..]));
        assert_eq!(bytes_tape.get(3), Some("🦀".as_bytes()));
    }

    #[test]
    fn conversion_round_trip() {
        // Test round-trip conversion preserves data
        let mut original = CharsTapeI32::new();
        original.push("first").unwrap();
        original.push("second").unwrap();
        original.push("third").unwrap();

        // Store expected values before conversion
        let expected = ["first", "second", "third"];

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
        let bytes_view = BytesTapeViewI32::from(chars_view);
        assert_eq!(bytes_view.len(), 3);
        assert_eq!(bytes_view.get(0), Some("abc".as_bytes()));
        assert_eq!(bytes_view.get(1), Some(b"" as &[u8]));
        assert_eq!(bytes_view.get(2), Some("世界".as_bytes()));

        // Bytes -> Chars view conversion is fallible, but should succeed for valid UTF-8
        let chars_back = CharsTapeViewI32::try_from(bytes_view);
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
        assert!(matches!(
            CharsTapeViewI32::try_from(bview),
            Err(StringTapeError::InvalidUtf8 { index: 1, .. })
        ));
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
        assert_eq!(result.unwrap_err(), StringTapeError::LengthOverflow);

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
            [data.as_str(), other_data.as_str()],
            Cow::Borrowed(data.as_bytes()),
        );
        assert_eq!(
            result.unwrap_err(),
            StringTapeError::OutsideData { index: 1 }
        );

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

        // BytesCows -> CharsCows, borrowing the buffer
        let borrowed = CharsCowsU32U8::try_from(&bytes).unwrap();
        assert_eq!(BytesCowsU32U8::try_from(&borrowed).unwrap(), bytes);

        // BytesCows -> CharsCows
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

    // region: Spanned text

    /// A range's bytes must cover the requested items, not the whole tape.
    #[test]
    fn range_indexing_spans_only_the_requested_items() {
        let mut tape = BytesTapeU32::new();
        tape.extend([&b"alpha"[..], b"beta", b"gamma"]).unwrap();

        let view = tape.view();
        assert_eq!(&view[1..3], b"betagamma");
        assert_eq!(&view[0..1], b"alpha");
        assert_eq!(&view[..2], b"alphabeta");
        assert_eq!(&view[1..], b"betagamma");
        assert_eq!(&view[1..=2], b"betagamma");
        assert_eq!(&view[..=1], b"alphabeta");
        assert_eq!(&view[..], b"alphabetagamma");
        assert_eq!(
            view.subview(1..3).map(|items| items.as_bytes()),
            Some(&b"betagamma"[..])
        );
    }

    #[test]
    fn range_indexing_works_on_the_tape_itself() {
        let mut tape = BytesTapeU32::new();
        tape.extend([&b"alpha"[..], b"beta", b"gamma"]).unwrap();
        assert_eq!(&tape[1..3], b"betagamma");
        assert_eq!(&tape[..], b"alphabetagamma");
    }

    /// Item boundaries are character boundaries, so a Chars range yields `&str`.
    #[test]
    fn range_indexing_of_chars_yields_text() {
        let mut tape = CharsTapeU32::new();
        tape.extend(["alpha", "béta", "gamma"]).unwrap();
        assert_eq!(&tape[1..3], "bétagamma");
        assert_eq!(&tape.view()[0..2], "alphabéta");
        assert_eq!(&tape[..], "alphabétagamma");
    }

    #[test]
    fn range_indexing_of_an_empty_view_is_empty() {
        let tape = BytesTapeU32::new();
        assert!(tape.view()[..].is_empty());
        assert!(tape[..].is_empty());
    }

    /// Like `str`, an out-of-bounds range panics; `subview` is the checked form.
    #[test]
    #[should_panic(expected = "range out of bounds")]
    fn range_indexing_past_the_end_panics() {
        let mut tape = CharsTapeU32::new();
        tape.extend(["alpha"]).unwrap();
        assert!(tape.subview(0..2).is_none());
        let _ = &tape[0..2];
    }

    // endregion: Spanned text

    // region: BytesCows ordering

    fn bytes_cows(data: &[u8]) -> BytesCows<'_, u32, u16> {
        BytesCows::from_iter_and_data(data.split(|&byte| byte == b' '), Cow::Borrowed(data))
            .unwrap()
    }

    fn collect<'a>(cows: &'a BytesCows<'a, u32, u16>) -> Vec<&'a [u8]> {
        cows.iter().collect()
    }

    #[test]
    fn bytes_cows_sorts_lexicographically() {
        let data = b"zebra apple banana";
        let mut cows = bytes_cows(data);
        cows.sort();
        assert_eq!(collect(&cows), vec![&b"apple"[..], b"banana", b"zebra"]);
    }

    #[test]
    fn bytes_cows_sorts_with_comparator() {
        let data = b"aaa bb c";
        let mut cows = bytes_cows(data);
        cows.sort_by(|first, second| first.len().cmp(&second.len()));
        assert_eq!(collect(&cows), vec![&b"c"[..], b"bb", b"aaa"]);
    }

    #[test]
    fn bytes_cows_sorts_by_key() {
        let data = b"aaa bb c";
        let mut cows = bytes_cows(data);
        cows.sort_by_key(|slice| slice.len());
        assert_eq!(collect(&cows), vec![&b"c"[..], b"bb", b"aaa"]);
    }

    /// Ordering must handle non-UTF-8 payloads, which is why `BytesCows` exists
    /// separately from `CharsCows`.
    #[test]
    fn bytes_cows_sorts_arbitrary_bytes() {
        let data = &[0xFFu8, 0x00, b' ', 0x01, 0x02][..];
        let mut cows = BytesCows::<u32, u16>::from_iter_and_data(
            data.split(|&byte| byte == b' '),
            Cow::Borrowed(data),
        )
        .unwrap();
        cows.sort();
        assert_eq!(collect(&cows), vec![&[0x01u8, 0x02][..], &[0xFF, 0x00][..]]);
    }

    // endregion: BytesCows ordering

    // region: Auto forwarding

    /// The tape wrappers forwarded a handful of methods and dropped the rest; every one
    /// added here is exercised, since hand-written arms are where omissions hide.
    #[test]
    fn chars_tape_auto_forwards_the_tape_surface() {
        let mut tape = CharsTapeAuto::with_capacity(128, 8).unwrap();
        tape.extend(["alpha", "beta", "gamma"]).unwrap();

        assert_eq!(tape.len(), 3);
        assert_eq!(tape.first(), Some("alpha"));
        assert_eq!(tape.last(), Some("gamma"));
        assert_eq!(tape.get(1), Some("beta"));
        assert!(tape.contains("beta"));
        assert_eq!(tape.data_len(), 14);
        assert!(tape.capacity() >= 8);
        assert!(tape.data_capacity() >= 128);
        assert!(tape.offsets_capacity() >= 9);
        assert_eq!(tape.data(), b"alphabetagamma");
        assert_eq!(tape.type_name(), "CharsTape<i32>");
        assert_eq!(unsafe { tape.get_unchecked(2) }, "gamma");

        assert_eq!(
            tape.iter().collect::<Vec<&str>>(),
            vec!["alpha", "beta", "gamma"]
        );
        assert_eq!(tape.iter().next_back(), Some("gamma"));
        assert_eq!((&tape).into_iter().count(), 3);

        assert_eq!(tape.pop(), Some("gamma"));
        assert_eq!(tape.len(), 2);
        tape.truncate(1);
        assert_eq!(tape.len(), 1);
        tape.shrink_to_fit().unwrap();
        tape.clear();
        assert!(tape.is_empty());
        tape.reserve(256, 16).unwrap();
        assert!(tape.data_capacity() >= 256);
    }

    #[test]
    fn bytes_tape_auto_forwards_the_tape_surface() {
        let mut tape = BytesTapeAuto::with_capacity(128, 8).unwrap();
        tape.extend([&b"acgt"[..], b"tgca"]).unwrap();

        assert_eq!(tape.len(), 2);
        assert_eq!(tape.first(), Some(&b"acgt"[..]));
        assert_eq!(tape.last(), Some(&b"tgca"[..]));
        assert_eq!(tape.get(0), Some(&b"acgt"[..]));
        assert!(tape.contains(b"tgca"));
        assert_eq!(tape.data_len(), 8);
        assert_eq!(tape.type_name(), "BytesTape<u16>");
        assert_eq!(unsafe { tape.get_unchecked(1) }, b"tgca");
        assert_eq!(tape.iter().count(), 2);

        for byte in tape.data_mut() {
            byte.make_ascii_uppercase();
        }
        assert_eq!(tape.get(0), Some(&b"ACGT"[..]));

        tape.clear();
        tape.reserve(256, 16).unwrap();
        assert!(tape.capacity() >= 16);
    }

    #[test]
    fn cows_auto_convert_by_reference() {
        let data = "alpha beta";
        let cows =
            CharsCowsAuto::from_iter_and_data(data.split(' '), Cow::Borrowed(data.as_bytes()))
                .unwrap();
        let bytes = BytesCowsAuto::try_from(&cows).unwrap();
        assert_eq!(bytes.get(0), Some(&b"alpha"[..]));
        assert_eq!(CharsCowsAuto::try_from(&bytes).unwrap(), cows);
    }

    /// Past 64 KiB a `u16` tape refuses the push, and the wrapper widens instead of failing.
    #[test]
    fn bytes_tape_auto_widens_its_offsets() {
        let mut tape = BytesTapeAuto::new();
        tape.push(b"first").unwrap();
        tape.push(&[7; 70_000]).unwrap();
        assert_eq!(tape.type_name(), "BytesTape<u32>");
        assert_eq!(tape.len(), 2);
        assert_eq!(tape.first(), Some(&b"first"[..]));
        assert_eq!(tape.last().map(<[u8]>::len), Some(70_000));
    }

    // endregion: Auto forwarding

    // region: Reorderable

    /// Exercises every implementor through the trait, so a type that stops keeping up
    /// with the others fails to compile rather than going unnoticed.
    fn reorder_through_trait<C>(collection: &mut C) -> Result<(), StringTapeError>
    where
        C: Reorderable,
    {
        collection.sort();
        collection.sort_by(|first, second| first.cmp(second));
        collection.sort_by_key(|item| item.cmp(item));
        collection.permute(&[0])
    }

    #[test]
    fn every_cows_type_is_reorderable() {
        let text = "solo";
        let bytes = b"solo";

        let mut chars =
            CharsCows::<u32, u16>::from_iter_and_data([text], Cow::Borrowed(text.as_bytes()))
                .unwrap();
        let mut raw =
            BytesCows::<u32, u16>::from_iter_and_data([&bytes[..]], Cow::Borrowed(&bytes[..]))
                .unwrap();
        let mut chars_auto =
            CharsCowsAuto::from_iter_and_data([text], Cow::Borrowed(text.as_bytes())).unwrap();
        let mut bytes_auto =
            BytesCowsAuto::from_iter_and_data([&bytes[..]], Cow::Borrowed(&bytes[..])).unwrap();

        reorder_through_trait(&mut chars).unwrap();
        reorder_through_trait(&mut raw).unwrap();
        reorder_through_trait(&mut chars_auto).unwrap();
        reorder_through_trait(&mut bytes_auto).unwrap();

        assert_eq!(chars.get(0), Some("solo"));
        assert_eq!(bytes_auto.get(0), Some(&b"solo"[..]));
    }

    /// The ordering family reached `BytesCowsAuto` late; check it forwards for real.
    #[test]
    fn bytes_cows_auto_sorts_and_permutes() {
        let data = b"zebra apple banana";
        let mut cows =
            BytesCowsAuto::from_iter_and_data(data.split(|&b| b == b' '), Cow::Borrowed(&data[..]))
                .unwrap();

        cows.sort();
        assert_eq!(cows.get(0), Some(&b"apple"[..]));
        assert_eq!(cows.get(2), Some(&b"zebra"[..]));

        cows.permute(&[2, 1, 0]).unwrap();
        assert_eq!(cows.get(0), Some(&b"zebra"[..]));
        assert_eq!(cows.bytes_per_entry(), 5);
        assert_eq!(cows.type_name(), "BytesCows<u32, u8>");
    }

    // endregion: Reorderable

    // region: Cows construction

    #[test]
    fn chars_cows_builds_from_offsets() {
        let data = "alphabetagamma";
        let cows = CharsCows::<u32, u16>::from_offsets_and_data(
            [(0, 5), (5, 4), (9, 5)],
            Cow::Borrowed(data.as_bytes()),
        )
        .unwrap();
        assert_eq!(
            cows.iter().collect::<Vec<&str>>(),
            vec!["alpha", "beta", "gamma"]
        );
    }

    /// Unlike the byte constructor, the string one must reject spans that split a
    /// character, since every accessor assumes validated UTF-8.
    #[test]
    fn chars_cows_from_offsets_rejects_split_characters() {
        let data = "béta";
        assert!(matches!(
            CharsCows::<u32, u16>::from_offsets_and_data(
                [(0, 1), (1, 1)],
                Cow::Borrowed(data.as_bytes())
            ),
            Err(StringTapeError::InvalidUtf8 { index: 1, .. })
        ));
    }

    #[test]
    fn cows_from_offsets_rejects_spans_past_the_buffer() {
        let data = "alpha";
        assert!(matches!(
            CharsCows::<u32, u16>::from_offsets_and_data([(0, 99)], Cow::Borrowed(data.as_bytes())),
            Err(StringTapeError::OutsideData { index: 0 })
        ));
        assert!(matches!(
            BytesCows::<u32, u16>::from_offsets_and_data([(0, 99)], Cow::Borrowed(data.as_bytes())),
            Err(StringTapeError::OutsideData { index: 0 })
        ));
    }

    /// Owned items must work, which is why the shared builder wraps rather than maps.
    #[test]
    fn chars_cows_builds_from_owned_strings() {
        let data = "alpha beta";
        let owned: Vec<String> = data.split(' ').map(String::from).collect();
        // Owned copies do not point into `data`, so the pointer check must reject them.
        assert!(CharsCows::<u32, u16>::from_iter_and_data(
            owned.iter(),
            Cow::Borrowed(data.as_bytes())
        )
        .is_err());

        let borrowed: Vec<&str> = data.split(' ').collect();
        let cows =
            CharsCows::<u32, u16>::from_iter_and_data(borrowed, Cow::Borrowed(data.as_bytes()))
                .unwrap();
        assert_eq!(cows.iter().collect::<Vec<&str>>(), vec!["alpha", "beta"]);
    }

    // endregion: Cows construction

    // region: Permutation

    #[test]
    fn bytes_cows_permutes() {
        let data = b"alpha beta gamma";
        let mut cows = bytes_cows(data);
        cows.permute(&[2, 0, 1]).unwrap();
        assert_eq!(collect(&cows), vec![&b"gamma"[..], b"alpha", b"beta"]);
    }

    #[test]
    fn chars_cows_permutes() {
        let data = "alpha beta gamma";
        let mut cows = CharsCows::<u32, u8>::from_iter_and_data(
            data.split_whitespace(),
            Cow::Borrowed(data.as_bytes()),
        )
        .unwrap();
        cows.permute(&[2, 0, 1]).unwrap();
        assert_eq!(
            cows.iter().collect::<Vec<&str>>(),
            vec!["gamma", "alpha", "beta"]
        );
    }

    #[test]
    fn permute_identity_is_a_no_op() {
        let data = b"alpha beta gamma";
        let mut cows = bytes_cows(data);
        cows.permute(&[0, 1, 2]).unwrap();
        assert_eq!(collect(&cows), vec![&b"alpha"[..], b"beta", b"gamma"]);
    }

    #[test]
    fn permute_rejects_a_wrong_length_order() {
        let data = b"alpha beta gamma";
        let mut cows = bytes_cows(data);
        assert!(matches!(
            cows.permute(&[0, 1]),
            Err(StringTapeError::NotAPermutation)
        ));
    }

    #[test]
    fn permute_rejects_an_out_of_range_index() {
        let data = b"alpha beta gamma";
        let mut cows = bytes_cows(data);
        assert!(matches!(
            cows.permute(&[0, 1, 7]),
            Err(StringTapeError::NotAPermutation)
        ));
    }

    /// Repeats duplicate one slice and drop another, so they are rejected in every build.
    #[test]
    fn permute_rejects_repeats() {
        let data = b"alpha beta gamma";
        let mut cows = bytes_cows(data);
        assert!(matches!(
            cows.permute(&[0, 0, 1]),
            Err(StringTapeError::NotAPermutation)
        ));
    }

    /// The intended pairing: an external solver produces the order, `permute` applies it,
    /// and the result matches an in-place sort.
    #[test]
    fn permute_applies_an_externally_computed_order() {
        let data = b"delta alpha charlie bravo";
        let mut sorted_in_place = bytes_cows(data);
        sorted_in_place.sort();

        let mut permuted = bytes_cows(data);
        let mut order: Vec<usize> = (0..permuted.len()).collect();
        order.sort_by_key(|&index| permuted.get(index).unwrap().to_vec());
        permuted.permute(&order).unwrap();

        assert_eq!(collect(&permuted), collect(&sorted_in_place));
    }

    /// Payload bytes must not move — only the entries do.
    #[test]
    fn permute_leaves_the_payload_untouched() {
        let data = b"alpha beta gamma";
        let mut cows = bytes_cows(data);
        let before = cows.data().to_vec();
        cows.permute(&[2, 1, 0]).unwrap();
        assert_eq!(cows.data(), &before[..]);
    }

    // endregion: Permutation
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
        // Auto tapes widen their offsets as the data grows
        let mut tape = CharsTapeAuto::new();
        tape.extend(content.split_whitespace())?;

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
