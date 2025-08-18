#![cfg_attr(not(feature = "std"), no_std)]

//! # StringTape
//!
//! Memory-efficient string storage compatible with Apache Arrow.
//!
//! ```rust
//! use stringtape::{StringTape32, StringTapeError};
//!
//! let mut tape = StringTape32::new();
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

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::alloc::{alloc, dealloc, realloc, Layout, GlobalAlloc};
#[cfg(feature = "std")]
use std::alloc::{alloc, dealloc, realloc, Layout, GlobalAlloc};

use core::fmt;
use core::marker::PhantomData;
use core::ops::Index;
use core::ptr::NonNull;
use core::slice;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// Errors that can occur when working with StringTape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringTapeError {
    /// The string data size exceeds the maximum value representable by the offset type.
    ///
    /// This can happen when using 32-bit offsets (`StringTape32`) and the total data
    /// exceeds 2GB, or when memory allocation fails.
    OffsetOverflow,
    /// Memory allocation failed.
    AllocationError,
    /// Index is out of bounds for the current number of strings.
    IndexOutOfBounds,
}

impl fmt::Display for StringTapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StringTapeError::OffsetOverflow => write!(f, "offset value too large for offset type"),
            StringTapeError::AllocationError => write!(f, "memory allocation failed"),
            StringTapeError::IndexOutOfBounds => write!(f, "index out of bounds"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StringTapeError {}

/// A memory-efficient string storage structure compatible with Apache Arrow.
///
/// `StringTape` stores multiple strings in a contiguous memory layout using offset-based
/// indexing, similar to Apache Arrow's String and LargeString arrays. All string data
/// is stored in a single buffer, with a separate offset array tracking string boundaries.
///
/// # Type Parameters
///
/// * `Offset` - The offset type used for indexing (`i32` for StringTape32, `i64` for StringTape64)
/// * `A` - The allocator type used for memory allocation
///
/// # Examples
///
/// ```rust
/// use stringtape::{StringTape, StringTapeError, DefaultAllocator};
///
/// // Create a new StringTape with i32 offsets and default allocator
/// let mut tape: StringTape<i32, DefaultAllocator> = StringTape::new();
/// tape.push("hello")?;
/// tape.push("world")?;
///
/// assert_eq!(tape.len(), 2);
/// assert_eq!(&tape[0], "hello");
/// assert_eq!(tape.get(1), Some("world"));
/// # Ok::<(), StringTapeError>(())
/// ```
///
/// # Memory Layout
///
/// The memory layout is compatible with Apache Arrow:
/// ```text
/// Data buffer:    [h,e,l,l,o,w,o,r,l,d]
/// Offset buffer:  [0, 5, 10]
/// ```
pub struct StringTape<Offset: OffsetType = i32, A: Allocator = DefaultAllocator> {
    data: NonNull<u8>,
    offsets: NonNull<Offset>,
    capacity_bytes: usize,
    capacity_offsets: usize,
    len_bytes: usize,
    len_strings: usize,
    allocator: A,
    _phantom: PhantomData<Offset>,
}

/// Trait for custom allocators used in StringTape.
///
/// This trait wraps the standard library's `GlobalAlloc` trait to provide
/// allocation functionality for StringTape. The default implementation
/// uses the global allocator.
pub trait Allocator {
    /// Allocate memory with the given layout.
    /// 
    /// # Safety
    /// 
    /// See `GlobalAlloc::alloc` for safety requirements.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8;
    
    /// Deallocate memory with the given layout.
    /// 
    /// # Safety
    /// 
    /// See `GlobalAlloc::dealloc` for safety requirements.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);
    
    /// Reallocate memory with the given layout.
    /// 
    /// # Safety
    /// 
    /// See `GlobalAlloc::realloc` for safety requirements.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8;
}

/// Default allocator implementation using the global allocator.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultAllocator;

impl Allocator for DefaultAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        alloc(layout)
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        dealloc(ptr, layout)
    }
    
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        realloc(ptr, layout, new_size)
    }
}

/// Wrapper around any type that implements `GlobalAlloc`.
#[derive(Debug)]
pub struct GlobalAllocWrapper<T: GlobalAlloc>(pub T);

impl<T: GlobalAlloc> Allocator for GlobalAllocWrapper<T> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.0.alloc(layout)
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.0.dealloc(ptr, layout)
    }
    
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.0.realloc(ptr, layout, new_size)
    }
}

/// Trait for offset types used in StringTape.
///
/// This trait defines the interface for offset types that can be used to index
/// into the string data buffer. Implementations are provided for `i32` and `i64`
/// to match Apache Arrow's String and LargeString array types.
pub trait OffsetType: Copy + Default + PartialOrd {
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

impl<Offset: OffsetType, A: Allocator> StringTape<Offset, A> {
    /// Creates a new, empty StringTape with the default allocator.
    ///
    /// This operation is O(1) and does not allocate memory until the first string is pushed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::StringTape;
    ///
    /// let tape: StringTape<i32> = StringTape::new();
    /// assert!(tape.is_empty());
    /// assert_eq!(tape.len(), 0);
    /// ```
    pub fn new() -> Self
    where
        A: Default,
    {
        Self::new_in(A::default())
    }

    /// Creates a new, empty StringTape with a custom allocator.
    ///
    /// This operation is O(1) and does not allocate memory until the first string is pushed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::{StringTape, DefaultAllocator};
    ///
    /// let allocator = DefaultAllocator;
    /// let tape: StringTape<i32, DefaultAllocator> = StringTape::new_in(allocator);
    /// assert!(tape.is_empty());
    /// assert_eq!(tape.len(), 0);
    /// ```
    pub fn new_in(allocator: A) -> Self {
        Self {
            data: NonNull::dangling(),
            offsets: NonNull::dangling(),
            capacity_bytes: 0,
            capacity_offsets: 0,
            len_bytes: 0,
            len_strings: 0,
            allocator,
            _phantom: PhantomData,
        }
    }

    /// Creates a new StringTape with pre-allocated capacity using the default allocator.
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
    /// use stringtape::{StringTape, StringTapeError};
    ///
    /// // Pre-allocate space for ~1KB of string data and 100 strings
    /// let tape: StringTape<i32> = StringTape::with_capacity(1024, 100)?;
    /// assert_eq!(tape.data_capacity(), 1024);
    /// # Ok::<(), StringTapeError>(())
    /// ```
    pub fn with_capacity(
        data_capacity: usize,
        strings_capacity: usize,
    ) -> Result<Self, StringTapeError>
    where
        A: Default,
    {
        Self::with_capacity_in(data_capacity, strings_capacity, A::default())
    }

    /// Creates a new StringTape with pre-allocated capacity and a custom allocator.
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
    /// use stringtape::{StringTape, StringTapeError, DefaultAllocator};
    ///
    /// let allocator = DefaultAllocator;
    /// let tape: StringTape<i32, DefaultAllocator> = StringTape::with_capacity_in(1024, 100, allocator)?;
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
            let new_capacity = self
                .capacity_bytes
                .checked_add(additional_bytes)
                .ok_or(StringTapeError::AllocationError)?;
            self.grow_data(new_capacity)?;
        }

        if additional_strings > 0 {
            let new_capacity = self
                .capacity_offsets
                .checked_add(additional_strings + 1)
                .ok_or(StringTapeError::AllocationError)?;
            self.grow_offsets(new_capacity)?;
        }
        Ok(())
    }

    fn grow_data(&mut self, new_capacity: usize) -> Result<(), StringTapeError> {
        if new_capacity <= self.capacity_bytes {
            return Ok(());
        }

        let new_layout = Layout::from_size_align(new_capacity, 1)
            .map_err(|_| StringTapeError::AllocationError)?;

        let new_ptr = if self.capacity_bytes == 0 {
            unsafe { self.allocator.alloc(new_layout) }
        } else {
            let old_layout = Layout::from_size_align(self.capacity_bytes, 1).unwrap();
            unsafe { self.allocator.realloc(self.data.as_ptr(), old_layout, new_capacity) }
        };

        self.data = NonNull::new(new_ptr).ok_or(StringTapeError::AllocationError)?;
        self.capacity_bytes = new_capacity;
        Ok(())
    }

    fn grow_offsets(&mut self, new_capacity: usize) -> Result<(), StringTapeError> {
        if new_capacity <= self.capacity_offsets {
            return Ok(());
        }

        let new_layout =
            Layout::from_size_align(new_capacity * Offset::SIZE, core::mem::align_of::<Offset>())
                .map_err(|_| StringTapeError::AllocationError)?;

        let new_ptr = if self.capacity_offsets == 0 {
            let ptr = unsafe { self.allocator.alloc(new_layout) };
            if ptr.is_null() {
                return Err(StringTapeError::AllocationError);
            }
            unsafe {
                core::ptr::write(ptr.cast::<Offset>(), Offset::default());
            }
            ptr
        } else {
            let old_layout = Layout::from_size_align(
                self.capacity_offsets * Offset::SIZE,
                core::mem::align_of::<Offset>(),
            )
            .unwrap();
            unsafe { self.allocator.realloc(self.offsets.as_ptr().cast(), old_layout, new_layout.size()) }
        };

        self.offsets = NonNull::new(new_ptr.cast()).ok_or(StringTapeError::AllocationError)?;
        self.capacity_offsets = new_capacity;
        Ok(())
    }

    /// Adds a string to the end of the StringTape.
    ///
    /// # Errors
    ///
    /// Returns `StringTapeError::OffsetOverflow` if adding this string would cause
    /// the total data size to exceed the maximum value representable by the offset type.
    ///
    /// Returns `StringTapeError::AllocationError` if memory allocation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::{StringTape32, StringTapeError};
    ///
    /// let mut tape = StringTape32::new();
    /// tape.push("hello")?;
    /// tape.push("world")?;
    ///
    /// assert_eq!(tape.len(), 2);
    /// # Ok::<(), StringTapeError>(())
    /// ```
    pub fn push(&mut self, s: &str) -> Result<(), StringTapeError> {
        let bytes = s.as_bytes();
        let required_capacity = self
            .len_bytes
            .checked_add(bytes.len())
            .ok_or(StringTapeError::AllocationError)?;

        if required_capacity > self.capacity_bytes {
            let new_capacity = (self.capacity_bytes * 2).max(required_capacity).max(64);
            self.grow_data(new_capacity)?;
        }

        if self.len_strings + 1 >= self.capacity_offsets {
            let new_capacity = (self.capacity_offsets * 2).max(self.len_strings + 2).max(8);
            self.grow_offsets(new_capacity)?;
        }

        unsafe {
            #[cfg(feature = "std")]
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.data.as_ptr().add(self.len_bytes),
                bytes.len(),
            );
            #[cfg(not(feature = "std"))]
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.data.as_ptr().add(self.len_bytes),
                bytes.len(),
            );
        }

        self.len_bytes += bytes.len();
        self.len_strings += 1;

        let offset = Offset::from_usize(self.len_bytes).ok_or(StringTapeError::OffsetOverflow)?;
        unsafe {
            core::ptr::write(self.offsets.as_ptr().add(self.len_strings), offset);
        }

        Ok(())
    }

    /// Returns a reference to the string at the given index, or `None` if out of bounds.
    ///
    /// This operation is O(1).
    pub fn get(&self, index: usize) -> Option<&str> {
        if index >= self.len_strings {
            return None;
        }

        unsafe {
            let start_offset = if index == 0 {
                0
            } else {
                core::ptr::read(self.offsets.as_ptr().add(index)).to_usize()
            };
            let end_offset = core::ptr::read(self.offsets.as_ptr().add(index + 1)).to_usize();

            let slice = slice::from_raw_parts(
                self.data.as_ptr().add(start_offset),
                end_offset - start_offset,
            );

            Some(core::str::from_utf8_unchecked(slice))
        }
    }

    /// Returns the number of strings in the StringTape.
    pub fn len(&self) -> usize {
        self.len_strings
    }

    /// Returns `true` if the StringTape contains no strings.
    pub fn is_empty(&self) -> bool {
        self.len_strings == 0
    }

    /// Returns the total number of bytes used by string data.
    pub fn data_len(&self) -> usize {
        self.len_bytes
    }

    /// Returns the number of strings currently stored (same as `len()`).
    pub fn capacity(&self) -> usize {
        self.len_strings
    }

    /// Returns the number of bytes allocated for string data.
    pub fn data_capacity(&self) -> usize {
        self.capacity_bytes
    }

    /// Removes all strings from the StringTape, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.len_bytes = 0;
        self.len_strings = 0;
        if self.capacity_offsets > 0 {
            unsafe {
                core::ptr::write(self.offsets.as_ptr(), Offset::default());
            }
        }
    }

    /// Shortens the StringTape, keeping the first `len` strings and dropping the rest.
    ///
    /// If `len` is greater than the current length, this has no effect.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len_strings {
            return;
        }

        self.len_strings = len;
        self.len_bytes = if len == 0 {
            0
        } else {
            unsafe { core::ptr::read(self.offsets.as_ptr().add(len)).to_usize() }
        };
    }

    /// Extends the StringTape with the contents of an iterator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringtape::{StringTape32, StringTapeError};
    ///
    /// let mut tape = StringTape32::new();
    /// tape.extend(["hello", "world"])?;
    ///
    /// assert_eq!(tape.len(), 2);
    /// # Ok::<(), StringTapeError>(())
    /// ```
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

    /// Returns the raw parts of the StringTape for Apache Arrow compatibility.
    ///
    /// Returns a tuple of:
    /// - Data buffer pointer
    /// - Offsets buffer pointer  
    /// - Data length in bytes
    /// - Number of strings
    ///
    /// # Safety
    ///
    /// The returned pointers are valid only as long as the StringTape is not modified.
    pub fn as_raw_parts(&self) -> (*const u8, *const Offset, usize, usize) {
        (
            self.data.as_ptr(),
            self.offsets.as_ptr(),
            self.len_bytes,
            self.len_strings,
        )
    }

    pub fn iter(&self) -> StringTapeIter<'_, Offset, A> {
        StringTapeIter {
            tape: self,
            index: 0,
        }
    }
}

impl<Offset: OffsetType, A: Allocator> Drop for StringTape<Offset, A> {
    fn drop(&mut self) {
        if self.capacity_bytes > 0 {
            let layout = Layout::from_size_align(self.capacity_bytes, 1).unwrap();
            unsafe {
                self.allocator.dealloc(self.data.as_ptr(), layout);
            }
        }
        if self.capacity_offsets > 0 {
            let layout = Layout::from_size_align(
                self.capacity_offsets * Offset::SIZE,
                core::mem::align_of::<Offset>(),
            )
            .unwrap();
            unsafe {
                self.allocator.dealloc(self.offsets.as_ptr().cast(), layout);
            }
        }
    }
}

unsafe impl<Offset: OffsetType + Send, A: Allocator + Send> Send for StringTape<Offset, A> {}
unsafe impl<Offset: OffsetType + Sync, A: Allocator + Sync> Sync for StringTape<Offset, A> {}

pub struct StringTapeIter<'a, Offset: OffsetType, A: Allocator> {
    tape: &'a StringTape<Offset, A>,
    index: usize,
}

impl<'a, Offset: OffsetType, A: Allocator> Iterator for StringTapeIter<'a, Offset, A> {
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

impl<'a, Offset: OffsetType, A: Allocator> ExactSizeIterator for StringTapeIter<'a, Offset, A> {}

impl<Offset: OffsetType, A: Allocator + Default> FromIterator<String> for StringTape<Offset, A> {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let mut tape = StringTape::new();
        for s in iter {
            tape.push(&s)
                .expect("Failed to build StringTape from iterator");
        }
        tape
    }
}

impl<'a, Offset: OffsetType, A: Allocator + Default> FromIterator<&'a str> for StringTape<Offset, A> {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        let mut tape = StringTape::new();
        for s in iter {
            tape.push(s)
                .expect("Failed to build StringTape from iterator");
        }
        tape
    }
}

impl<Offset: OffsetType, A: Allocator> Index<usize> for StringTape<Offset, A> {
    type Output = str;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<'a, Offset: OffsetType, A: Allocator> IntoIterator for &'a StringTape<Offset, A> {
    type Item = &'a str;
    type IntoIter = StringTapeIter<'a, Offset, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub type StringTape32 = StringTape<i32, DefaultAllocator>;
pub type StringTape64 = StringTape<i64, DefaultAllocator>;

impl<Offset: OffsetType> Default for StringTape<Offset, DefaultAllocator> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Offset: OffsetType, A: Allocator> StringTape<Offset, A> {
    /// Returns a reference to the allocator used by this StringTape.
    pub fn allocator(&self) -> &A {
        &self.allocator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn test_basic_operations() {
        let mut tape = StringTape32::new();
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
    fn test_64bit_offsets() {
        let mut tape = StringTape64::new();
        tape.push("test").unwrap();
        assert_eq!(tape.get(0), Some("test"));
    }

    #[test]
    fn test_iterator() {
        let mut tape = StringTape32::new();
        tape.push("a").unwrap();
        tape.push("b").unwrap();
        tape.push("c").unwrap();

        let strings: Vec<&str> = tape.iter().collect();
        assert_eq!(strings, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_empty_strings() {
        let mut tape = StringTape32::new();
        tape.push("").unwrap();
        tape.push("non-empty").unwrap();
        tape.push("").unwrap();

        assert_eq!(tape.len(), 3);
        assert_eq!(tape.get(0), Some(""));
        assert_eq!(tape.get(1), Some("non-empty"));
        assert_eq!(tape.get(2), Some(""));
    }

    #[test]
    fn test_index_trait() {
        let mut tape = StringTape32::new();
        tape.push("hello").unwrap();
        tape.push("world").unwrap();

        assert_eq!(&tape[0], "hello");
        assert_eq!(&tape[1], "world");
    }

    #[test]
    fn test_into_iterator() {
        let mut tape = StringTape32::new();
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
    fn test_from_iterator() {
        let strings = vec!["hello", "world", "test"];
        let tape: StringTape32 = strings.into_iter().collect();

        assert_eq!(tape.len(), 3);
        assert_eq!(tape.get(0), Some("hello"));
        assert_eq!(tape.get(1), Some("world"));
        assert_eq!(tape.get(2), Some("test"));
    }

    #[test]
    fn test_extend() {
        let mut tape = StringTape32::new();
        tape.push("initial").unwrap();

        let additional = vec!["hello", "world"];
        tape.extend(additional).unwrap();

        assert_eq!(tape.len(), 3);
        assert_eq!(tape.get(0), Some("initial"));
        assert_eq!(tape.get(1), Some("hello"));
        assert_eq!(tape.get(2), Some("world"));
    }

    #[test]
    fn test_clear_and_truncate() {
        let mut tape = StringTape32::new();
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
    fn test_capacity() {
        let tape = StringTape32::with_capacity(100, 10).unwrap();
        assert_eq!(tape.data_capacity(), 100);
        assert_eq!(tape.capacity(), 0); // No strings added yet
    }

    #[test]
    fn test_custom_allocator() {
        let allocator = DefaultAllocator;
        let mut tape: StringTape<i32, DefaultAllocator> = 
            StringTape::new_in(allocator);
        
        tape.push("hello").unwrap();
        tape.push("world").unwrap();
        
        assert_eq!(tape.len(), 2);
        assert_eq!(tape.get(0), Some("hello"));
        assert_eq!(tape.get(1), Some("world"));
        
        // Verify we can access the allocator
        let _allocator_ref = tape.allocator();
    }

    #[test]
    fn test_custom_allocator_with_capacity() {
        let allocator = DefaultAllocator;
        let tape: StringTape<i64, DefaultAllocator> = 
            StringTape::with_capacity_in(256, 50, allocator).unwrap();
        
        assert_eq!(tape.data_capacity(), 256);
        assert!(tape.is_empty());
    }
}
