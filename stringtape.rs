use std::alloc::Layout;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Index;
use std::ptr::NonNull;
use std::slice;

#[derive(Debug)]
pub enum StringTapeError {
    OffsetOverflow,
    AllocationError,
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

impl std::error::Error for StringTapeError {}

pub struct StringTape<Offset: OffsetType = i32> {
    data: NonNull<u8>,
    offsets: NonNull<Offset>,
    capacity_bytes: usize,
    capacity_offsets: usize,
    len_bytes: usize,
    len_strings: usize,
    _phantom: PhantomData<Offset>,
}

pub trait OffsetType: Copy + Default + PartialOrd {
    const SIZE: usize;
    fn from_usize(value: usize) -> Option<Self>;
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

impl<Offset: OffsetType> StringTape<Offset> {
    pub fn new() -> Self {
        Self {
            data: NonNull::dangling(),
            offsets: NonNull::dangling(),
            capacity_bytes: 0,
            capacity_offsets: 0,
            len_bytes: 0,
            len_strings: 0,
            _phantom: PhantomData,
        }
    }

    pub fn with_capacity(
        data_capacity: usize,
        strings_capacity: usize,
    ) -> Result<Self, StringTapeError> {
        let mut tape = Self::new();
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
            unsafe { std::alloc::alloc(new_layout) }
        } else {
            let old_layout = Layout::from_size_align(self.capacity_bytes, 1).unwrap();
            unsafe { std::alloc::realloc(self.data.as_ptr(), old_layout, new_capacity) }
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
            Layout::from_size_align(new_capacity * Offset::SIZE, std::mem::align_of::<Offset>())
                .map_err(|_| StringTapeError::AllocationError)?;

        let new_ptr = if self.capacity_offsets == 0 {
            let ptr = unsafe { std::alloc::alloc(new_layout) };
            if ptr.is_null() {
                return Err(StringTapeError::AllocationError);
            }
            unsafe {
                std::ptr::write(ptr.cast::<Offset>(), Offset::default());
            }
            ptr
        } else {
            let old_layout = Layout::from_size_align(
                self.capacity_offsets * Offset::SIZE,
                std::mem::align_of::<Offset>(),
            )
            .unwrap();
            unsafe {
                std::alloc::realloc(self.offsets.as_ptr().cast(), old_layout, new_layout.size())
            }
        };

        self.offsets = NonNull::new(new_ptr.cast()).ok_or(StringTapeError::AllocationError)?;
        self.capacity_offsets = new_capacity;
        Ok(())
    }

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
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.data.as_ptr().add(self.len_bytes),
                bytes.len(),
            );
        }

        self.len_bytes += bytes.len();
        self.len_strings += 1;

        let offset = Offset::from_usize(self.len_bytes).ok_or(StringTapeError::OffsetOverflow)?;
        unsafe {
            std::ptr::write(self.offsets.as_ptr().add(self.len_strings), offset);
        }

        Ok(())
    }

    pub fn get(&self, index: usize) -> Option<&str> {
        if index >= self.len_strings {
            return None;
        }

        unsafe {
            let start_offset = if index == 0 {
                0
            } else {
                std::ptr::read(self.offsets.as_ptr().add(index)).to_usize()
            };
            let end_offset = std::ptr::read(self.offsets.as_ptr().add(index + 1)).to_usize();

            let slice = slice::from_raw_parts(
                self.data.as_ptr().add(start_offset),
                end_offset - start_offset,
            );

            Some(std::str::from_utf8_unchecked(slice))
        }
    }

    pub fn len(&self) -> usize {
        self.len_strings
    }

    pub fn is_empty(&self) -> bool {
        self.len_strings == 0
    }

    pub fn data_len(&self) -> usize {
        self.len_bytes
    }

    pub fn capacity(&self) -> usize {
        self.len_strings
    }

    pub fn data_capacity(&self) -> usize {
        self.capacity_bytes
    }

    pub fn clear(&mut self) {
        self.len_bytes = 0;
        self.len_strings = 0;
        if self.capacity_offsets > 0 {
            unsafe {
                std::ptr::write(self.offsets.as_ptr(), Offset::default());
            }
        }
    }

    pub fn truncate(&mut self, len: usize) {
        if len >= self.len_strings {
            return;
        }

        self.len_strings = len;
        self.len_bytes = if len == 0 {
            0
        } else {
            unsafe { std::ptr::read(self.offsets.as_ptr().add(len)).to_usize() }
        };
    }

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

    pub fn as_raw_parts(&self) -> (*const u8, *const Offset, usize, usize) {
        (
            self.data.as_ptr(),
            self.offsets.as_ptr(),
            self.len_bytes,
            self.len_strings,
        )
    }

    pub fn iter(&self) -> StringTapeIter<'_, Offset> {
        StringTapeIter {
            tape: self,
            index: 0,
        }
    }
}

impl<Offset: OffsetType> Drop for StringTape<Offset> {
    fn drop(&mut self) {
        if self.capacity_bytes > 0 {
            let layout = Layout::from_size_align(self.capacity_bytes, 1).unwrap();
            unsafe {
                std::alloc::dealloc(self.data.as_ptr(), layout);
            }
        }
        if self.capacity_offsets > 0 {
            let layout = Layout::from_size_align(
                self.capacity_offsets * Offset::SIZE,
                std::mem::align_of::<Offset>(),
            )
            .unwrap();
            unsafe {
                std::alloc::dealloc(self.offsets.as_ptr().cast(), layout);
            }
        }
    }
}

unsafe impl<Offset: OffsetType + Send> Send for StringTape<Offset> {}
unsafe impl<Offset: OffsetType + Sync> Sync for StringTape<Offset> {}

pub struct StringTapeIter<'a, Offset: OffsetType> {
    tape: &'a StringTape<Offset>,
    index: usize,
}

impl<'a, Offset: OffsetType> Iterator for StringTapeIter<'a, Offset> {
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

impl<'a, Offset: OffsetType> ExactSizeIterator for StringTapeIter<'a, Offset> {}

impl<Offset: OffsetType> FromIterator<String> for StringTape<Offset> {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let mut tape = StringTape::new();
        for s in iter {
            tape.push(&s).expect("Failed to build StringTape from iterator");
        }
        tape
    }
}

impl<'a, Offset: OffsetType> FromIterator<&'a str> for StringTape<Offset> {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        let mut tape = StringTape::new();
        for s in iter {
            tape.push(s).expect("Failed to build StringTape from iterator");
        }
        tape
    }
}

impl<Offset: OffsetType> Index<usize> for StringTape<Offset> {
    type Output = str;
    
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<'a, Offset: OffsetType> IntoIterator for &'a StringTape<Offset> {
    type Item = &'a str;
    type IntoIter = StringTapeIter<'a, Offset>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub type StringTape32 = StringTape<i32>;
pub type StringTape64 = StringTape<i64>;

#[cfg(test)]
mod tests {
    use super::*;

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
}
