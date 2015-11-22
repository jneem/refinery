// Copyright 2015 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
This crate implements a simple and fast algorithm for building and refining a partition of the set
`[0usize, n)`. The most important type is `Partition` and its most important function is `refine`,
which refines the `Partition` according to some subset. There is also the function
`refine_with_callback` that does the same thing, but also lets you peek into what it's doing: you
provide it with a callback function which will be invoked every time a set of the partition is
split in two.

# Example
```rust
use refinery::Partition;

// Start with the single-element partition [0, 1, 2, 3, 4, 5, 6].
let mut part = Partition::simple(7);

// Refine it, bit by bit...
part.refine(&[0, 1, 2, 3]);
let result: &[&[usize]] = &[&[0, 1, 2, 3], &[4, 5, 6]];
assert_eq!(&part.iter().collect::<Vec<_>>()[..], result);

part.refine(&[3, 4, 5]);
// Note that the order of the sets is not defined (although each individual set is guaranteed to be
// sorted), so this assert is not really kosher.
let result: &[&[usize]] = &[&[3], &[4, 5], &[0, 1, 2], &[6]];
assert_eq!(&part.iter().collect::<Vec<_>>()[..], result);

// Refine it again, and this time print out some information each time a set gets split.
let alert = |original: &[usize], parts: (&[usize], &[usize])| {
    println!("splitting {:?} into {:?} and {:?}", original, parts.0, parts.1);
};
// This should print:
//     splitting [0, 1, 2] into [2] and [0, 1]
//     splitting [4, 5] into [4] and [5]
// Note that it doesn't say anything about splitting [3] into [3] because that isn't interesting.
part.refine_with_callback(&[2, 3, 4], alert);
```
*/

#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unsafe_code,
        unstable_features,
        unused_import_braces, unused_qualifications)]

use std::fmt::{Debug, Error, Formatter};
use std::{mem, usize};

/// A partition of a set of `usize`s. See the crate documentation for more information.
#[derive(Clone)]
pub struct Partition {
    // Contains the numbers 0..n in some order.
    elts: Vec<usize>,
    // A list of non-overlapping ranges. If `partition[i] = (j, k)` then the `i`th set in our
    // partition is `elts[j..k]`. We maintain the invariant that `elts[j..k]` is sorted.
    partition: Vec<(usize, usize)>,
    // If `rev_partition[i] == j` then `i` is in the `j`th set of our partition.
    rev_partition: Vec<usize>,

    // Some extra space that we allocate up front so that we don't need to allocate in our inner
    // loop.
    intersection: Vec<usize>,
    difference: Vec<usize>,
    active_sets: Vec<usize>,
}

/// An iterator over the sets in a `Partition`.
#[derive(Clone, Debug)]
pub struct PartitionIter<'a> {
    next_set_idx: usize,
    partition: &'a Partition,
}

impl Partition {
    /// Constructs a new `Partition`.
    ///
    /// `sets` is an iterator over the sets of the partition. The sets must be non-overlapping, and
    /// they must all be subsets of the half-open range `[0, size)`.
    ///
    /// # Panics
    ///  - if any sets overlap
    ///  - if any sets contain an element larger than or equal to `size`.
    pub fn new<I, J>(sets: I, size: usize) -> Partition
    where I: Iterator<Item=J>, J: Iterator<Item=usize> {
        let mut ret = Partition {
            elts: Vec::with_capacity(size),
            partition: Vec::with_capacity(size),
            rev_partition: vec![usize::MAX; size],
            intersection: Vec::with_capacity(size),
            difference: Vec::with_capacity(size),
            active_sets: Vec::with_capacity(size),
        };

        for set in sets {
            let set_idx = ret.partition.len();
            let set_start = ret.elts.len();
            for item in set {
                if item >= size {
                    panic!("all set elements must be smaller than `size`");
                } else if ret.rev_partition[item] != usize::MAX {
                    panic!("the element {} was repeated", item);
                } else {
                    ret.elts.push(item);
                    ret.rev_partition[item] = set_idx;
                }
            }
            ret.partition.push((set_start, ret.elts.len()));
        }

        for &(start, end) in &ret.partition {
            ret.elts[start..end].sort();
        }

        ret
    }

    /// Constructs a new `Partition` containing only one set (the half-open range `[0, size)`).
    pub fn simple(size: usize) -> Partition {
        Partition {
            elts: (0..size).into_iter().collect(),
            partition: vec![(0, size)],
            rev_partition: vec![0; size],
            intersection: Vec::with_capacity(size),
            difference: Vec::with_capacity(size),
            active_sets: Vec::with_capacity(size),
        }
    }

    /// Iterates over the sets in this partition, each of which is realized by a sorted `&[usize]`.
    pub fn iter<'a>(&'a self) -> PartitionIter<'a> {
        PartitionIter {
            next_set_idx: 0,
            partition: self,
        }
    }

    // Refines a single set (the one indexed by `set_idx`).
    fn refine_one<F>(&mut self, refiner: &[usize], set_idx: usize, cb: &mut F)
    where F: FnMut(&[usize], (&[usize], &[usize])) {
        self.intersection.clear();
        self.difference.clear();

        let (start, end) = self.partition[set_idx];
        { // Scope for `iter`, so that we don't borrow `self.elts` for too long.
            let mut iter = self.elts[start..end].iter().cloned().peekable();
            let mut last_refiner = 0usize;
            for &next_refiner in refiner {
                if next_refiner < last_refiner {
                    panic!("`refiner` must be sorted, but got {:?}", refiner);
                }

                // Every element strictly smaller than `next_refiner` goes into the difference. If
                // we find something equal to `next_refiner`, put in in the intersection.
                while let Some(&x) = iter.peek() {
                    if x < next_refiner {
                        self.difference.push(x);
                    } else if x == next_refiner {
                        self.intersection.push(x);
                    } else {
                        break;
                    }
                    iter.next();
                }
                last_refiner = next_refiner;
            }

            // Everything after the last element of `refiner` goes into the difference.
            self.difference.extend(iter);
        }

        // If the refinement was non-trivial, call the callback and update the partition.
        if !self.intersection.is_empty() && !self.difference.is_empty() {
            cb(&self.elts[start..end], (&self.intersection[..], &self.difference[..]));

            let mid = start + self.intersection.len();
            debug_assert!(mid + self.difference.len() == end);

            self.partition[set_idx].1 = mid;
            self.partition.push((mid, end));

            // TODO: once we have a stable way of copying slices, use that.
            for (i, &x) in self.intersection.iter().enumerate() {
                self.elts[start + i] = x;
            }
            for (i, &x) in self.difference.iter().enumerate() {
                self.elts[mid + i] = x;
            }

            for &x in self.difference.iter() {
                self.rev_partition[x] = self.partition.len() - 1;
            }
        }
    }

    /// Refines the partition with the given set.
    ///
    /// Every set `s` in the partition is replaced by `s.intersection(refiner)` and
    /// `s.difference(refiner)`.
    ///
    /// This function runs in time that is O(n log n) in the size of `refiner`, which must be
    /// sorted.
    ///
    /// # Panics
    ///  - if `refiner` contains any elements that are not in the partition
    ///  - if `refiner` is not sorted
     pub fn refine(&mut self, refiner: &[usize]) {
        self.refine_with_callback(refiner, |_, _| {});
    }

    /// Refines the partition with the given set.
    ///
    /// Every set `s` in the partition is replaced by `s.intersection(refiner)` and
    /// `s.difference(refiner)`. Every time we make such a non-trivial (meaning that both the
    /// intersection and the difference are non-empty) replacement, we call the supplied callback
    /// function with arguments `original_set` and `(intersection, difference)`.
    ///
    /// This function runs in time that is O(n log n) in the size of `refiner`, which must be
    /// sorted.
    ///
    /// # Panics
    ///  - if `refiner` contains any elements that are not in the partition
    ///  - if `refiner` is not sorted
    pub fn refine_with_callback<F>(&mut self, refiner: &[usize], mut split_callback: F)
    where F: FnMut(&[usize], (&[usize], &[usize])) {
        // Find the indices of all sets that we need to touch. This is key to being fast, because
        // it means we don't need to iterate over all sets in our partition.
        self.active_sets.clear();
        for &x in refiner.iter() {
            if x >= self.rev_partition.len() {
                panic!("`refiner` went out of bounds: {:?}", refiner);
            }
            self.active_sets.push(self.rev_partition[x]);
        }
        self.active_sets.sort();
        self.active_sets.dedup();

        // Swap out `self.active_sets` temporarily, so that we can iterate over it and modify
        // `self` without the borrow checker flipping out. I'm pretty sure this doesn't allocate.
        let mut tmp = Vec::new();
        mem::swap(&mut tmp, &mut self.active_sets);

        for &set_idx in &tmp {
            self.refine_one(refiner, set_idx, &mut split_callback);
        }
        mem::swap(&mut tmp, &mut self.active_sets);
    }
}

// This only exists so we can give a `Debug` impl that displays a slice like a set.
struct SliceSet<'a>(&'a [usize]);

impl<'a> Debug for SliceSet<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        fmt.debug_set().entries(self.0.iter()).finish()
    }
}

impl Debug for Partition {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        fmt.debug_set()
            .entries(self.partition.iter()
                     .map(|&(start, end)| { SliceSet(&self.elts[start..end]) }))
            .finish()
    }
}

impl<'a> Iterator for PartitionIter<'a> {
    type Item = &'a [usize];
    fn next(&mut self) -> Option<Self::Item> {
        if self.next_set_idx < self.partition.partition.len() {
            let (start, end) = self.partition.partition[self.next_set_idx];
            self.next_set_idx += 1;
            Some(&self.partition.elts[start..end])
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a Partition {
    type Item = &'a [usize];
    type IntoIter = PartitionIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(sets: &[&[usize]], size: usize) -> Partition {
        Partition::new(sets.iter().map(|set| set.iter().cloned()), size)
    }

    #[test]
    fn test_debug() {
        fn format(sets: &[&[usize]], size: usize) -> String {
            format!("{:?}", make(sets, size))
        }

        assert_eq!(format(&[], 0), "{}");
        assert_eq!(format(&[&[0, 1, 2]], 3), "{{0, 1, 2}}");
        assert_eq!(format(&[&[0], &[1, 2]], 3), "{{0}, {1, 2}}");
    }

    #[test]
    #[should_panic]
    fn test_new_panics_on_overlap() {
        make(&[&[0, 1, 2], &[2]], 3);
    }

    #[test]
    #[should_panic]
    fn test_new_panics_on_out_of_bounds() {
        make(&[&[0, 1], &[2]], 2);
    }

    #[test]
    fn test_refine() {
        let mut part = Partition::simple(5);
        part.refine_with_callback(&[1, 3, 4], |orig, (int, diff)| {
            assert_eq!(orig, &[0, 1, 2, 3, 4]);
            assert_eq!(int, &[1, 3, 4]);
            assert_eq!(diff, &[0, 2]);
        });
        part.refine_with_callback(&[1], |orig, (int, diff)| {
            assert_eq!(orig, &[1, 3, 4]);
            assert_eq!(int, &[1]);
            assert_eq!(diff, &[3, 4]);
        });

        let mut times_called = 0usize;
        part.refine_with_callback(&[0, 2], |_, _| { times_called += 1; });
        assert_eq!(times_called, 0);
        part.refine_with_callback(&[0, 3], |_, _| { times_called += 1; });
        assert_eq!(times_called, 2);
    }

    #[test]
    #[should_panic]
    fn rest_refine_panics_on_unsorted() {
        Partition::simple(5).refine(&[3, 2, 1]);
    }

    #[test]
    #[should_panic]
    fn rest_refine_panics_on_out_of_bounds() {
        Partition::simple(5).refine(&[1, 5]);
    }
}
