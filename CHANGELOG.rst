==========
Changelog
==========

.. Newest changes should be on top.

.. This document is user facing. Please word the changes in such a way
.. that users understand how the changes affect the new version.

version 0.2.0-dev
-----------------
+ The reverse complement of the canonical sequence is included in the
  overrepresented sequences table.
+ Make the number of threads configurable on the command line.
+ Fix build errors on windows

version 0.1.0
-----------------
+ In order to get overrepresented sequences across the entire read, reads
  are cut into fragments of 31 bp which are stored and counted. If the fragment
  store is full, only already stored sequences are counted. One in eight
  reads is processed this way.
+ Add fingerprint-based deduplication estimation based on `a technique used in
  filesystem deduplication estimation
  <https://www.usenix.org/system/files/conference/atc13/atc13-xie.pdf>`_.
+ Add a BAM parser to allow reading dorado produced unaligned BAM as well as
  already aligned BAM files.
+ Guess sequencing technology from the file header, so only appropriate
  adapters can be loaded in the adapter searcher. This improves speed.
+ Make an assortment of nanopore adapter probes that make it possible to
  distuinghish between nannopore adapters despite the nanopore adapters having
  a lot of shared subsequences.
+ Add a module to retrieve nanopore specific information from the header.
+ Classify overrepresented sequences by using NCBI's UniVec database and an
  assortment of nanopore adapters, ligation kits and primers.
+ Estimate duplication fractions based on counted unique sequences.
+ Add a JSON report
+ Add a progressbar powered by tqdm.
+ Implement a custom parser based on memchr for finding newlines.
+ Count overrepresented sequences using a hash table implemented in C.
+ Add a per tile sequence quality module.
+ Count adapters using a fast shift-AND algorithm.
+ Create diverse graphs using pygal based on the count matrix.
+ Implement base module using an optimised count matrix.