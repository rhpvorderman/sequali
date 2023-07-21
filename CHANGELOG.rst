==========
Changelog
==========

.. Newest changes should be on top.

.. This document is user facing. Please word the changes in such a way
.. that users understand how the changes affect the new version.

version 0.1.0-dev
-----------------
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