==========
Changelog
==========

.. Newest changes should be on top.

.. This document is user facing. Please word the changes in such a way
.. that users understand how the changes affect the new version.

version 0.8.0
-----------------
+ A citation file was added to the repository.
+ Calculate insert sizes and used adapters based on overlap between the
  read pairs.
+ Both reads from paired-end reads are taken into consideration when
  evaluating the duplication rate.
+ Support for paired-end reads added.
+ Minor performance improvement by providing a non-temporal cache hint in the
  QCMetrics module.

version 0.7.1
-----------------
+ Fix a small visual bug in the report sidebar.
+ PyGAL report htmls are now fully HTML5 compliant. HTML5 validation has been
  made a part of the integration testing.

version 0.7.0
-----------------
+ Image files can now be saved as SVG files.
+ The javascript file for the tooltip highlighting is now embedded in the
  html file so no internet access is needed for the functionality.
+ A sidebar with a table of contents is added to the report for easier
  navigation.
+ Graph fonts are made a little bigger. Graphs now respond to zooming in and
  out on the web page.
+ Enable building on ARM platforms such as M1 macintosh and Aarch64.
+ Speedup the overrepresented sequences module by adding an AVX2 k-mer
  construction algorithm.

version 0.6.0
-----------------
+ Add links to the documentation in the report.
+ Moved documentation to readthedocs and added extensive module documentation.
+ Change the ``-deduplication-estimate-bits`` to a more understandable
  ``--duplication-max-stored-fingerprints``.
+ Add a small table that lists how many reads are >=Q5, >=Q7 etc. in the
  per sequence average quality report.
+ The progressbar can track progress through more file formats.
+ The deduplication fingerprint that is used is now configurable from the
  command line.
+ The deduplication module starts by gathering all sequences rather than half
  of the sequences. This allows all sequences to be considered using a big
  enough hash table.

version 0.5.1
-----------------
+ Fix a bug in the overrepresented sequence sampling where the fragments from
  the back half of the sequence were incorrectly sampled. Leading to the last
  fragment being sampled over and over again.

version 0.5.0
-----------------
+ Base the percentage in the overrepresented sequences section on the number
  of found fragments divided by the number of sampled sequences. Previously
  this was based on the number of sampled fragments, which led to very low
  percentages for long read sequences, whilst also being less intuitive to
  understand. There were some inconsistencies in the documentation about this
  that are now fixed.
+ Add a new `meta` section to the JSON report to allow integration with
  `MultiQC <https://github.com/multiqc/MultiQC>`_.
+ Add all nanopore barcode sequences and native adapters to the contaminants.
+ Add native adapters to the adapter search.

version 0.4.1
-----------------
+ Fixed an issue that caused an off by one error if start and end time
  of a Nanopore run were at certain intervals.

version 0.4.0
-----------------
+ Fix bugs that were triggered when empty reads were present on
  illumina and nanopore platforms.
+ Fix a bug that was triggered when a single nucleotide read was present on
  a nanopore platform.
+ Add a ``--version`` command line flag.
+ Add an ``--adapter-file`` file flag which can be used to set custom adapter
  files by users.

version 0.3.0
-----------------
+ Fingerprint using offsets of 64 bases from both ends of the sequence.
  On nanopore sequencing this prevents taking into account adapter sequences
  for the duplication estimate. It also prevents taking sequences from the
  error-prone regions. The fingerprint consists of two 8 bp sequences rather
  than the two 16 bp sequences that were used before. This made the fingerprint
  less prone to sequencing errors, especially in long read sequencing
  technologies. As a result the duplication estimate on nanopore reads
  should be more accurate.
+ Added a small header with information on where to submit bug reports.
+ Use different adapter probes for nanopore adapters, such that the probes
  do occur at some distance from the strand extremities. The start and end
  of nanopore sequences are prone to errors and this hindered adapter
  detection.
+ Distinguish between top and bottom adapters for the adapter occurrence plot.
+ Update pygal to 3.0.4 to prevent installation errors on Python 3.12.
+ Fix several divide by 0 errors that occurred on empty reads and empty files.
+ Change default fragment length from 31 to 21 which increases the sensitivity
  of the overrepresented sequences module.

version 0.2.0
-----------------
+ Fixed a crash that occurred in the illumina header checking code on
  illumina headers without the comment part.
+ ``--max-unique-sequences`` flag replaced with
  ``--overrepresentation-max-unique-fragments`` to be consistent with the
  report and other flags.
+ Lots of formatting improvements were made to the report:

  + The quality distribution plot now use Matplotlib's RdBu colormap. Like
    the old colormap, it goes from red to blue via white, but is much
    clearer visually.
  + Tables now have zebra-style coloring and mouse-over coloring to clearly
    distinguish rows.
  + The base content plot now uses a green and blue color scheme for GC and
    AT bases respectively. Previously it was red and blue.
  + Sans-serif fonts used throughout the report.
  + Explanation paragraphs are now in a smaller font and italic to visually
    distuingish them from data generated specifically for the sequencing
    file.
  + Plots are now rendered in sans-serif rather than monospace fonts.
  + Minor formatting, spelling and style issues were fixed.
+ The programs CLI help messages have been improved by clearer phrasing,
  better metavar names and consistent punctuation.
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
