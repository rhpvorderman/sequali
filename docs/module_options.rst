==========================
Module option explanations
==========================

Adapter Content Module
----------------------

The adapter content module searches for adapter stubs that are 12 bp in length.
These adapter probes are saved in the default adapter file which has the
following structure:

.. csv-table:: adapter_file.tsv
    :header: "#Name", "Sequencing Technology", "Probe sequence", "sequence position"

    "Illumina Universal Adapter", "illumina", "AGATCGGAAGAG", "end"
    "Illumina Small RNA 3' adapter", "illumina", "TGGAATTCTCGG", "end"

All empty rows and rows starting with ``#`` are ignored. The file is tab
separated. The columns are as follows:

+ Name: The name of the sequence that shows up in the report.
+ Sequencing Technology: The name of the technology, currently ``illumina``,
  ``nanopore`` and ``all`` are supported. Sequali detects the technology from
  the file header and only loads the appropriate adapters and adapters with
  ``all``.
+ Probe sequence: the sequence to probe for. Can be up to 64 bp in length.
  Since exact matching is used false postives versus false negatives need to
  be weighed when considering probe length.
+ Sequence position: Whether the adapter occurs at the begin or end. In the
  resulting adapter graph, counts for this adapter will accumulate towards the
  begin or end depending on this field.

A new adapter file can be set with the ``--adapter-file`` flag on the CLI.

Overrepresented Sequences Module
----------------------------------
Determining overrepresented sequences is challenging. One way is to take
all the k-mers of each sequence and count all the k-mer occurences. To avoid
issues with read orientation the canonical k-mers should be taken [#F1]_.
Storing all kmers and counting them is very compute intensive as a k-mer has to
be calculated and stored for every position in the sequence.

Sequali therefore divides a sequence in fragments of length k. Unlike k-mers
which are overlapping, this ensures that each part of the sequence is
represented by just one fragment. The disadvantage is that these fragments
can be caught in different frames, unlikely k-mers which capture all possible
frames for length k. This hampers the detection rate.

Since most overrepresented sequences will be adapter and helper sequences
and since most of these sequences will be anchored at the beginning and end
of the read, this problem is alleviated by capturing the fragments from the
ends towards the middle. This means that the first and last fragment will
always be the first 21 bp of the beginning and the last 21 bp in the end. As
such the adapter sequences will always be sampled in the same frame.

.. figure:: _static/images/overrepresented_sampling.svg

    This figure shows how fragments are sampled in sequali. The silver elements
    represent the fragments. Sequence #1 is longer and hence more fragments are
    sampled. Despite the length differences between sequence #1 and sequence #2
    the fragments for the adapter and barcode sequences are the same.
    In Sequence #1 the fragments sampled from the back end overlap somewhat
    with sequences from the front end. This is a necessity to ensure all of the
    sequence is sampled when the length is not divisible by the size of the
    fragments.

Fragments are stored and counted in a hash table. When the hash table is full
only fragments that are already present will be counted. To diminish the time
spent on the module, by default 1 in 8 sequences is analysed.

After the module is run, stored fragments are checked for their counts. If the
count exceeds a certain threshold it is considered overrepresented. Sequali
does a k-mer analysis of the sequences and compares that with sequences from
the NCBI UniVec database to determine possible origins.

The following command line parameters affect this module:

+ ``--overrepresentation-threshold-fraction``: If count / total exceeds this
  fraction, the fragment is considered overrepresented.
+ ``--overrepresentation-min-threshold``: The minimum count to be considered
  overrepresented.
+ ``--overrepresentation-max-threshold``: The maximum count to be considered
  overrepresented. On large libraries with billions of sampled fragments this
  can be used to force detection for certain counts regardless of threshold.
+ ``--overrepresentation-max-unique-fragments``: The amount of fragments to
  store.
+ ``--overrepresentation-sample-every``: How often a sequence is sampled. Default
  is every 8 sequences.

.. [#F1] A canonical k-mer is the k-mer that has the lowest sort order compared
         to itself and its reverse complement. This way the canonical-kmer is
         always the same regardless of read orientation.

Duplication Estimation Module
-----------------------------


