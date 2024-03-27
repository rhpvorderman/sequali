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

Overrepresentated Sequences Module
----------------------------------

Duplication Estimation Module
-----------------------------


