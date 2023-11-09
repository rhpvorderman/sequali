========
sequali
========
Sequence quality metrics

Features:

+ Low memory footprint, small install size and fast execution times.
+ Informative graphs that allow for judging the quality of a sequence at
  a quick glance.
+ Overrepresentation analysis using 31 bp sequence fragments. Overrepresented
  sequences are checked against the NCBI univec database.
+ Estimate duplication rate using a `fingerprint subsampling technique which is
  also used in filesystem duplication estimation
  <https://www.usenix.org/system/files/conference/atc13/atc13-xie.pdf>`_.
+ Checks for 6 illumina adapter sequences and 15 nanopore adapter sequences.
+ Per tile quality plots for illumina reads.
+ Channel and other plots for nanopore reads.
+ FASTQ and unaligned BAM are supported. See "Supported formats".

Supported formats
=================
- FASTQ. Only the Sanger variation with a phred offset of 33 and the error rate
  calculation of 10 ^ (-phred/10) is supported. All sequencers use this
  format today.

  - For sequences called by illumina base callers an additional plot with the
    per tile quality will be provided.
  - For sequences called by guppy additional plots for nanopore specific
    data will be provided.

- unaligned BAM. Any alignment flags are currently ignored.

  - For uBAM data as delivered by dorado additional nanopore plots will be
    provided.

Installation
============

    pip install git+https://github.com/rhpvorderman/sequali.git

Usage
=====

    usage: sequali [-h] [--json JSON] [--html HTML] [--dir DIR]
                   [--overrepresentation-threshold-fraction OVERREPRESENTATION_THRESHOLD_FRACTION]
                   [--overrepresentation-min-threshold OVERREPRESENTATION_MIN_THRESHOLD]
                   [--overrepresentation-max-threshold OVERREPRESENTATION_MAX_THRESHOLD]
                   [--max-unique-sequences MAX_UNIQUE_SEQUENCES]
                   [--overrepresentation-fragment-length OVERREPRESENTATION_FRAGMENT_LENGTH]
                   [--overrepresentation-sample-every OVERREPRESENTATION_SAMPLE_EVERY]
                   [--deduplication-estimate-bits DEDUPLICATION_ESTIMATE_BITS]
                   input

    positional arguments:
      input                 Input FASTQ file

    options:
      -h, --help            show this help message and exit
      --json JSON           JSON output file. default: '<input>.json'
      --html HTML           HTML output file. default: '<input>.html'
      --dir DIR             Output directory. default: current working directory
      --overrepresentation-threshold-fraction OVERREPRESENTATION_THRESHOLD_FRACTION
                            At what fraction a sequence is determined to be
                            overrepresented. Default: 0.0001 (1 in 100 000).
      --overrepresentation-min-threshold OVERREPRESENTATION_MIN_THRESHOLD
                            The minimum amount of sequences that need to be
                            present to be considered overrepresented even if the
                            threshold fraction is surpassed. Useful for smaller
                            files. Default: 100
      --overrepresentation-max-threshold OVERREPRESENTATION_MAX_THRESHOLD
                            The threshold above which a sequence is considered
                            overrepresented even if the threshold fraction is not
                            surpassed. Useful for very large files. Default:
                            unlimited.
      --max-unique-sequences MAX_UNIQUE_SEQUENCES
                            The maximum amount of unique fragments to gather.
                            Larger amounts increase the sensitivity of finding
                            overrepresented sequences at the cost of increasing
                            memory usage. Default: 5,000,000
      --overrepresentation-fragment-length OVERREPRESENTATION_FRAGMENT_LENGTH
                            The length of the fragments to sample. The maximum is
                            31. Default: 31.
      --overrepresentation-sample-every OVERREPRESENTATION_SAMPLE_EVERY
                            How often a read should be sampled. Default: 1 in 8.
                            More samples leads to better precision, lower speed,
                            and also towards more bias towards the beginning of
                            the file as the fragment store gets filled up with
                            more sequences from the beginning.
      --deduplication-estimate-bits DEDUPLICATION_ESTIMATE_BITS
                            Determines how many sequences are maximally stored to
                            estimate the deduplication rate. Maximum stored
                            sequences: 2 ** bits * 7 // 10. Memory required: 2 **
                            bits * 24. Default: 21.

Acknowledgements
================
+ `FastQC <https://www.bioinformatics.babraham.ac.uk/projects/fastqc/>`_ for
  its excellent selection of relevant metrics. For this reason these metrics
  are also gathered by sequali.
+ Wouter de Coster for his `excellent post on how to correctly average phred
  scores <https://gigabaseorgigabyte.wordpress.com/2017/06/26/averaging-basecall-quality-scores-the-right-way/>`_.

License
=======

This project is licensed under the GNU Affero General Public License v3. Mainly
to avoid commercial parties from using it without notifying the users that they
can run it themselves. If you want to include code from sequali in your
open source project, but it is not compatible with the AGPL, please contact me
and we can discuss a separate license.
