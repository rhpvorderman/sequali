========
sequali
========
Sequence quality metrics

FASTQ Quality control with the following goals:

+ Informative graphs that allow for judging the quality of a sequence at
  a quick glance.
+ Correct interpretation of Phred quality scores. For nanopore QC programs this
  is already quite common as the basecallers provide the correct average
  quality as metadata. Unfortunately, not all programs do this correctly.
  For more explanation check `this excellent blog post
  <https://gigabaseorgigabyte.wordpress.com/2017/06/26/averaging-basecall-quality-scores-the-right-way/>`_
  from Wouter de Coster (nanoplot author).
+ Low resource usage in terms of CPU, memory and install size.

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
                            files.
      --overrepresentation-max-threshold OVERREPRESENTATION_MAX_THRESHOLD
                            The threshold above which a sequence is considered
                            overrepresented even if the threshold fraction is not
                            surpassed. Useful for very large files.
      --max-unique-sequences MAX_UNIQUE_SEQUENCES
                            The maximum amount of unique sequences to gather.
                            Larger amounts increase the sensitivity of finding
                            overrepresented sequences and increase the accuracy of
                            the duplication estimate, at the cost of increasing
                            memory usage at about 50 bytes per sequence.


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
