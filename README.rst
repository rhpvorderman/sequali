.. image:: https://img.shields.io/pypi/v/sequali.svg
  :target: https://pypi.org/project/sequali/
  :alt:

.. image:: https://img.shields.io/conda/v/bioconda/sequali.svg
  :target: https://bioconda.github.io/recipes/sequali/README.html
  :alt:

.. image:: https://img.shields.io/pypi/pyversions/sequali.svg
  :target: https://pypi.org/project/sequali/
  :alt:

.. image:: https://img.shields.io/pypi/l/sequali.svg
  :target: https://github.com/rhpvorderman/sequali/blob/main/LICENSE
  :alt:

========
sequali
========
Sequence quality metrics

Features:

+ Low memory footprint, small install size and fast execution times.
+ Informative graphs that allow for judging the quality of a sequence at
  a quick glance.
+ Overrepresentation analysis using 21 bp sequence fragments. Overrepresented
  sequences are checked against the NCBI univec database.
+ Estimate duplication rate using a `fingerprint subsampling technique which is
  also used in filesystem duplication estimation
  <https://www.usenix.org/system/files/conference/atc13/atc13-xie.pdf>`_.
+ Checks for 6 illumina adapter sequences and 17 nanopore adapter sequences.
+ Per tile quality plots for illumina reads.
+ Channel and other plots for nanopore reads.
+ FASTQ and unaligned BAM are supported. See "Supported formats".

Example reports:

+ `GM24385_1.fastq.gz <https://github.com/rhpvorderman/sequali/files/14617717/GM24385_1.fastq.gz.html.zip>`_;
  HG002 (Genome In A Bottle) on ultra-long Nanopore Sequencing. `Sequence file download <https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/UCSC_Ultralong_OxfordNanopore_Promethion/GM24385_1.fastq.gz>`_.

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

Installation via pip is available with::

    pip install sequali

Sequali is also distributed via bioconda. It can be installed with::

    conda install -c conda-forge -c bioconda sequali

Usage
=====

.. code-block::

    usage: sequali [-h] [--json JSON] [--html HTML] [--outdir OUTDIR]
                   [--adapter-file ADAPTER_FILE]
                   [--overrepresentation-threshold-fraction FRACTION]
                   [--overrepresentation-min-threshold THRESHOLD]
                   [--overrepresentation-max-threshold THRESHOLD]
                   [--overrepresentation-max-unique-fragments N]
                   [--overrepresentation-fragment-length LENGTH]
                   [--overrepresentation-sample-every DIVISOR]
                   [--deduplication-estimate-bits BITS] [-t THREADS] [--version]
                   INPUT

    Create a quality metrics report for sequencing data.

    positional arguments:
      INPUT                 Input FASTQ or uBAM file. The format is autodetected
                            and compressed formats are supported.

    options:
      -h, --help            show this help message and exit
      --json JSON           JSON output file. default: '<input>.json'.
      --html HTML           HTML output file. default: '<input>.html'.
      --outdir OUTDIR, --dir OUTDIR
                            Output directory for the report files. default:
                            current working directory.
      --adapter-file ADAPTER_FILE
                            File with adapters to search for. See default file for
                            formatting. Default: src/sequali/adapters/adapter_list.tsv.
      --overrepresentation-threshold-fraction FRACTION
                            At what fraction a sequence is determined to be
                            overrepresented. The threshold is calculated as
                            fraction times the number of sampled sequences.
                            Default: 0.001 (1 in 1,000).
      --overrepresentation-min-threshold THRESHOLD
                            The minimum amount of occurrences for a sequence to be
                            considered overrepresented, regardless of the bound
                            set by the threshold fraction. Useful for smaller
                            files. Default: 100.
      --overrepresentation-max-threshold THRESHOLD
                            The amount of occurrences for a sequence to be
                            considered overrepresented, regardless of the bound
                            set by the threshold fraction. Useful for very large
                            files. Default: unlimited.
      --overrepresentation-max-unique-fragments N
                            The maximum amount of unique fragments to store.
                            Larger amounts increase the sensitivity of finding
                            overrepresented sequences at the cost of increasing
                            memory usage. Default: 5,000,000.
      --overrepresentation-fragment-length LENGTH
                            The length of the fragments to sample. The maximum is
                            31. Default: 21.
      --overrepresentation-sample-every DIVISOR
                            How often a read should be sampled. More samples leads
                            to better precision, lower speed, and also towards
                            more bias towards the beginning of the file as the
                            fragment store gets filled up with more sequences from
                            the beginning. Default: 1 in 8.
      --deduplication-estimate-bits BITS
                            Determines how many sequences are maximally stored to
                            estimate the deduplication rate. Maximum stored
                            sequences: 2 ** bits * 7 // 10. Memory required: 2 **
                            bits * 24. Default: 21.
      -t THREADS, --threads THREADS
                            Number of threads to use. If greater than one sequali
                            will use an additional thread for gzip decompression.
                            Default: 2.
      --version             show program's version number and exit

Acknowledgements
================
+ `FastQC <https://www.bioinformatics.babraham.ac.uk/projects/fastqc/>`_ for
  its excellent selection of relevant metrics. For this reason these metrics
  are also gathered by sequali.
+ The matplotlib team for their excellent work on colormaps. Their work was
  an inspiration for how to present the data and their RdBu colormap is used
  to represent quality score data. Check their `writings on colormaps
  <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_ for
  a good introduction.
+ Wouter de Coster for his `excellent post on how to correctly average phred
  scores <https://gigabaseorgigabyte.wordpress.com/2017/06/26/averaging-basecall-quality-scores-the-right-way/>`_.
+ Marcel Martin for providing very extensive feedback.

License
=======

This project is licensed under the GNU Affero General Public License v3. Mainly
to avoid commercial parties from using it without notifying the users that they
can run it themselves. If you want to include code from sequali in your
open source project, but it is not compatible with the AGPL, please contact me
and we can discuss a separate license.
