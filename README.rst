.. |python-version-shield| image:: https://img.shields.io/pypi/v/sequali.svg
  :target: https://pypi.org/project/sequali/
  :alt:

.. |conda-version-shield| image:: https://img.shields.io/conda/v/bioconda/sequali.svg
  :target: https://bioconda.github.io/recipes/sequali/README.html
  :alt:

.. |python-install-version-shield| image:: https://img.shields.io/pypi/pyversions/sequali.svg
  :target: https://pypi.org/project/sequali/
  :alt:

.. |license-shield| image:: https://img.shields.io/pypi/l/sequali.svg
  :target: https://github.com/rhpvorderman/sequali/blob/main/LICENSE
  :alt:

.. |docs-shield| image:: https://readthedocs.org/projects/sequali/badge/?version=latest
  :target: https://sequali.readthedocs.io/en/latest/?badge=latest
  :alt:

.. |coverage-shield| image:: https://codecov.io/gh/rhpvorderman/sequali/graph/badge.svg?token=MSR1A6BEGC
  :target: https://codecov.io/gh/rhpvorderman/sequali
  :alt:

.. |zenodo-shield| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10854010.svg
  :target: https://doi.org/10.5281/zenodo.10854010
  :alt:

|python-version-shield| |conda-version-shield| |python-install-version-shield|
|license-shield| |docs-shield| |coverage-shield| |zenodo-shield|

========
Sequali
========

.. introduction start

Sequence quality metrics for FASTQ and uBAM files.

Features:

+ Low memory footprint, small install size and fast execution times.
+ Informative graphs that allow for judging the quality of a sequence at
  a quick glance.
+ Overrepresentation analysis using 21 bp sequence fragments. Overrepresented
  sequences are checked against the NCBI univec database.
+ Estimate duplication rate using a `fingerprint subsampling technique which is
  also used in filesystem duplication estimation
  <https://www.usenix.org/system/files/conference/atc13/atc13-xie.pdf>`_.
+ Checks for 6 illumina adapter sequences and 17 nanopore adapter sequences
  for single read data.
+ Determines adapters by overlap analysis for paired read data.
+ Insert size metrics for paired read data.
+ Per tile quality plots for illumina reads.
+ Channel and other plots for nanopore reads.
+ FASTQ and unaligned BAM are supported. See "Supported formats".

Example reports:

+ `GM24385_1.fastq.gz <https://sequali.readthedocs.io/en/latest/GM24385_1.fastq.gz.html>`_;
  HG002 (Genome In A Bottle) on ultra-long Nanopore Sequencing. `Sequence file download <https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/UCSC_Ultralong_OxfordNanopore_Promethion/GM24385_1.fastq.gz>`_.

.. introduction end

For more information check `the documentation <https://sequali.readthedocs.io>`_.

Supported formats
=================

.. formats start

- FASTQ. Only the Sanger variation with a phred offset of 33 and the error rate
  calculation of 10 ^ (-phred/10) is supported. All sequencers use this
  format today.

  - Paired end sequencing data is supported.
  - For sequences called by illumina base callers an additional plot with the
    per tile quality will be provided.
  - For sequences called by guppy additional plots for nanopore specific
    data will be provided.
- unaligned BAM. Any alignment flags are currently ignored.

  - For uBAM data as delivered by dorado additional nanopore plots will be
    provided.

.. formats end

Installation
============

.. installation start

Installation via pip is available with::

    pip install sequali

Sequali is also distributed via bioconda. It can be installed with::

    conda install -c conda-forge -c bioconda sequali

.. installation end

Quickstart
==========

.. quickstart start

.. code-block::

    sequali path/to/my.fastq.gz

This will create a report ``my.fastq.gz.html`` and a json ``my.fastq.gz.json``
in the current working directory.

.. quickstart end

For all command line options checkout the
`usage documentation <https://sequali.readthedocs.io/#usage>`_.

For more extensive information about the module options check the
`documentation on the module options
<https://sequali.readthedocs.io/#module-option-explanations>`_.

Acknowledgements
================

.. acknowledgements start

+ `FastQC <https://www.bioinformatics.babraham.ac.uk/projects/fastqc/>`_ for
  its excellent selection of relevant metrics. For this reason these metrics
  are also gathered by Sequali.
+ The matplotlib team for their excellent work on colormaps. Their work was
  an inspiration for how to present the data and their RdBu colormap is used
  to represent quality score data. Check their `writings on colormaps
  <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_ for
  a good introduction.
+ Wouter de Coster for his `excellent post on how to correctly average phred
  scores <https://gigabaseorgigabyte.wordpress.com/2017/06/26/averaging-basecall-quality-scores-the-right-way/>`_.
+ Marcel Martin for providing very extensive feedback.

.. acknowledgements end

License
=======

.. license start

This project is licensed under the GNU Affero General Public License v3. Mainly
to avoid commercial parties from using it without notifying the users that they
can run it themselves. If you want to include code from Sequali in your
open source project, but it is not compatible with the AGPL, please contact me
and we can discuss a separate license.

.. license end