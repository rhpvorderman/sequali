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

.. |zenodo-shield| image:: ./docs/_static/images/doi_image.svg
  :target: https://doi.org/10.1093/bioadv/vbaf010
  :alt:

|python-version-shield| |conda-version-shield| |python-install-version-shield|
|license-shield| |docs-shield| |coverage-shield| |zenodo-shield|

========
Sequali
========

.. introduction start

Sequence quality metrics for FASTQ and uBAM files.

Features:

+ `MultiQC <https://multiqc.info>`_ support since MultiQC version 1.22.
+ Low memory footprint, small install size and fast execution times.

  + Sequali typically needs less than 2 GB of memory and 3-30 minutes runtime
    when run on 2 cores (the default).
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
+ Reproducible reports without timestamps.

Example reports:

+ `GM24385_1.fastq.gz <https://sequali.readthedocs.io/en/latest/GM24385_1.fastq.gz.html>`_;
  HG002 (Genome In A Bottle) on ultra-long Nanopore Sequencing. ENA accession:
  `ERR3988483 <https://www.ebi.ac.uk/ena/browser/view/ERR3988483>`_.
+ `GM24385_1_cut.fastq.gz <https://sequali.readthedocs.io/en/latest/GM24385_1_cut.fastq.gz.html>`_;
  ``GM24385_1.fastq.gz`` processed with cutadapt:
  ``cutadapt -o GM24385_1_cut.fastq.gz --cut -64 --cut 64 --minimum-length 500 -Z --max-aer 0.1 GM24385_1.fastq.gz``.
  The resulting file has 64 bp cut off from both its ends and after that
  filtered for a minimum length of 500 and a maximum average error rate of 0.1.
+ `21C125_R1.fastq.gz <https://sequali.readthedocs.io/en/latest/21C125_R1.fastq.gz.html>`_;
  Illumina NovaSeq X paired-end sequencing of *Campylobacter jejuni*. ENA accession:
  `ERR11204024 <https://www.ebi.ac.uk/ena/browser/view/ERR11204024>`_.

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
- (unaligned) BAM with single reads. Read-pair information is currently ignored.

  - For BAM data as delivered by dorado additional nanopore plots will be
    provided.
  - For aligned BAM files, secondary and supplementary reads are ignored
    similar to how ``samtools fastq`` handles the data.

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

To set the directory where the reports are created the ``--outdir`` flag can
be used. This is useful when using [MultiQC](https://github.com/multiqc/multiqc).

.. code-block::

    sequali --out-dir /my/dir/all_sequali_reports my.fastq.gz

The html and json filenames can be set separately.

.. code-block::

    sequali --html before_qc.html --json before_qc.json my.fastq.gz
    sequali --html after_qc.html --json after_qc.json my.cutadapt.fastq.gz

Sequali can handle paired-end data.

.. code-block::

    sequali /sequencing_data/sample100_R1.fastq.gz /sequencing_data/sample100_R2.fastq.gz

Additionally sequali can handle BAM data. Proper pair handling is not yet supported for
BAM data, so this is primarily useful for ONT datasets.

.. code-block::

    sequali /sequencing_data/sample100_dorado_called_hac_v4.30.bam

Sequali by default uses one thread per compressed input file and one thread for
the read processing, typically keeping two cores busy. Sequali can also use a single
core, which is slower, but typically more efficient for HPC scenarios where
multiple files can be run simultaneously. (Below a SLURM example.)

.. code-block::

    sbatch -c 1 --time 59 --partition short \
    --wrap 'sequali --threads 1 /cluster-scratch/myusername/my.fastq.gz'

Using a thread count higher than ``2`` has no effect. Due to the decompression
bottleneck, bringing the full power of multithreading to Sequali has limited
utility whilst having a disproportionally high cost in additional code
complexity.

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
  scores <https://gigabaseorgigabyte.wordpress.com/2017/06/26/averaging-basecall-quality-scores-the-right-way/>`_
  as well as the idea for using end-anchored plots from `NanoQC
  <https://github.com/wdecoster/nanoQC>`_.
+ Marcel Martin for providing very extensive feedback.
+ Agnès Barnabé for creating a Galaxy wrapper.

.. acknowledgements end

Citation
========
.. citation start

If you wish to credit Sequali please cite `the Sequali article
<https://doi.org/10.1093/bioadv/vbaf010>`_.

.. citation end

License
=======

.. license start

This project is licensed under the GNU Affero General Public License v3. Mainly
to avoid commercial parties from using it without notifying the users that they
can run it themselves. If you want to include code from Sequali in your
open source project, but it is not compatible with the AGPL, please contact me
and we can discuss a separate license.

.. license end