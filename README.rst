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

    sequali /path/to/fastq/file > result.html

License
=======

This project is licensed under the GNU Affero General Public License v3. Mainly
to avoid commercial parties from using it without notifying the users that they
can run it themselves. If you want to include code from sequali in your
open source project, but it is not compatible with the AGPL, please contact me
and we can discuss a separate license.
