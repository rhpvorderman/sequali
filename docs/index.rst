.. Sequali documentation master file, created by
   sphinx-quickstart on Mon Mar 25 14:47:42 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================
Welcome to Sequali's documentation!
===================================

.. contents:: Table of contents


==================
Introduction
==================

.. include:: includes/README.rst
   :start-after: .. introduction start
   :end-before: .. introduction end

==================
Supported formats
==================

.. include:: includes/README.rst
   :start-after: .. formats start
   :end-before: .. formats end

==================
Installation
==================

.. include:: includes/README.rst
   :start-after: .. installation start
   :end-before: .. installation end

==================
Quickstart
==================

.. include:: includes/README.rst
   :start-after: .. quickstart start
   :end-before: .. quickstart end

For a complete overview of the available command line options check the
usage below.

For more information about how the different modules see the
`Module option explanations`_.

==================
Usage
==================

.. argparse::
   :module: sequali.__main__
   :func: argument_parser
   :prog: sequali

.. include:: module_options.rst

==================
Acknowledgements
==================

.. include:: includes/README.rst
   :start-after: .. acknowledgements start
   :end-before: .. acknowledgements end

.. include:: includes/CHANGELOG.rst
