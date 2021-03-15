Installation
============

Installing with pip
-------------------
We assume you have the default Python 3 environment already configured on your
system and you intend to install `ProMS` inside of it.
If you want to create and work with Python virtual environments, 
please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

First, make sure you have the latest version of ``pip`` (the Python 3 package manager)
installed. If you do not, refer to the `pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

Install the current release of `ProMS` with ``pip``:

.. code-block:: none

  $ pip install proms

To upgrade to a newer release use the ``--upgrade`` flag:

.. code-block:: none

  $ pip install --upgrade proms

If you do not have permission to install software system wide, you can
install into your user directory using the ``--user`` flag:

.. code-block:: none

  $ pip install --user proms

To install the development version, you can use:

.. code-block:: none

  $ pip install git+https://github.com/bzhanglab/proms


