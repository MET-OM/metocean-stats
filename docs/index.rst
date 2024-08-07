.. metocean-stats documentation master file, created by
   sphinx-quickstart on Thu Sep 14 10:18:36 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to metocean-stats's documentation!
=====================================

**metocean-stats** is a Python package (under development) for metocean analysis of NORA3 (wind and wave) hindcast.

The package contains functions that:
  * generate statistics (tables, diagrams etc)

Installing **metocean-stats**
=============================================
...

Create scatter Hs-Tp diagram:

.. code-block:: python
   
   tables.scatter_diagram(ds, var1='HS', 
                          step_var1=1, 
                          var2='TP', 
                          step_var2=1, 
                          output_file='Hs_Tp_scatter.csv')

.. csv-table:: Scatter diagram
   :header-rows: 1
   :file: files/Hs_Tp_scatter.csv

.. code-block:: python
   
   tables.scatter_diagram(ds, var1='HS', 
                          step_var1=1, 
                          var2='TP', 
                          step_var2=1, 
                          output_file='Hs_Tp_scatter.png')

.. image:: files/Hs_Tp_scatter.png
  :width: 500

...
