###################
Bank Marketing Data
###################

This dataset was sourced from the `UCI Machine Learning Repository <https://archive-beta.ics.uci.edu/ml/datasets/bank+marketing>`_.

This dataset is public available for research. The details are described in
[Moro et al., 2011].

Please include this citation if you plan to use this database:

[Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for
Bank Direct Marketing: An Application of the CRISP-DM Methodology.

In P. Novais et al. (Eds.), Proceedings of the European Simulation and
Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October,
2011. EUROSIS.

Available as:

* PDF - http://hdl.handle.net/1822/14838)
* BIB - http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt

**Source**

Created by: Paulo Cortez (Univ. Minho) and Sérgio Moro (ISCTE-IUL) @ 2012

==========
Past Usage
==========

The full dataset was described and analyzed in:

S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct
Marketing: An Application of the CRISP-DM Methodology.

In P. Novais et al. (Eds.), Proceedings of the European Simulation and
Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October,
2011. EUROSIS.

========
Overview
========

The data is related with direct marketing campaigns of a Portuguese banking
institution. The marketing campaigns were based on phone calls. Often, more
than one contact to the same client was required, in order to assess whether
the bank's term deposit product was subscribed to.

There are two datasets:

* ``bank-full.csv`` with all observations (45,211), ordered by date (from May
  2008 to November 2010).
* ``bank.csv`` with 10% of the observations (4,521), randomly selected from the
  full dataset

-------
Columns
-------

There are 17 total columns where 16 are observation attributes and column 17 is
the subscription outcome.

*There should be no missing Attribute Values*

+----+-----------+---------+--------------------------------------------------+
| #  | Name      | Type    | Description                                      |
+====+===========+=========+==================================================+
| 01 | age       | numeric | Contact's age in years                           |
+----+-----------+---------+--------------------------------------------------+
| 02 | job       | string  | Contact's type of job (admin, unknown,           |
|    |           |         | unemployed, management, housemaid, entrepreneur, |
|    |           |         | student, blue-collar, self-employed, retired,    |
|    |           |         | technician, services)                            |
+----+-----------+---------+--------------------------------------------------+
| 03 | marital   | string  | Contact's marital status (married, divorced,     |
|    |           |         | single) Note that divorced means divorced or     |
|    |           |         | widowed                                          |
+----+-----------+---------+--------------------------------------------------+
| 04 | education | string  | Contacts highest level of education (unknown,    |
|    |           |         | secondary, primary, tertiary)                    |
+----+-----------+---------+--------------------------------------------------+
| 05 | default   | string  | Whether contact has credit in default? (yes,no)  |
+----+-----------+---------+--------------------------------------------------+
| 06 | balance   | numeric | Contact's average yearly balance, in euros       |
+----+-----------+---------+--------------------------------------------------+
| 07 | housing   | string  | Whether contact has housing loan? (yes,no)       |
+----+-----------+---------+--------------------------------------------------+
| 08 | loan      | string  | Whether contact has personal loan? (yes,no)      |
+----+-----------+---------+--------------------------------------------------+
| 09 | contact   | string  | Communication type of contact (unknown,          |
|    |           |         | telephone, cellular)                             |
+----+-----------+---------+--------------------------------------------------+
| 10 | day       | numeric | Day of month of last contact                     |
+----+-----------+---------+--------------------------------------------------+
| 11 | month     | string  | Month of year of last contact (jan, feb, mar,    |
|    |           |         | ..., nov, dec)                                   |
+----+-----------+---------+--------------------------------------------------+
| 12 | duration  | numeric | Duration of last contact in seconds              |
+----+-----------+---------+--------------------------------------------------+
| 13 | campaign  | numeric | Number of contacts performed during this         |
|    |           |         | campaign and for this client (includes last      |
|    |           |         | contact)                                         |
+----+-----------+---------+--------------------------------------------------+
| 14 | pdays     | numeric | Number of days that passed by after the client   |
|    |           |         | was last contacted from a previous campaign      |
|    |           |         | (numeric, -1 means client was not previously     |
|    |           |         | contacted)                                       |
+----+-----------+---------+--------------------------------------------------+
| 15 | previous  | numeric | Number of contacts performed before this         |
|    |           |         | campaign and for this client (numeric)           |
+----+-----------+---------+--------------------------------------------------+
| 16 | poutcome  | string  | Outcome of the previous marketing campaign       |
|    |           |         | (unknown, other, failure, success)               |
+----+-----------+---------+--------------------------------------------------+
| 17 | y         | string  | Target outcome has the client subscribed a term  |
|    |           |         | deposit in the campaign? (yes, no)               |
+----+-----------+---------+--------------------------------------------------+
