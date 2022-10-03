##########################
Professional Training Data
##########################

See (NYU NBER data)[https://users.nber.org/~rdehejia/nswdata2.html] for hosted
data source.

All the source text files were converted to CSV and Parquet files with headers
and treated and control data files were combined into a single dataset to make
the data easier to work with. Additionally, two extra columns 'dataset' and
'observation' were added to make it easier to track and work with individual
observations.

**Data from**

"Causal Effects in Non-Experimental Studies: Reevaluating the Evaluation of
Training Programs," Journal of the American Statistical Association, Vol. 94,
No. 448 (December 1999), pp. 1053-1062.

and

"Propensity Score Matching Methods for Non-Experimental Causal Studies," Review
of Economics and Statistics, Vol. 84, (February 2002), pp. 151-161.

The data are drawn from a paper by Robert Lalonde, "Evaluating the Econometric
Evaluations of Training Programs," American Economic Review, Vol. 76, pp.
604-620. We are grateful to him for allowing us to use this data, assistance in
reading his original data tapes, and permission to publish it here.

**NSW Data Files (Lalonde Sample)**

These files contain the treated and control units from the male sub-sample from
the National Supported Work Demonstration as used by Lalonde in his paper.

* nsw_treated (297 observations)
* nsw_control (425 observations)

The order of the variables from left to right match the numbered variable
in order in the below table. The last variable (re78) is the outcome and
all other variables are pre-treatment.

+----+-------------+---------+--------------------------------------------------+
| #  | Name        | Type    | Description                                      |
+====+=============+=========+==================================================+
| 01 | dataset     | int     | Data from 'nsw' study dataset used by Lalonde    |
+----+-------------+---------+--------------------------------------------------+
| 02 | observation | int     | Unique observation ID across treated/non-treated |
+----+-------------+---------+--------------------------------------------------+
| 03 | treatment   | int     | Binary treatment indicator where 1 if treated    |
+----+-------------+---------+--------------------------------------------------+
| 04 | age         | int     | Person's age in years                            |
+----+-------------+---------+--------------------------------------------------+
| 05 | education   | int     | Highest grade level person completed             |
+----+-------------+---------+--------------------------------------------------+
| 06 | black       | int     | Binary indicator where 1 if person is black      |
+----+-------------+---------+--------------------------------------------------+
| 07 | hispanic    | int     | Binary indicator where 1 if person is hispanic   |
+----+-------------+---------+--------------------------------------------------+
| 08 | married     | int     | Binary indicator where 1 if person is married    |
+----+-------------+---------+--------------------------------------------------+
| 09 | nodegree    | int     | Binary indicator where 1 if person has no degree |
+----+-------------+---------+--------------------------------------------------+
| 10 | re75        | int     | Person's earnings in dollars in 1975             |
+----+-------------+---------+--------------------------------------------------+
| 11 | re78        | int     | Person's earnings in dollars in 1978             |
+----+-------------+---------+--------------------------------------------------+
