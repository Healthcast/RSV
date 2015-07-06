## code skeleton for analysis pipeline
##
## ASSUME the rsv data is in the file ../data/rsv.csv
## I put the data in a separate directory so it is easy to
## keep it out of the repo. We have to be very strict about
## NEVER including the data; Git makes it very hard to remove
## something once it has been added.

## From the file ../data/rsv.csv,

## extract for each city,
##   each season which has recorded activity
##   Preliminary definitions:
##      a season is the period 1 Oct to 1 July, spanning the new year
##      "recorded activity" means at least 3 weeks with > 0 cases

## For each extracted season, flag the start week of the epidemic
##   Preliminary flag:
##      start week is the first week with >0 cases, for which the
##      next two weeks show more cases than the initial week.
##      In each of the following, the first positive integer marks
##      the start week: (0,0,1,2,3) (0,4,12,12)  (0,3,20,17)

## Make a function which plots each season as a line graph (x axis is time,
## y axis is number of cases). Add a mark at each week flagged as start week.
## Use this function to see if the flag definition makes sense.

## Create a test harnass for stat learning algos from  http://scikit-learn.org/

