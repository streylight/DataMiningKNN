# DataMiningKNN
K-nearest neighbor implementation for recommending related jobs

##1. Design##
For the design of this project I started out loading all the job records and parsing the
description and requirement before saving them in a hash map. I first removed all the
HTML for both fields using regex and then remove all the stop words using a stop words
list I found at http://www.cs.cmu.edu/~mccallum/bow/rainbow/. Once all the stop words
had been removed I create the description and requirement word bags.

After all the jobs and terms are loaded I create the weight matrix for each term for both
the description and requirements word bags. The matrix is partitioned like the table
below and each job holds the weight columns as vectors for each word bag term.
<br/><img src="http://i.imgur.com/zcMCdPJ.png"/><br/><br/>
Once the weight matrices have been calculated for both description and requirements I
then select k random centroids from the jobs list. After the initial centroids have been
established I start k-means by looping through each job and compare its cosine similarity
with each cluster centroid and assigning it to the one that it has the highest similarity to.

After each job is clustered I then recomputed the centroids based on the updated cluster.
To do this I calculate the tf-idf averages for each cluster by summing the weights for both
the description and requirements vectors and dividing them by vector length. I then store
the new centroids in a separate data structure from the clusters. Once the new centroids
have been evaluated I compare them to the centroids for the current clusters and if there
is no change then convergence has been achieved and I stop the k-means loop.
Finally I save the jobs list along with each jobs corresponding cluster to an output.tsv file.

##2. Output##
In order to determine the best value for k and for my intial centroid selection I ran my
program multiple times trying different values for k. In order to determine the
effectivness of my clusters I printing the SSE after each run to compare the output. The
project is currently using 10 as the value of k and the initial centroids are selected
at random. After running multiple passes with different random values for k I ended up
using 10 since it produced the highest SSE values. My SSE is measured by the highest
value being the better clusters because I am using cosine similarity as my “distance
measurement” and therefore the higher the similarity the better.
