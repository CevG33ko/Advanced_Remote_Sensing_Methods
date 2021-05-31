# Input data

A 3x3 gray value image is given. The pixels are numbered row by row, starting by 1 and ending with 9.
Together with the image the formular to calculate the weights is given.

w_ij = w_ji = e^(-d_ij^2/sigma^2)

With d as gray value between the adjacent pixels and sigma 14,72 which is in the range of 10-20% of the mean feature differences.

For the image three segments are defined.

Segment 1 = [1, 2, 4, 5]
Segment 2 = [7, 8, 9]
Segment 3 = [3, 6]
 
The neighborhoods are defined along the rows and columns. That means, that the first pixel has only two neighbors  - 2 and 4.

Together with this image a jupyter notebook with a python script inside is provided with the materials.

# Methods

The given image should be segmented by using a graphbased method, more precise the normalized cut. But to use graph based methods the input data has to be transferred into a graph representation which is explained in the next section.

## Defining the graph

A graph is defined by n nodes and edges where each edge defines the connections between two nodes.
Every edge has a weight that describes how strong the connection is.
These connections are stored as entries in a so called adjacent matrix, which has a size of n x n and is initialized with all fields set to 0.
Then every edge is represented with a one in the corresponding field.
In case of an undirected graph the resulting matrix is mirrored along the main diagonal.

The weight is calculated for every neighorhood and the result is saved in a seperate matrix, that is called the weight matrix w.
In this matrix the diagonal is set to ones.

A third matrix is the degree matrix which has only the sum of all weights for the corresponding node on the diagonal.

## ncut

The normalized cut works by using the sum of weights between clusters and normalizing these with the sum of all weights in edges which are connected to the cluster.

The function to calculate the weight between two clusters is called CUT(A, B), where A and B are the set of nodes in one cluster.
The second function used in NCUT is called ASSOC(A, V) or Volume with A the cluster for which the ASSOC should be calculated and V the set of all nodes.
In this function all weights from every edge connected to a node in the cluster are summed up.

These two functions are used to normalize the cut by the volume of the cluster.

This results in the following formula:

Ncut(A, B) = cut(A,B)/assoc(A,V) + cut(A,B)/assoc(B,V)

The Ncut with the minimal result is the best result.

But the ncut can also be approximated by solving the eigenvalue problem.

Ncut(A,B) = (y'(D-W)y)/y'Dy with y as the second smallest eigenvector from (D-W)

Which can also be written as (D-W)y = lamdaDy

The second smallest eigenvector is used because the smallest eigenvector first eigenvector is always 0.
This is because it represents the minimal ncut(A,V)=0 result.
From this eigenvector two segments can be obtained where segment one contain all nodes represented by negative values and segment two contain all other nodes.

The whole algorithm goes as:

 1. Set up the weight, degree matrix
 2. Solve the eigenvalue problem and set y to the second smallest eigenvector
 3. Split the nodes into two segments
 4. Recursively do this for each of the two segments until a minimum amount of nodes for one segment is reached or the ncut goes over the threshold.

# Results

The grey value image transfered into a graph by applying the given data, rules and formula for weight.
These are the weight matrix and the degree matrix.

Degree Matrix:
2.99539549868157	0	0	0	0	0	0	0	0
0	2.99772772542747	0	0	0	0	0	0	0
0	0	2.01601891214543	0	0	0	0	0	0
0	0	0	3.73965268746309	0	0	0	0	0
0	0	0	0	3.74819066953996	0	0	0	0
0	0	0	0	0	2.24640519193442	0	0	0
0	0	0	0	0	0	2.73965268746309	0	0
0	0	0	0	0	0	0	3.72136150074513	0
0	0	0	0	0	0	0	0	2.20588933774006

Weight Matrix:
1	0.995395498681570	0	1	0	0	0	0	0
0.995395498681570	1	0.0206234134638635	0	0.981708813282036	0	0	0	0
0	0.0206234134638635	1	0	0	0.995395498681570	0	0	0
1	0	0	1	0.995395498681570	0	0.744257188781523	0	0
0	0.981708813282036	0	0.995395498681570	1	0.0268291687948316	0	0.744257188781523	0
0	0	0.995395498681570	0	0.0268291687948316	1	0	0	0.224180524458019
0	0	0	0.744257188781523	0	0	1	0.995395498681570	0
0	0	0	0	0.744257188781523	0	0.995395498681570	1	0.981708813282036
0	0	0	0	0	0.224180524458019	0	0.981708813282036	1

Using the given segmentation with the three segments the different Ncuts are:

ncut1 = ncut([1, 2, 4, 5], [3, 6]) = 0.0147
ncut2 = ncut([1, 2, 4, 5], [7, 8, 9]) = 0.2822
ncut3 = ncut([3, 6], [7, 8, 9]) = 0.0785

[Calculated](Calculated.md) the segmentation using the eigenvalue approach results in a ncut value of 0.0586 and a segmentation of

    0 0 1 0 0 1 0 0 1
    
with an eigenvector of

    0.1277 0.1185 -0.4527 0.1145 0.1000 -0.4108 0.0823 0.0450 -0.0445

    
# Discussion

Looking at the results of the three ncut calculations ncut2 is much higher than the others and the ncut1 is the lowest.
That shows that the two segments in ncut1 are far away from each other and the ncut2 segments are much closer.

Not using the given segments but solving the eigenvalue problem gives an other segmentation, where pixel 9 is in the same segment as 3 and 6.
But the value in the eigenvector is really close to the threshold of 0.
This confirms that the ncut1 has the lowest value.

# Problem 3




