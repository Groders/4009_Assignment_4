# 4009_Assignment_4

Simple Genetic algorithm find a shortest (not guaranteed to be the best) path between all nodes in a full connected graph. (traveling salesman problem) 

Q1: Create a sequential solution

Q2: Using MPI create a parallel solution

genM.py generates a 2D cost matrix with entries A[i][j] corresponding to the cost of going from i to j.

\_hostfile  specifies the ips with which you'll run the parallel code on. You can run locally you'll just see no speed ups. The parameters of the genetic algorithm need to be tuned for specific applications, as is it converge very quickly to an "ok" solution. Other configurations results in better results at a longer execution time.
