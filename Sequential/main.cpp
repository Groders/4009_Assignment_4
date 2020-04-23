
# include <cstdlib>
# include "mpi.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>    // std::random_shuffle
#include <random>


using namespace std;


void print_2darray(int**matrix, int n);
int find_best_path(int**matrix, int** path, int M, int n);
void copyPath (int* path1, int* path2, int n);
void copyHalf (int* path1, int* path2, int* path3, int n, bool* selected);

//Helper function for debugging
void print_2darray(int** matrix, int n){
    for(int z = 0 ; z < n ; z++) {
        for(int y = 0 ; y < n ; y++){
			cout << matrix[z][y] << " ";
        }
        cout << endl;
    }
}

//Helper function for debugging
void print_2darray_1(int** matrix, int n, int k){
    for(int z = 0 ; z < n ; z++) {
        for(int y = 0 ; y < k ; y++){
			cout << matrix[z][y] << " ";
        }
        cout << endl;
    }
}

//Helper function for debugging
void print_array(int* arr, int n){
    for(int z = 0 ; z < n ; z++) {
		cout << arr[z] << " ";
    }
}

/**
 * matrix -> pointer to hold all the costs of going between each node
 * n -> size of matrix (n by n)
 * */
void gen_matrix (int** matrix, int n){
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			if (j < i){
				matrix[i][j] = matrix[j][i];
			}
			else {
				matrix[i][j] = rand() % 100 + 1;
			}
		}
	}
}

/*
 * matrix -> pointer to hold all the costs of going between each node
 * path -> one path going through each of the nodes
 * n -> size of matrix (n by n)
 * */
double fitness(int** matrix, int* path, int n){
	int cost = 0;
	for (int i = 0; i < n; i++){
		cost += matrix[path[i]][path[(i+1)%n]];
	}
	return  1.0 / cost;
}

/*
 * matrix -> pointer to hold all the costs of going between each node
 * path -> one path going through each of the nodes
 * n -> size of matrix (n by n)
 * */
double cost(int**matrix, int* path, int n){
	int cost = 0;
	for (int i = 0; i < n; i++){
		cost += matrix[path[i]][path[(i+1)%n]];
	}
	return cost;
}


/*
 * x -> number we are looking for
 * aarr -> array we are looking through
 * n -> size of array
 * returns index i such that arr[i-1] < a <= arr[i]
 * */
int BinSearch(double x, double* arr, int n){
    int l = 0;
    int r = n-1;
    int m = l+(r-l)/2;
    while(l <= r){
        if (arr[m] < x){
            l = m + 1;
        } 
        else if (arr[m] > x){
            r = m - 1;
        }else{
            break;
        }
        m = l+(r-l)/2;
    }
    return m;
}

/*
 * matrix -> n by n matrix containing costs between each node
 * path -> buffer containing all the old paths
 * newPaths -> buffer to be filled will all the new paths 
 * chosen -> number of elements chosen for newPath, will be <= S 
 * n -> size of matrix and size of a path 
 * S -> number of paths to add to newPaths
 * M -> number of total paths (size of paths and newPaths)
 * F -> buffer for holding all the fitness values of each path 
 * selected -> buffer for keeping track of if a path has been chosen
 * This function selects S paths from path biased by their wieghts, it'll give up after 2S tries to find an path to add
 * */
void select_pop(int**matrix, int** path, int** newPaths, int * chosen, int n, int S, int M, double* F, bool *selected){
	//select S from M
	for (int i = 0; i < M; i++){
		selected [i] = false;
	}
	int loops = 0; 
	double max;

	double best_fit = fitness(matrix, path[0], n);
	int best = 0;
	F[0] = fitness (matrix, path[0], n);
	for (int i = 1; i < M; i++){
		double fit = fitness(matrix, path[i], n);
		if (fit > best_fit){
			best_fit = fit;
			best = i;
		}
		F[i] = F[i-1] + fit;
	}
	
	selected[best] = true;
	copyPath(path[best], newPaths[0], n);
	*chosen = 1; 

	max = F[M-1];
	//random numbers
	std::random_device r;
	std::uniform_real_distribution<double> unif(0,max);
	std::default_random_engine re(r());	
	
	while (*chosen < S and loops < 2*S){
		double a_random_double = unif(re);
		int k = BinSearch (a_random_double, F,M); 
			if (selected[k] == false){
				copyPath(path[k], newPaths[*chosen], n);
				*chosen += 1;
			}
		loops += 1;
	}
}

/*
 * matrix -> n by n matrix containing costs between each node
 * newPaths -> buffer to be filled will all the new paths 
 * chosen -> number of elements chosen for newPath, will be <= S 
 * n -> size of matrix and size of a path 
 * M -> number of total paths (size of paths and newPaths)
 * selected -> buffer for keeping track of if a node has been selected (not used here)
 * */
void crossover(int**matrix, int** newPaths, int* chosen, int n, int M, bool* selected){
	while (*chosen < M){
		int x = rand() % *chosen;
		int y = rand() % (*chosen - 1);
		if (x == y){
			y = *chosen -1; 
		}

		copyHalf(newPaths[x], newPaths[y], newPaths[*chosen], n, selected);
		*chosen += 1;
		if (*chosen >= M){
			break;
		}
		copyHalf(newPaths[y], newPaths[x], newPaths[*chosen], n, selected);
	}
}

/*
 * path1 -> path1 to copy
 * path2 -> path2 to copy 
 * path3 -> destination path
 * n -> size of matrix and size of a path 
 * selected -> buffer for keeping track of if a node has been selected
 * */
 void copyHalf (int* path1, int* path2, int* path3, int n, bool* selected){
	int min = 0; 
	for (int i = 0; i < n; i++){
		selected[i] = false;
	}

	for (int i = 0; i < n/2; i++){
		path3[i] = path1[i];
		selected[path1[i]] = true;
	}
	
	for (int i = n/2 +1; i < n; i++){
		path3[i] = path2[i];
	}
	
	for (int i = n/2 +1; i < n; i++){
		if (selected[path3[i]] == true){
			for (int j = min; j < n; j++){
				if (selected[i] == false){
					path3[i] = j;
					selected[j] = true;
					min = j;
				}
			}
		}
	}
}

/*
 * path -> path to mutate
 * n -> size of a path 
 * Z -> number of mutations per path
 * */
void mutate(int* path, int n, int Z){
	for (int i = 0; i < Z; i++){
		int x = rand() % n;
		int y = rand() % n;
		int temp = path[x];
		path[x] = path[y];
		path[y] = temp;
	}
}


/*
 * newPaths -> list of paths to mutate
 * n -> size of a path 
 * M -> number of paths
 * U -> number of paths to mutate (roughly)
 * Z -> number of mutations per path
 * */
void mutate_list(int** newPaths, int n, int M, int U, int Z){
	for (int i = 0; i < M; i++){
		if ((rand() % M + 1) < U){
			mutate(newPaths[i], n, Z);
		}
	}
}

/*
 * path1 -> src path
 * path1 -> dest path
 * n -> size of a path 
 * */
void copyPath (int* path1, int* path2, int n){
	for (int i = 0; i < n; i++){
		path2[i] = path1[i];
	}
}

/*
 * path1 -> first path to compare
 * path1 -> second path to compare
 * n -> size of a path
 * returns true if they are the same 
 * */
bool compare_path (int* path1, int* path2, int n){
	for (int i = 0; i < n; i++){
		if (path1[i] != path2[i]) return false;
	}
	return true;
}

/*
 * matrix -> matrix of costs between nodes
 * path -> list of paths
 * M -> number of paths
 * n -> size of a path
 * returns index of the best path 
 * */
 int find_best_path(int**matrix, int** path, int M, int n){
	double best_fit = fitness(matrix, path[0], n);
	int best = 0;
	for (int i = 1; i < M; i++){
		double fit = fitness(matrix, path[i], n);
		if (fit > best_fit){
			best_fit = fit;
			best = i;
		}
	}
	return best; 
}

/*
 * path -> list of paths
 * M -> number of paths
 * dim -> size of a path
 * */
void init_paths (int **paths, int M, int dim){
	for (int i = 0; i < M; i++){
		for (int j = 0; j < dim; j++){
			paths[i][j] = j;
		}
		std::random_shuffle (paths[i], &paths[i][dim]);
	}
}

void init_distance_matrix(int **D, int N, string inputName){
  ifstream input(inputName);
  if (input.is_open()){;

    for(int i = 0 ; i< N ; i++){
		for (int j = 0; j <N; j++){
			input>>D[i][j];
		}
    }

  }

}

int main(int argc, char *argv[])
{
	int id;
	int p;
	double wtime;
  	int n, k, m;
	bool *send_data, *gather_data;

	MPI::Init(argc, argv); //  Initialize MPI.
	p = MPI::COMM_WORLD.Get_size(); //  Get the number of processes.
	id = MPI::COMM_WORLD.Get_rank(); //  Get the individual process ID.
	
	srand (time(NULL));

	int dim = 100;
	int **matrix = new int*[dim];
	for(int i =0; i<dim; i++){
		matrix[i] = new int[dim];
	}

	init_distance_matrix(matrix, dim, "distance_matrix.txt");
	
	//print_2darray(matrix, dim);
	//send data to everyone :)
	//this is seq tho so we don't need to
	
	
	
	int M = 100000; //num in each pop
	int K = 1000; //max num round
	int R = 10; //exit when best is same for R rounds
	int S = 75000; // pop to keep
	int U = 10000; // pop to mutate
	int Z = dim/20; // pop to permute 
	
	int ** paths = new int* [M];
	int ** newPaths = new int* [M];
	int ** tempPaths;
	for (int i = 0; i < M; i++){
		paths[i] = new int [dim];
		newPaths[i] = new int [dim];
	}
	init_paths(paths, M, dim);
	
	int same = 0;
	int chosen = 0;
	int *bestPath = new int[dim];
	double bestFit = 0;
	int curr_best;
	double *F = new double [M];
	bool *selected = new bool[M];

	wtime = MPI::Wtime();
	
	curr_best = find_best_path(matrix, paths, M, dim);
	bestFit = fitness(matrix, paths[curr_best], dim);
	copyPath(paths[curr_best],bestPath, dim);
	for (int cycles = 0; cycles < K && same < R; cycles++){
		select_pop(matrix, paths, newPaths, &chosen, dim, S, M, F, selected);
		crossover(matrix, newPaths, &chosen, dim, M, selected);
		mutate_list(newPaths, dim, M, U, Z);
		
		curr_best = find_best_path(matrix, newPaths, M, dim);
		
		if (bestFit == 0){
			bestFit = fitness(matrix, newPaths[curr_best], dim);
			copyPath(newPaths[curr_best],bestPath, dim);
		}
		else if (compare_path(bestPath, newPaths[curr_best], dim) == true){
			same += 1;
		} else if (fitness(matrix, newPaths[curr_best], dim) > bestFit){
			bestFit = fitness(matrix, newPaths[curr_best], dim);
			copyPath(newPaths[curr_best],bestPath, dim);
			same = 0;
		} else {
			same += 1;
		}
		
		//print_2darray_1(paths, M, dim);
		//cout <<"TEST" << endl;
		//print_2darray_1(newPaths, M, dim);

		tempPaths = newPaths;
		newPaths = paths;
		paths = tempPaths;


	}
	print_array(bestPath, dim);
	cout << bestFit << " " << cost(matrix, bestPath, dim) << endl;

	if (id == 0)
	{	
		wtime = MPI::Wtime() - wtime;
		cout << "Elapsed wall clock time = " << wtime << " seconds.\n";
	}
	
	
//  Terminate MPI.
	MPI::Finalize();
	return 0;
}

