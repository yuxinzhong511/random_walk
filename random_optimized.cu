#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <map>
#include <fstream>
#include <utility>
#include <assert.h>
#include "benchmark.h"
#define MAX_EDGE 42


typedef NodeWeight<int64_t, float> WNode;
typedef EdgePair<NodeID, WNode> Edge;
typedef pvector<Edge> EdgeList;

static std::random_device rd;
static std::mt19937 rng(rd());

// Print datasets for debugging?
// CAUTION: This will print the entire training/testing datasets
//          Can fill up the terminal and slow down the program!
bool print_datasets = false;




/*
  Random number generator
*/
double RandomNumberGenerator() 
{
    static std::uniform_real_distribution<double> uid(0,1);//  
    return uid(rng);
}

/*
5 device_global_walk indexing

*/

void __global__ device_rwalk(
	int max_walk_length,
	int num_walks_per_node,
	int num_nodes, 
	unsigned long long rnumber, 
	int64_t * device_p_scan_list, 
  int64_t * device_v_list, 
  float * device_w_list, 
  int64_t *device_global_walk,
  //double* rand_device,
  //int *device_rand_int,
  //int w_n,
  int64_t * device_outdegree_list
  ){
		int64_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
		if(i >= num_nodes){
			return;
		}
    //int64_t valid_t_node[MAX_EDGE] = {0}; //++++++++++++++++++++++++

		long long int w;
	    for(int w_n = 0; w_n < num_walks_per_node; ++w_n) {
			//device_global_walk[( num_nodes * w_n * max_walk_length) + ( i * max_walk_length ) + 0] = i;
			device_global_walk[( i * max_walk_length * num_walks_per_node ) + ( w_n * max_walk_length ) + 0] = i;
			float prev_time_stamp = 0;
			int64_t src_node = i;
			int walk_cnt;
			for(walk_cnt = 1; walk_cnt < max_walk_length; ++walk_cnt) {
			  int valid_neighbor_cnt = 0;
			  for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){
          w = device_p_scan_list[src_node] + idx;
			  	if(device_w_list[w] > prev_time_stamp){
          //valid_t_node[valid_neighbor_cnt] = w;   //+++++++++++++++
				  valid_neighbor_cnt++; 
				  break; //++++++++++++++++++++++++++++++
				  }
			  }
			  if(valid_neighbor_cnt == 0) {
				break;
			  }
			  float min_bound = device_w_list[device_p_scan_list[src_node]];
			  float max_bound = device_w_list[device_p_scan_list[src_node]];
			  for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){
        w = device_p_scan_list[src_node] + idx;
				if(device_w_list[w] < min_bound)
				  min_bound = device_w_list[w];
				if(device_w_list[w] > max_bound)
				  max_bound = device_w_list[w];
			  }
			  float time_boundary_diff = (max_bound - min_bound);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			  if(time_boundary_diff < 0.0000001){
				for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){
					w = device_p_scan_list[src_node] + idx;
          if(device_w_list[w] > prev_time_stamp){
						//device_global_walk[( num_nodes * w_n * max_walk_length) + ( i * max_walk_length ) + walk_cnt] = device_v_list[w];
            device_global_walk[( i * max_walk_length * num_walks_per_node ) + ( w_n * max_walk_length ) + walk_cnt] = device_v_list[w];
						src_node = device_v_list[w];
						prev_time_stamp = device_w_list[w];
						break;
					}
				}

        //_______________________________________________________________________ added
        // int64_t rand_int = device_rand_int[( num_nodes * w_n * max_walk_length) + ( i * max_walk_length ) + walk_cnt];
        // //device_global_walk[( num_nodes * w_n * max_walk_length) + ( i * max_walk_length ) + walk_cnt] = device_v_list[valid_t_node[rand_int % valid_neighbor_cnt]];
        // device_global_walk[( i * max_walk_length * num_walks_per_node ) + ( w_n * max_walk_length ) + walk_cnt] = device_v_list[valid_t_node[rand_int % valid_neighbor_cnt]];
        // src_node = device_v_list[valid_t_node[rand_int % valid_neighbor_cnt]];
        // prev_time_stamp = device_w_list[valid_t_node[rand_int % valid_neighbor_cnt]];
        //__________________________________________________________________________________
				continue; 
			  }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////			  
			  double exp_sum = 0;            
			  for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){
          w = device_p_scan_list[src_node] + idx;
				  if(device_w_list[w] > prev_time_stamp){
				    exp_sum += exp((float)(device_w_list[w]-prev_time_stamp)/time_boundary_diff);
			  	}
			  }

			  double curCDF = 0, nextCDF = 0;
        //double random_number = rand_device[( i * (max_walk_length-1) * num_walks_per_node ) + ( w_n * (max_walk_length-1) ) + walk_cnt];
			  double random_number = rnumber * 1.0 / ULLONG_MAX;
        rnumber = rnumber * (unsigned long long)25214903917 + 11;   
			  bool fall_through = false;
			  for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){
          w = device_p_scan_list[src_node] + idx;
				if(device_w_list[w] > prev_time_stamp){
					nextCDF += (exp((float)(device_w_list[w]-prev_time_stamp)/time_boundary_diff) * 1.0 / exp_sum);
					if(nextCDF >= random_number && curCDF <= random_number) {
					  //device_global_walk[( num_nodes * w_n * max_walk_length) + ( i * max_walk_length ) + walk_cnt] = device_v_list[w];
					  device_global_walk[( i * max_walk_length * num_walks_per_node ) + ( w_n * max_walk_length ) + walk_cnt] = device_v_list[w];
					  src_node = device_v_list[w];
					  prev_time_stamp = device_w_list[w];
					  fall_through = true;
					  break;
				  } else {
					  curCDF = nextCDF;
				  }
				}
			  }
			  if(!fall_through){
				for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){ 
				   w = device_p_scan_list[src_node] + idx;
          if(device_w_list[w] > prev_time_stamp){
					//device_global_walk[( num_nodes * w_n * max_walk_length) + ( i * max_walk_length ) + walk_cnt] = device_v_list[w];
					device_global_walk[( i * max_walk_length * num_walks_per_node ) + ( w_n * max_walk_length ) + walk_cnt] = device_v_list[w];
					src_node = device_v_list[w];
					prev_time_stamp = device_w_list[w];
					break; 
				  }
				}
			  }
			}
			if (walk_cnt != max_walk_length){	
			  //device_global_walk[( num_nodes * w_n * max_walk_length) + ( i * max_walk_length ) + walk_cnt] = -1;
        device_global_walk[( i * max_walk_length * num_walks_per_node ) + ( w_n * max_walk_length ) + walk_cnt] = -1;
			}
		
		}
	}

 


#define cudaCheck(err) { \
	if (err != cudaSuccess) { \
		printf("CUDA error: %s: %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		assert(err == cudaSuccess); \
	} \
}

void WriteWalkToAFile(
  NodeID* global_walk, 
  int num_nodes, 
  int max_walk,
  int num_walks_per_node,
  std::string walk_filename) 
{
  std::ofstream random_walk_file(walk_filename);
  for(int w_n = 0; w_n < num_walks_per_node; ++w_n) {
    for(NodeID iter = 0; iter < num_nodes; iter++) {
      NodeID *local_walk = 
        global_walk + 
        ( iter * max_walk * num_walks_per_node ) +
        ( w_n * max_walk );
      // NodeID *local_walk = 
      // global_walk + 
      // ( w_n * max_walk * num_walks_per_node ) +
      // (  iter* max_walk );
      for (int i = 0; i < max_walk; i++) {
          if (local_walk[i] == -1)
            break;
          random_walk_file << local_walk[i] << " ";
      }
      random_walk_file << "\n";
    }
  }
  random_walk_file.close();
}


void compute_random_walk_main(
  WGraph &g, 
  int max_walk_length,
  int num_walks_per_node,
  std::string walk_filename) {
  max_walk_length++;
  NodeID *global_walk_device;

  int64_t * p_scan_list = (int64_t *)malloc (sizeof(int64_t) * (g.num_nodes()+1));
  int64_t * outdegree_list=(int64_t *)malloc (sizeof(int64_t) * (g.num_nodes()));
  int64_t * v_list = (int64_t *)malloc (sizeof(int64_t) * g.num_edges());
  float   * w_list = (float*)malloc(sizeof(float)*g.num_edges());
  int64_t * device_outdegree_list;
  int64_t * device_p_scan_list ;
  int64_t * device_v_list;
  float   * device_w_list;
 
  p_scan_list[0] = 0;
 
  for(NodeID i = 0; i < g.num_nodes(); ++i) {
    p_scan_list[i + 1] = p_scan_list[i] + g.out_degree(i);
    outdegree_list[i] = g.out_degree(i) + 0;
    int cnt = 0;
    for(auto v: g.out_neigh(i)){
      int64_t idx = p_scan_list[i] + cnt;
      v_list[idx] = v.v;
      w_list[idx] = v.w;
      cnt++;
    }
  }
  //  for(NodeID i = 0; i < g.num_nodes(); ++i) {
  //    std::cout << p_scan_list[i] << " " << outdegree_list[i] <<std::endl ;
  //  }
  cudaCheck(cudaMalloc((void**)&device_outdegree_list,sizeof(int64_t) * g.num_nodes()));
  cudaCheck(cudaMalloc((void**)&device_v_list,sizeof(int64_t) * g.num_edges()));
  cudaCheck(cudaMalloc((void**)&device_w_list,sizeof(float) * g.num_edges()));
  cudaCheck(cudaMalloc((void**)&device_p_scan_list,sizeof(int64_t) * (g.num_nodes()+1)));
  cudaCheck(cudaMemcpy(device_outdegree_list,outdegree_list,sizeof(int64_t) * g.num_nodes(),cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(device_v_list,v_list,sizeof(int64_t) * g.num_edges(),cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(device_w_list,w_list,sizeof(float) * g.num_edges(),cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(device_p_scan_list,p_scan_list,sizeof(int64_t) * (g.num_nodes()+1),cudaMemcpyHostToDevice));

  


  std::cout << "Computing random walk for " << g.num_nodes() << " nodes and " 
      << g.num_edges() << " edges." << std::endl;
  NodeID *global_walk = new NodeID[g.num_nodes() * max_walk_length * num_walks_per_node];
  //         
  Timer t;
  t.Start();
  cudaMalloc((void **)&global_walk_device, sizeof(int64_t)* g.num_nodes() * max_walk_length * num_walks_per_node);

  dim3 blocksize(64,1,1);
  dim3 gridsize(ceil((float)g.num_nodes()/blocksize.x),1,1);
  
  
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++rand number
  // double* rand_double = (double *)malloc(sizeof(double)*num_walks_per_node*(max_walk_length-1)* g.num_nodes());
  // for (int i=0; i < (max_walk_length-1)*num_walks_per_node* g.num_nodes(); i++ ){
  //     rand_double[i] =  RandomNumberGenerator();
  // }
  // double * rand_device;
  // cudaCheck(cudaMalloc((void **)&rand_device,sizeof(double)*(max_walk_length-1)*num_walks_per_node* g.num_nodes()));
  // cudaCheck(cudaMemcpy(rand_device,rand_double,sizeof(double)*num_walks_per_node*(max_walk_length-1)* g.num_nodes(),cudaMemcpyHostToDevice));
  
  
  // int* rand_int = (int *)malloc(sizeof(double)*num_walks_per_node*(max_walk_length-1)* g.num_nodes());
  // for(int i = 0; i < (max_walk_length-1)*num_walks_per_node* g.num_nodes(); i++ ) {
  //    rand_int [i] = rand(); 
  // }

  // int* device_rand_int;
  // cudaCheck(cudaMalloc((void **)&device_rand_int,sizeof(int)*(max_walk_length-1)*num_walks_per_node* g.num_nodes()));
  // cudaCheck(cudaMemcpy (device_rand_int,rand_int,sizeof(int)*num_walks_per_node*(max_walk_length-1)* g.num_nodes(),cudaMemcpyHostToDevice));
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++rand number
//for(int w_n = 0; w_n < num_walks_per_node; ++w_n) {
  for(int i = 0; i < num_walks_per_node / 10; i++){
  device_rwalk<<<gridsize,blocksize>>>(
  max_walk_length,
  num_walks_per_node,
  g.num_nodes(),
  (unsigned long long) (RandomNumberGenerator() * 1.0 * ULLONG_MAX),
  device_p_scan_list,
  device_v_list,
  device_w_list,
  global_walk_device,
  //rand_device,
  //device_rand_int,
  //w_n,
  device_outdegree_list
  );
  cudaDeviceSynchronize();
 }

//}
  // cudaFree(rand_device);
  // free(rand_double);
  // cudaFree(device_rand_int);
  // free(rand_int);
 
  cudaCheck(cudaMemcpy(global_walk, global_walk_device, sizeof(int64_t)*g.num_nodes() * max_walk_length * num_walks_per_node, cudaMemcpyDeviceToHost));
  cudaFree(global_walk_device);
  t.Stop();
  
  PrintStep("[TimingStat] Random walk time (s):", t.Seconds());
  WriteWalkToAFile(global_walk, g.num_nodes(), 
  max_walk_length, num_walks_per_node, walk_filename);
  delete[] global_walk;
  delete[] p_scan_list;
  delete[] w_list;
  delete[] v_list;
}

int main(int argc, char **argv) {
  CLApp cli(argc, argv, "link-prediction");

  if (!cli.ParseArgs())
    return -1;

  // Data structures
  WeightedBuilder b(cli);
  EdgeList el;
  WGraph g = b.MakeGraph(&el);

  // Read parameter configuration file
  cli.read_params_file();

  // Parameter initialization
  int   max_walk_length     =   cli.get_max_walk_length();
  int   num_walks_per_node  =   cli.get_num_walks_per_node();



  // Compute temporal random walk
  for(int i=0; i<20; ++i) {

    compute_random_walk_main(
      /* temporal graph */ g, 
      /* max random walk length */ max_walk_length,
      /* number of rwalks/node */ num_walks_per_node,
      /* filename of random walk */ "out_random_walk_main.txt"
    );
  }
  

  return 0;
}
