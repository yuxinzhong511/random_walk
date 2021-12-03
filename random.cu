//#include <wb.h>
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
//#include "thrust/device_vector.h"
//#include "thrust/host_vector.h"
#include "LocalVector.h"
//#include <curand_kernel.h>
//#include <curand.h>

typedef NodeWeight<int64_t, float> WNode;
typedef EdgePair<NodeID, WNode> Edge;
typedef pvector<Edge> EdgeList;

//typedef std::pair<NodeID, WeightT> TNode;
typedef std::vector<double> DoubleVector;
typedef LocalVector<double> device_DoubleVector;

static std::random_device rd;
static std::mt19937 rng(rd());

// Print datasets for debugging?
// CAUTION: This will print the entire training/testing datasets
//          Can fill up the terminal and slow down the program!
bool print_datasets = false;

typedef struct {
    NodeID id;
    WeightT weight;
    //float* elements;
} TNode;
typedef std::vector<TNode> TempNodeVector;
typedef LocalVector<TNode> device_TempNodeVector;
//#include "rwalk.h"



#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__device__  device_TempNodeVector FilterEdgesPostTime(
  //const WGraph &g, 
  NodeID src_node, 
  WeightT src_time,
  int64_t * device_outdegree_list,
  int64_t * device_v_list,
  float * device_w_list,
  int64_t *device_p_scan_list) 
{
  device_TempNodeVector filtered_edges;
  for(int i = 0; i< device_outdegree_list[src_node]; i++) {//auto v : g.out_neigh(src_node)
    
    int64_t idx= device_p_scan_list[src_node] + i;
    if(device_w_list[idx] > src_time){
        TNode tnode;
        tnode.id = device_v_list[idx];
        tnode.weight = device_w_list[idx];
        filtered_edges.add(tnode);
    }

  }
  // Sort edges by timestamp?
  //std::sort(filtered_edges.begin(), filtered_edges.end(), SortByTime);
  return filtered_edges;
}

/*
  Random number generator
*/
double RandomNumberGenerator() 
{
    static std::uniform_real_distribution<double> uid(0,1);//  
    return uid(rng);
}

__device__ int FindNeighborIdx(device_DoubleVector prob_dist,
double *rand_device, 
int * rand_int_device,
int tid,
int walk_cnt) 
{
  /* 
  Algorithm: get a random number between 0 and 1
  calculate the CDF of probTimestamps and compare
  against random number
  */
  double curCDF = 0, nextCDF = 0;
  int cnt = 0;
  double random_number = rand_device[tid*5+walk_cnt-1];
  for(auto it : prob_dist) {
      nextCDF += it;
      if(nextCDF >= random_number && curCDF <= random_number) {
          return cnt;
      }  else {
          curCDF = nextCDF;
          cnt++;
    }
  }
  // Ideally, it should never hit this point
  // The only time when it hits this is when the timestamps of all
  //     outgoing edges are same, in which case, just randomly
  //     selecting one edge out of all (as done below) should work.
  return rand_int_device[tid*4 + walk_cnt] % prob_dist.getSize();
}

__device__ float TimeBoundsDelta(int64_t node_id, 
    int64_t * device_outdegree_list,
    int64_t * device_v_list,
    float* device_w_list,
    int64_t * device_p_scan_list)  {
    // PrintNeighbors(node_id);
    float min_bound = 0, max_bound = 0;
   // int cnt = 0;
    for(int i=0 ; i< device_outdegree_list[node_id];i++ ) {
      int64_t idx = device_p_scan_list[node_id] + i;
      if(i == 0)
        min_bound = max_bound = device_w_list[idx];
      if(device_w_list[idx] < min_bound)
        min_bound =device_w_list[idx];
      if(device_w_list[idx] > max_bound)
        max_bound = device_w_list[idx];
     // cnt++;
    }
    return (max_bound - min_bound);
}

// __device__ bool GetNeighborToWalk(
//   //const WGraph &g, 
//   int64_t src_node, 
//   float src_time,
//   TNode& next_neighbor,
//   double * rand_device,
//   int tid,
//   int cnt,
//   int* rand_int_device,
//   int64_t * device_outdegree_list,
//   int64_t * device_v_list,
//   float * device_w_list,
//   int64_t * device_p_scan_list
//   ) 
// {
//   int neighborhood_size = device_outdegree_list[src_node];//g.out_degree(src_node);
//   if(neighborhood_size == 0) {
//     return false;
//   } else {
//     device_TempNodeVector filtered_edges = FilterEdgesPostTime(
//     src_node, 
//     src_time, 
//     device_outdegree_list,
//     device_v_list,
//     device_w_list,
//     device_p_scan_list);
//     if(filtered_edges.empty()) {
//       return false;
//     }
//     if(filtered_edges.getSize() == 1) {
//       next_neighbor = filtered_edges[0];
//       next_neighbor.id = 0;
//       return true;
//     } else {
//       device_DoubleVector prob_dist;
//       WeightT time_boundary_diff;
//       time_boundary_diff =TimeBoundsDelta(src_node,
//       device_outdegree_list,
//       device_v_list,
//       device_w_list,
//       device_p_scan_list);
//       if(time_boundary_diff == 0)
//       {
//         next_neighbor = filtered_edges[rand_int_device[tid*5+cnt-1] % filtered_edges.getSize()];
//         next_neighbor.id =0;
//         return true;
//       } else {
//         // TODO: parallelism?
//         for(auto it : filtered_edges) {
//           prob_dist.add(exp((float)(it.weight-src_time)/time_boundary_diff));
//         }
//         double exp_sum = 0;
//         for(int i = 0;i<prob_dist.getSize();i++){
//           exp_sum += prob_dist[i];
//         }
//         for (uint32_t i = 0; i < prob_dist.getSize(); ++i) {
//           prob_dist[i] = prob_dist[i] / exp_sum;
//         }
//         int neighbor_index = FindNeighborIdx(prob_dist,rand_device,rand_int_device,tid,cnt);
//         next_neighbor = filtered_edges[neighbor_index];
//         next_neighbor.id = 0;
//         return true;
//       }
//     }
//   }
// }

// __device__ bool compute_walk_from_a_node(
//   //const WGraph& g, 
//   int64_t src_node,
//   float prev_time_stamp, 
//   TNode& next_neighbor_ret, 
//   int max_walk_length, 
//   int64_t *local_array, 
//   int32_t pos,
//   double *rand_device,
//   int tid,
//   int cnt,
//   int * rand_int_device,
//   int64_t * device_outdegree_list,
//   int64_t * device_v_list,
//   float * device_w_list,
//   int64_t * device_p_scan_list) 
// {
//   TNode next_neighbor ;
//   if(device_outdegree_list[src_node] != 0 && 
//     GetNeighborToWalk(
//     //g, 
//     src_node, 
//     prev_time_stamp,
//     next_neighbor,
//     rand_device,
//     tid,
//     cnt,
//     rand_int_device,
//     device_outdegree_list,
//     device_v_list,
//     device_w_list,
//     device_p_scan_list)) {
//     // local_array[pos] = next_neighbor.id;
//     next_neighbor_ret = next_neighbor;
//     return true;
//   }
//   return false;
// }
void __global__ device_rwalk(
	int m_walk_length,
	int n_walks_per_node,
	int total_num_nodes, 
	// unsigned long long rnumber, 
	int64_t * d_p_scan_list, 
  int64_t * d_v_list, 
  float * d_w_list, 
  int64_t *d_global_walk,
  double* rand_device,
  int64_t * device_outdegree_list
  ){
		int64_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
		if(i >= total_num_nodes){
			return;
		}

		long long int w;
	    for(int w_n = 0; w_n < n_walks_per_node; ++w_n) {
			d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + 0] = i;
			//d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + 0] = i;
			float prev_time_stamp = 0;
			int64_t src_node = i;
			int walk_cnt;
			for(walk_cnt = 1; walk_cnt < m_walk_length; ++walk_cnt) {
			  int valid_neighbor_cnt = 0;
			  for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){
          w = d_p_scan_list[src_node] + idx;
         // __syncthreads();
				if(d_w_list[w] > prev_time_stamp){
				  valid_neighbor_cnt++;
				  break;
				}
			  }
			  if(valid_neighbor_cnt == 0) {
				break;
			  }
			  float min_bound = d_w_list[d_p_scan_list[src_node]];
			  float max_bound = d_w_list[d_p_scan_list[src_node]];
			  for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){
        w = d_p_scan_list[src_node] + idx;
         // __syncthreads();
				if(d_w_list[w] < min_bound)
				  min_bound = d_w_list[w];
				if(d_w_list[w] > max_bound)
				  max_bound = d_w_list[w];
			  }
			  float time_boundary_diff = (max_bound - min_bound);

			  if(time_boundary_diff < 0.0000001){
				for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){ // We randomly pick 1 neighbor, we just pick the first
					 w = d_p_scan_list[src_node] + idx;
         // __syncthreads();
          if(d_w_list[w] > prev_time_stamp){
						d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = d_v_list[w];
            //d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + walk_cnt] = d_v_list[w];
						src_node = d_v_list[w];
						prev_time_stamp = d_w_list[w];
						break;
					}
				}
				continue; 
			  }
			  
			  double exp_summ = 0;            
			  for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){
           w = d_p_scan_list[src_node] + idx;
         // __syncthreads();
				if(d_w_list[w] > prev_time_stamp){
				  exp_summ += exp((float)(d_w_list[w]-prev_time_stamp)/time_boundary_diff);
				}
			  }

			  double curCDF = 0, nextCDF = 0;
        double random_number = rand_device[( i * (m_walk_length-1) * n_walks_per_node ) + ( w_n * (m_walk_length-1) ) + walk_cnt];
			  //double random_number = rnumber * 1.0 / ULLONG_MAX;
        //rnumber = rnumber * (unsigned long long)25214903917 + 11;   
			  bool fall_through = false;
			  for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){
          w = d_p_scan_list[src_node] + idx;
				if(d_w_list[w] > prev_time_stamp){
					nextCDF += (exp((float)(d_w_list[w]-prev_time_stamp)/time_boundary_diff) * 1.0 / exp_summ);
					if(nextCDF >= random_number && curCDF <= random_number) {
					  d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = d_v_list[w];
					  //d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + walk_cnt] = d_v_list[w];
					  src_node = d_v_list[w];
					  prev_time_stamp = d_w_list[w];
					  fall_through = true;
					  break;
				  } else {
					  curCDF = nextCDF;
				  }
				}
			  }
			  if(!fall_through){
				for(int64_t idx=0; idx < device_outdegree_list[src_node]; idx++){ // This line should not be reached anyway (reaching this line means something is wrong). But just for testing, we randomly pick 1 neighbor, we just pick the first
				   w = d_p_scan_list[src_node] + idx;
          if(d_w_list[w] > prev_time_stamp){
					d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = d_v_list[w];
					//d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + walk_cnt] = d_v_list[w];
					src_node = d_v_list[w];
					prev_time_stamp = d_w_list[w];
					break; 
				  }
				}
			  }
			}
			if (walk_cnt != m_walk_length){	
			  //d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = -1;
        d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + walk_cnt] = -1;
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
      // random_walk_file << local_walk[0] << " ";
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
  
  
    
  // std::cout << "walk number: " << w_n << std::endl;
  //++++++++++++++++++++++++++++++++++++++++++++++++rand number
  double* rand_double = (double *)malloc(sizeof(double)*num_walks_per_node*(max_walk_length-1)* g.num_nodes());
  for (int i=0; i < (max_walk_length-1)*num_walks_per_node* g.num_nodes(); i++ ){
      rand_double[i] =  RandomNumberGenerator();
  }
 
  double * rand_device;
  
  cudaCheck(cudaMalloc((void **)&rand_device,sizeof(double)*(max_walk_length-1)*num_walks_per_node* g.num_nodes()));
  cudaCheck(cudaMemcpy(rand_device,rand_double,sizeof(double)*num_walks_per_node*(max_walk_length-1)* g.num_nodes(),cudaMemcpyHostToDevice));
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++rand number
  
  for(int i = 0; i < num_walks_per_node / 10; i++){
  device_rwalk<<<gridsize,blocksize>>>(
  max_walk_length,
  num_walks_per_node,
  g.num_nodes(),
 // (unsigned long long) (RandomNumberGenerator() * 1.0 * ULLONG_MAX),
  device_p_scan_list,
  device_v_list,
  device_w_list,
  global_walk_device,
  rand_device,
  device_outdegree_list
  );
  cudaDeviceSynchronize();
  cudaFree(rand_device);
  free(rand_double);
  }
  cudaCheck(cudaMemcpy(global_walk, global_walk_device, sizeof(int64_t)*g.num_nodes() * max_walk_length * num_walks_per_node, cudaMemcpyDeviceToHost));
  cudaFree(global_walk_device);
  t.Stop();
  
  PrintStep("[TimingStat] Random walk time (s):", t.Seconds());
  WriteWalkToAFile(global_walk, g.num_nodes(), 
    max_walk_length, num_walks_per_node, walk_filename);
  delete[] global_walk;
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
