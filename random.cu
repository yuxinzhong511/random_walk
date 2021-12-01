#include <wb.h>
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
#include "benchmark.h"
//#include "thrust/device_vector.h"
//#include "thrust/host_vector.h"
#include "LocalVector.h"
//#include <curand_kernel.h>
//#include <curand.h>

typedef NodeWeight<NodeID, WeightT> WNode;
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
  double random_number = rand_device[tid*4+walk_cnt-1];
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

__device__ bool GetNeighborToWalk(
  //const WGraph &g, 
  NodeID src_node, 
  WeightT src_time,
  TNode& next_neighbor,
  double * rand_device,
  int tid,
  int cnt,
  int* rand_int_device,
  int64_t * device_outdegree_list,
  int64_t * device_v_list,
  float * device_w_list,
  int64_t * device_p_scan_list
  ) 
{
  int neighborhood_size = device_outdegree_list[src_node];//g.out_degree(src_node);
  if(neighborhood_size == 0) {
    return false;
  } else {
    device_TempNodeVector filtered_edges = FilterEdgesPostTime(src_node, 
    src_time, 
    device_outdegree_list,
    device_v_list,
    device_w_list,
    device_p_scan_list);
    if(filtered_edges.empty()) {
      return false;
    }
    if(filtered_edges.getSize() == 1) {
      next_neighbor = filtered_edges[0];
      return true;
    } else {
      device_DoubleVector prob_dist;
      WeightT time_boundary_diff;
      time_boundary_diff =TimeBoundsDelta(src_node,
      device_outdegree_list,
      device_v_list,
      device_w_list,
      device_p_scan_list);
      if(time_boundary_diff == 0)
      {
        next_neighbor = filtered_edges[rand_int_device[tid*4+cnt-1] % filtered_edges.getSize()];
        return true;
      } else {
        // TODO: parallelism?
        for(auto it : filtered_edges) {
          prob_dist.add(exp((float)(it.weight-src_time)/time_boundary_diff));
        }
        double exp_sum = 0;
        for(int i = 0;i<prob_dist.getSize();i++){
          exp_sum += prob_dist[i];
        }
        for (uint32_t i = 0; i < prob_dist.getSize(); ++i) {
          prob_dist[i] = prob_dist[i] / exp_sum;
        }
        int neighbor_index = FindNeighborIdx(prob_dist,rand_device,rand_int_device,tid,cnt);
        next_neighbor = filtered_edges[neighbor_index];
        return true;
      }
    }
  }
}

__device__ bool compute_walk_from_a_node(
  //const WGraph& g, 
  NodeID src_node,
  WeightT prev_time_stamp, 
  TNode& next_neighbor_ret, 
  int max_walk_length, 
  NodeID *local_array, 
  int32_t pos,
  double *rand_device,
  int tid,
  int cnt,
  int * rand_int_device,
  int64_t * device_outdegree_list,
  int64_t * device_v_list,
  float * device_w_list,
  int64_t * device_p_scan_list) 
{
  TNode next_neighbor;
  if(device_outdegree_list[src_node] != 0 && 
    GetNeighborToWalk(
    //g, 
    src_node, 
    prev_time_stamp,
    next_neighbor,
    rand_device,
    tid,
    cnt,
    rand_int_device,
    device_outdegree_list,
    device_v_list,
    device_w_list,
    device_p_scan_list)) {
    local_array[pos] = next_neighbor.id;
    next_neighbor_ret = next_neighbor;
    return true;
  }
  return false;
}


__global__ void random_per_node(
  int w_n, 
  int num_nodes,
  int max_walk_length,
  int num_walks_per_node ,
  NodeID *global_walk,  
  double * rand_device, 
  int * rand_int_device,
  int64_t * device_outdegree_list,
  int64_t *device_v_list,
  float *device_w_list,
  int64_t *  device_p_scan_list
  ) {
  //@@ 
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  NodeID *local_walk = 
        global_walk + 
        ( tid * max_walk_length * num_walks_per_node ) +
        ( w_n * max_walk_length );
      local_walk[0] = tid;
      WeightT prev_time_stamp = 0;
      NodeID next_neighbor = tid;
      TNode next_neighbor_ret;
      int walk_cnt;
      for(walk_cnt = 1; walk_cnt < max_walk_length; ++walk_cnt) {
        bool cont = compute_walk_from_a_node(
          //g_device, 
          next_neighbor, 
          prev_time_stamp, 
          next_neighbor_ret, 
          max_walk_length, 
          local_walk, 
          walk_cnt,
          rand_device,
          tid,
          walk_cnt,
          rand_int_device,
          device_outdegree_list,
          device_v_list,
          device_w_list,
          device_p_scan_list
        );
        if(!cont) break;
        next_neighbor = next_neighbor_ret.id;
        prev_time_stamp = next_neighbor_ret.weight;
      }
      if (walk_cnt != max_walk_length)
          local_walk[walk_cnt] = -1;
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

  NodeID *global_walk_device;
  
//   cudaMemcpyToSymbol(g_device,g,sizeof(WGraph));

//   WGraph* g_device;
  
  // p_scan_list = new int64_t [num_of_nodes + 1];
  int64_t * p_scan_list = (int64_t *)malloc (sizeof(int64_t) * (g.num_nodes()+1));
  int64_t * outdegree_list=(int64_t *)malloc (sizeof(int64_t) * (g.num_nodes()));
  int64_t * v_list = (int64_t *)malloc (sizeof(int64_t) * g.num_edges());
  float   * w_list = (float*)malloc(sizeof(float)*g.num_edges());
  int64_t * device_outdegree_list;
  int64_t * device_p_scan_list ;
  int64_t * device_v_list;
  float   * device_w_list;
  p_scan_list[0] = 0;
  // v_list = new int64_t[num_of_edges];
  // w_list = new float[num_of_edges];

  for(NodeID i = 0; i < g.num_nodes(); ++i) {
    p_scan_list[i + 1] = p_scan_list[i] + g.out_degree(i);
    int cnt = 0;
    for(auto v: g.out_neigh(i)){
      int64_t idx = p_scan_list[i] + cnt;
      v_list[idx] = v.v;
      w_list[idx] = v.w;
      cnt++;
    }
  }

  cudaMalloc((void**)&device_outdegree_list,sizeof(int64_t) * g.num_nodes());
  cudaMalloc((void**)&device_v_list,sizeof(int64_t) * g.num_edges());
  cudaMalloc((void**)&device_w_list,sizeof(float) * g.num_edges());
  cudaMalloc((void**)&device_p_scan_list,sizeof(int64_t) * (g.num_nodes()+1));
  cudaMemcpy(device_outdegree_list,outdegree_list,sizeof(int64_t) * g.num_nodes(),cudaMemcpyHostToDevice);
  cudaMemcpy(device_v_list,v_list,sizeof(int64_t) * g.num_nodes(),cudaMemcpyHostToDevice);
  cudaMemcpy(device_w_list,w_list,sizeof(float) * g.num_nodes(),cudaMemcpyHostToDevice);
  cudaMemcpy(device_p_scan_list,p_scan_list,sizeof(int64_t) * (g.num_nodes()+1),cudaMemcpyHostToDevice);

  


  std::cout << "Computing random walk for " << g.num_nodes() << " nodes and " 
      << g.num_edges() << " edges." << std::endl;
  max_walk_length++;
  NodeID *global_walk = new NodeID[g.num_nodes() * max_walk_length * num_walks_per_node];
  Timer t;
  t.Start();
  cudaMalloc((void **)&global_walk_device, sizeof(NodeID)*g.num_nodes() * max_walk_length * num_walks_per_node);
  cudaMemcpy(global_walk_device, global_walk, sizeof(NodeID)*g.num_nodes() * max_walk_length * num_walks_per_node,
             cudaMemcpyHostToDevice);
//   cudaMalloc((void **)&g_device, sizeof(WGraph));
//   cudaMemcpy(g_device,g,sizeof(WGraph),cudaMemcpyHostToDevice);
  dim3 blocksize(64,1,1);
  dim3 gridsize(ceil((float)g.num_nodes()/blocksize.x),1,1);
  
  for(int w_n = 0; w_n < num_walks_per_node; ++w_n) {
    std::cout << "walk number: " << w_n << std::endl;
    double* rand_double = (double *)malloc(sizeof(double)*(max_walk_length-1)* g.num_nodes());
    int* rand_int = (int*)malloc(sizeof(int) * (max_walk_length-1)*g.num_nodes());
    for (int i=0; i < (max_walk_length-1)* g.num_nodes(); i++ ){
        rand_double[i] =  RandomNumberGenerator();
    }
    for (int i=0; i < (max_walk_length-1)* g.num_nodes(); i++ ){
         rand_int[i]= rand();
    }
    double * rand_device;
    int    * rand_int_device;
    cudaMalloc((void **)&rand_device,sizeof(double)*(max_walk_length-1)* g.num_nodes());
    cudaMemcpy(rand_device,rand_double,sizeof(double)*(max_walk_length-1)* g.num_nodes(),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&rand_int_device,sizeof(int) * (max_walk_length-1)*g.num_nodes());
    cudaMemcpy(rand_int_device,rand_int,sizeof(int)*(max_walk_length-1)* g.num_nodes(),cudaMemcpyHostToDevice);

    random_per_node<<<gridsize,blocksize>>>(w_n,g.num_nodes(),
    max_walk_length,
    num_walks_per_node,
    global_walk_device, 
    rand_device,
    rand_int,
    device_outdegree_list,
    device_v_list,
    device_w_list,
    device_p_scan_list);
    cudaDeviceSynchronize();
    cudaFree(rand_device);
    cudaFree(rand_int_device);
    free(rand_double);
    free(rand_int);
  }
  cudaMemcpy(global_walk, global_walk_device, sizeof(NodeID)*g.num_nodes() * max_walk_length * num_walks_per_node, cudaMemcpyDeviceToHost);
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
  


  wbTime_start(GPU, "Freeing GPU Memory");
  
  
  wbTime_stop(GPU, "Freeing GPU Memory");

  //wbSolution(args, hostOutput, dim);

  //free(hostCSRCols);
  

  return 0;
}
