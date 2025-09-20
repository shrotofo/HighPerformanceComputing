#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdint.h>
#include <omp.h>
#include "mpi.h"
#include <sys/time.h>
#include <mm_malloc.h>


#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

#define MASTER 0

//FLATTENED SOA-SEP BLOCK FOR EACH SPEED

/* struct to hold the parameter values */
typedef struct {
  int   nx;             /* no. of cells in x-direction */
  int   ny;             /* no. of cells in y-direction */
  int   maxIters;       /* no. of iterations */
  int   reynolds_dim;   /* dimension for Reynolds number */
  float density;        /* density per link */
  float accel;          /* density redistribution */
  float omega;          /* relaxation parameter */
} t_param;

int rank; //holds ID of current MPI process - rank 0 rank 1 
int left; //ID of left neighbour - halo exchANGE(SENDING top row to left neighbour recieving their bottom ro )
int right;
int size; //total number of process running
int tag = 0; //message identification 
int rows_v; //num of rows assigned per rank 
int cols_v; //num of col in domain-same in all rank 
int remote_nrows; //used for master rank 0 to temp store number of rows assigned other rank- scatter/gather
MPI_Status status; //to store mpi related return data 
int total_cells; //tot non obs cells 

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, float* restrict cells, float* restrict tmp_cells, const int* restrict obstacles, int rank);
int write_values(const t_param params, float* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* cells);

/* compute average velocity */
float av_velocity(const t_param params, float* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

float collision(const t_param params, float* cells, float* tmp_cells, const int* obstacles, int rank);
int accelerate_flow(const t_param params, float* cells, float* tmp_cells, const int* obstacles, int rank);
void halo_exchange(float* cells, int rows_v, int cols_v, int left, int right, MPI_Status* status, int tag);


int AssignedRows(int rank, int size, int ny);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]) {
  char*    paramfile    = NULL; /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  int* obstacles = NULL;    /* grid indicating which cells are blocked */
  float*   av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */
  float*   cells;
  float*   tmp_cells;
  struct timeval timstr;
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc;

  

  //MPI intialization
  MPI_Init(&argc, &argv);
 //domain decomposition
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // parse the command line
  if (argc != 3) {
    usage(argv[0]);
  } else {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts- initialise data structures, load values from file */

  if (rank == MASTER) {
    gettimeofday(&timstr, NULL);
    tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    init_tic = tot_tic;
}
 // sets full simulation grid 
  if (rank == MASTER) {
    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);
  }
  //calc rank of neighbours for halo 
  left  = (rank - 1 + size) % size;
  right = (rank + 1)        % size;

  //rank 0 sends-broadcasts all params to all process and all process receives the params -PARAMTER 

  if (rank == MASTER) {
    for (int k = 1; k < size; k++) {
      MPI_Ssend(&params.nx, 1, MPI_INT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.ny, 1, MPI_INT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.maxIters, 1, MPI_INT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.reynolds_dim, 1, MPI_INT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.density, 1, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.accel, 1, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.omega, 1, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(&params.nx, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.ny, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.maxIters, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.reynolds_dim, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.density, 1, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.accel, 1, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.omega, 1, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD, &status);
  }

  /* Init time stops here, compute time starts*/

  if (rank == MASTER) {
    gettimeofday(&timstr, NULL);
    init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    comp_tic = init_toc;
}

//divides the domain horizontally 
rows_v= AssignedRows(rank, size, params.ny);
cols_v = params.nx;

   //local variable  buffer for each rank

  int* local_obstacles = _mm_malloc(sizeof(int) * (cols_v * rows_v), 64);
  float* local_av_vels = _mm_malloc(sizeof(float) * params.maxIters, 64);

  float* local_cells = _mm_malloc(sizeof(float) * NSPEEDS * (rows_v + 2) * cols_v, 64);
  float* local_tmp_cells = _mm_malloc(sizeof(float) * NSPEEDS * (rows_v + 2) * cols_v, 64);

  //rank 0 copies to local cells and stores in local buffer(for each rank) and sends to other process through buffer and other process recieves grid/CELLS DATA based on slices 
  for (int x = 0; x < params.nx; x++) { //iterating over cols 
    if (rank == MASTER) {
      for (int y = 1; y < (rows_v + 2) - 1; y++) { //iterating over internal rows- skip halo rows
        //copying each speed data from cells to local cells for current process  
        local_cells[0 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells[0 * params.ny * params.nx + (x + (y-1)*params.nx)];
        local_cells[1 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells[1 * params.ny * params.nx + (x + (y-1)*params.nx)];
        local_cells[2 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells[2 * params.ny * params.nx + (x + (y-1)*params.nx)];
        local_cells[3 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells[3 * params.ny * params.nx + (x + (y-1)*params.nx)];
        local_cells[4 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells[4 * params.ny * params.nx + (x + (y-1)*params.nx)];
        local_cells[5 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells[5 * params.ny * params.nx + (x + (y-1)*params.nx)];
        local_cells[6 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells[6 * params.ny * params.nx + (x + (y-1)*params.nx)];
        local_cells[7 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells[7 * params.ny * params.nx + (x + (y-1)*params.nx)];
        local_cells[8 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells[8 * params.ny * params.nx + (x + (y-1)*params.nx)];
        local_obstacles[x + (y-1)*params.nx] = obstacles[x + (y-1)*params.nx];
      }
      for (int k = 1; k < size; k++) {
         
        remote_nrows = AssignedRows(k, size, params.ny);
       //temp buffer for sending
        float* cells_send_buffer = (float*) malloc(sizeof(float) * remote_nrows * NSPEEDS);
        int* obstacles_send_buffer = (int*) malloc(sizeof(int) * remote_nrows);
        for (int y = 0; y < remote_nrows; y++) {
          cells_send_buffer[y + 0*remote_nrows] = cells[0 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)];
          cells_send_buffer[y + 1*remote_nrows] = cells[1 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)];
          cells_send_buffer[y + 2*remote_nrows] = cells[2 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)];
          cells_send_buffer[y + 3*remote_nrows] = cells[3 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)];
          cells_send_buffer[y + 4*remote_nrows] = cells[4 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)];
          cells_send_buffer[y + 5*remote_nrows] = cells[5 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)];
          cells_send_buffer[y + 6*remote_nrows] = cells[6 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)];
          cells_send_buffer[y + 7*remote_nrows] = cells[7 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)];
          cells_send_buffer[y + 8*remote_nrows] = cells[8 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)];
          obstacles_send_buffer[y] = obstacles[x + (rows_v * k + y) * params.nx];
        }
        //sends to worker temp buffer using mpi
        MPI_Ssend(cells_send_buffer, remote_nrows * NSPEEDS, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
        MPI_Ssend(obstacles_send_buffer, remote_nrows, MPI_INT, k, tag, MPI_COMM_WORLD);
      }
    } else {
      //alocate space to recive slice 
      float* cells_recv_buffer = (float*) malloc(sizeof(float) * rows_v * NSPEEDS);
      int* obstacles_recv_buffer = (int*) malloc(sizeof(int) * rows_v);
      MPI_Recv(cells_recv_buffer, rows_v * NSPEEDS, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD, &status);
      MPI_Recv(obstacles_recv_buffer, rows_v, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
      for (int y = 1; y < (rows_v + 2) - 1; y++) {
        local_cells[0 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells_recv_buffer[y-1 + 0*rows_v];
        local_cells[1 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells_recv_buffer[y-1 + 1*rows_v];
        local_cells[2 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells_recv_buffer[y-1 + 2*rows_v];
        local_cells[3 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells_recv_buffer[y-1 + 3*rows_v];
        local_cells[4 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells_recv_buffer[y-1 + 4*rows_v];
        local_cells[5 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells_recv_buffer[y-1 + 5*rows_v];
        local_cells[6 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells_recv_buffer[y-1 + 6*rows_v];
        local_cells[7 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells_recv_buffer[y-1 + 7*rows_v];
        local_cells[8 * (rows_v+2) * cols_v + (x + y*params.nx)] = cells_recv_buffer[y-1 + 8*rows_v];
        local_obstacles[x + (y-1)*params.nx] = obstacles_recv_buffer[y-1];
      }
    }
  }

  

  for (int tt = 0; tt < params.maxIters; tt+=2) {
    //returns avg velocity for that step 
    local_av_vels[tt]   = timestep(params, local_cells, local_tmp_cells, local_obstacles, rank);
    local_av_vels[tt+1] = timestep(params, local_tmp_cells, local_cells, local_obstacles, rank); 
    #ifdef DEBUG
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      printf("tot density: %.12E\n", total_density(params, cells));
    #endif
  }

  //CALCULATE LOCAL VELOCITY 
  //combines all process local_av_vels to av_vels on rank 0 

  MPI_Reduce(local_av_vels, av_vels, params.maxIters, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  if (rank == MASTER) {
    for (int i = 0; i < params.maxIters; i++) {
      av_vels[i] /= total_cells;
    }
  }

   /* Compute time stops here, collate time starts*/

  if (rank == MASTER) {
    gettimeofday(&timstr, NULL);
    comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    col_tic = comp_toc;
}

// Collate data from ranks here


//gathers final grid from workers into masters back into cells

  for (int x = 0; x < params.nx; x++) {
    if (rank == MASTER) {
      for (int y = 1; y < (rows_v + 2) - 1; y++) {
        cells[0 * params.ny * params.nx + (x + (y-1)*params.nx)] = local_cells[0 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells[1 * params.ny * params.nx + (x + (y-1)*params.nx)] = local_cells[1 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells[2 * params.ny * params.nx + (x + (y-1)*params.nx)] = local_cells[2 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells[3 * params.ny * params.nx + (x + (y-1)*params.nx)] = local_cells[3 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells[4 * params.ny * params.nx + (x + (y-1)*params.nx)] = local_cells[4 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells[5 * params.ny * params.nx + (x + (y-1)*params.nx)] = local_cells[5 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells[6 * params.ny * params.nx + (x + (y-1)*params.nx)] = local_cells[6 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells[7 * params.ny * params.nx + (x + (y-1)*params.nx)] = local_cells[7 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells[8 * params.ny * params.nx + (x + (y-1)*params.nx)] = local_cells[8 * (rows_v+2) * cols_v + (x + y*params.nx)];
      }
      for (int k = 1; k < size; k++) {
        
        remote_nrows = AssignedRows(k, size, params.ny);

        float* cells_recv_buffer = (float*) malloc(sizeof(float) * remote_nrows * NSPEEDS);
        MPI_Recv(cells_recv_buffer, remote_nrows * NSPEEDS, MPI_FLOAT, k, tag, MPI_COMM_WORLD, &status);
        for (int y = 0; y < remote_nrows; y++) {
          cells[0 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)] = cells_recv_buffer[y + 0*remote_nrows];
          cells[1 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)] = cells_recv_buffer[y + 1*remote_nrows];
          cells[2 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)] = cells_recv_buffer[y + 2*remote_nrows];
          cells[3 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)] = cells_recv_buffer[y + 3*remote_nrows];
          cells[4 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)] = cells_recv_buffer[y + 4*remote_nrows];
          cells[5 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)] = cells_recv_buffer[y + 5*remote_nrows];
          cells[6 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)] = cells_recv_buffer[y + 6*remote_nrows];
          cells[7 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)] = cells_recv_buffer[y + 7*remote_nrows];
          cells[8 * params.ny * params.nx + (x + (rows_v * k + y) * params.nx)] = cells_recv_buffer[y + 8*remote_nrows];
        }
      }
    } else {
      float* cells_send_buffer = (float*) malloc(sizeof(float) * rows_v * NSPEEDS);
      for (int y = 1; y < (rows_v + 2) - 1; y++) {
        cells_send_buffer[y-1 + 0*rows_v] = local_cells[0 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells_send_buffer[y-1 + 1*rows_v] = local_cells[1 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells_send_buffer[y-1 + 2*rows_v] = local_cells[2 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells_send_buffer[y-1 + 3*rows_v] = local_cells[3 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells_send_buffer[y-1 + 4*rows_v] = local_cells[4 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells_send_buffer[y-1 + 5*rows_v] = local_cells[5 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells_send_buffer[y-1 + 6*rows_v] = local_cells[6 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells_send_buffer[y-1 + 7*rows_v] = local_cells[7 * (rows_v+2) * cols_v + (x + y*params.nx)];
        cells_send_buffer[y-1 + 8*rows_v] = local_cells[8 * (rows_v+2) * cols_v + (x + y*params.nx)];
      }
      MPI_Ssend(cells_send_buffer, rows_v * NSPEEDS, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
    }
  }

   /* Total/collate time stops here.*/
   gettimeofday(&timstr, NULL);
   col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
   tot_toc = col_toc;

  _mm_free(local_obstacles);
  _mm_free(local_av_vels);

  if (rank == MASTER) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
    write_values(params, cells, obstacles, av_vels);
    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
}

  MPI_Finalize();

}
//subdomain of grid based on rows per mpi rank 
float collision(const t_param params, float* cells, float* tmp_cells, const int* obstacles, int rank){

  const float c_sq = 1.f / 3.f;  // square of speed of sound
  const float w0   = 4.f / 9.f;  // weighting factor
  const float w1   = 1.f / 9.f;  // weighting factor
  const float w2   = 1.f / 36.f; // weighting factor
  const float denominator = 2.f * c_sq * c_sq; 


  float velocity_u= 0.f;   // accumulated magnitudes of velocity for each cell

  int local_active_cells = 0;



  
  //loop over local grid exclude halo rows
  for (int jj = 1; jj < rows_v + 1; jj++) {
    for (int ii = 0; ii < cols_v; ii++) {

      const int y_n = jj + 1;
      const int x_e = (ii + 1) % cols_v;
      const int y_s = (jj - 1);
      const int x_w = (ii == 0) ? (ii + cols_v - 1) : (ii - 1);
     //read speed manually in flat float array 
      const float speed0 = cells[(0 * (rows_v+2) * cols_v) + (ii + jj*cols_v)];
      const float speed1 = cells[(1 * (rows_v+2) * cols_v) + (x_w + jj*cols_v)];
      const float speed2 = cells[(2 * (rows_v+2) * cols_v) + (ii + y_s*cols_v)];
      const float speed3 = cells[(3 * (rows_v+2) * cols_v) + (x_e + jj*cols_v)];
      const float speed4 = cells[(4 * (rows_v+2) * cols_v) + (ii + y_n*cols_v)];
      const float speed5 = cells[(5 * (rows_v+2) * cols_v) + (x_w + y_s*cols_v)];
      const float speed6 = cells[(6 * (rows_v+2) * cols_v) + (x_e + y_s*cols_v)];
      const float speed7 = cells[(7 * (rows_v+2) * cols_v) + (x_e + y_n*cols_v)];
      const float speed8 = cells[(8 * (rows_v+2) * cols_v) + (x_w + y_n*cols_v)];

      // compute local density total
      const float local_density = speed0 + speed1 + speed2
                                + speed3 + speed4 + speed5
                                + speed6 + speed7 + speed8;

      // compute x and y velocity components
      const float u_x = (speed1 + speed5 + speed8 - (speed3 + speed6 + speed7)) / local_density;
      const float u_y = (speed2 + speed5 + speed6 - (speed4 + speed7 + speed8)) / local_density;

      // if the cell contains an obstacle
      if (obstacles[(jj-1)*cols_v + ii]) {
        //rebound step 

        tmp_cells[(0 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed0;
        tmp_cells[(1 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed3;
        tmp_cells[(2 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed4;
        tmp_cells[(3 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed1;
        tmp_cells[(4 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed2;
        tmp_cells[(5 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed7;
        tmp_cells[(6 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed8;
        tmp_cells[(7 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed5;
        tmp_cells[(8 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed6;

      } else {
        // Calculate equilibrium densities

        const float constant = 1.f - (u_x * u_x + u_y * u_y) * 1.5f;

        // directional velocity components
        const float u1 =   u_x;        // east
        const float u2 =         u_y;  // north
        const float u3 = - u_x;        // west
        const float u4 =       - u_y;  // south
        const float u5 =   u_x + u_y;  // north-east
        const float u6 = - u_x + u_y;  // north-west
        const float u7 = - u_x - u_y;  // south-west
        const float u8 =   u_x - u_y;  // south-east

        // relaxation step
        tmp_cells[(0 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed0 + params.omega * (w0 * local_density * constant - speed0);
        tmp_cells[(1 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed1 + params.omega * (w1 * local_density * (u1 / c_sq + (u1 * u1) / denominator + constant) - speed1);
        tmp_cells[(2 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed2 + params.omega * (w1 * local_density * (u2 / c_sq + (u2 * u2) / denominator + constant) - speed2);
        tmp_cells[(3 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed3 + params.omega * (w1 * local_density * (u3 / c_sq + (u3 * u3) / denominator + constant) - speed3);
        tmp_cells[(4 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed4 + params.omega * (w1 * local_density * (u4 / c_sq + (u4 * u4) / denominator + constant) - speed4);
        tmp_cells[(5 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed5 + params.omega * (w2 * local_density * (u5 / c_sq + (u5 * u5) / denominator + constant) - speed5);
        tmp_cells[(6 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed6 + params.omega * (w2 * local_density * (u6 / c_sq + (u6 * u6) / denominator + constant) - speed6);
        tmp_cells[(7 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed7 + params.omega * (w2 * local_density * (u7 / c_sq + (u7 * u7) / denominator + constant) - speed7);
        tmp_cells[(8 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] = speed8 + params.omega * (w2 * local_density * (u8 / c_sq + (u8 * u8) / denominator + constant) - speed8);

        // Accumulate velocity magnitude 
      velocity_u += (obstacles[(jj-1)*cols_v + ii]) ? 0 : sqrtf((u_x * u_x) + (u_y * u_y));
      // After calculating velocity_u
      local_active_cells++;
      }

      

    }
  }

  if (local_active_cells > 0)
  return velocity_u / (float)local_active_cells;
else
  return 0.f;



}

int accelerate_flow(const t_param params, float* cells, float* tmp_cells, const int* obstacles, int rank){

    //only last rank -contains second-to-last global row
  if (rank == size - 1) {

    // accelerate_flow

    // compute weighting factors
    const float init_w1 = params.density * params.accel / 9.f;
    const float init_w2 = params.density * params.accel / 36.f;

    // modify the 2nd row of the grid
    const int jj = (rows_v + 1) - 2;

    for (int ii = 0; ii < cols_v; ii++) {
      /* if the cell is not occupied and
      ** we don't send a negative density */
      if (!obstacles[ii + (jj-1)*cols_v]
      && (cells[(3 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] - init_w1) > 0.f
      && (cells[(6 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] - init_w2) > 0.f
      && (cells[(7 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] - init_w2) > 0.f) {
        /* increase 'east-side' densities */
        cells[(1 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] += init_w1;
        cells[(5 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] += init_w2;
        cells[(8 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] += init_w2;
        /* decrease 'west-side' densities */
        cells[(3 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] -= init_w1;
        cells[(6 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] -= init_w2;
        cells[(7 * (rows_v+2) * cols_v) + (ii + jj*cols_v)] -= init_w2;
      }
    }

  }

}


void halo_exchange(float* cells, int rows_v, int cols_v, int left, int right, MPI_Status* status, int tag) {
  float* send_buffer = (float*) malloc(sizeof(float) * cols_v * NSPEEDS);
  float* recv_buffer = (float*) malloc(sizeof(float) * cols_v * NSPEEDS);

  // --- Send to left, receive from right ---
  for (int x = 0; x < cols_v; x++) {
      for (int d = 0; d < NSPEEDS; d++) {
          send_buffer[x + d*cols_v] = cells[(d * (rows_v + 2) * cols_v) + (x + 1*cols_v)];
      }
  }

  MPI_Sendrecv(send_buffer, cols_v * NSPEEDS, MPI_FLOAT, left, tag,
    recv_buffer, cols_v * NSPEEDS, MPI_FLOAT, right, tag,
               MPI_COMM_WORLD, status);

  for (int x = 0; x < cols_v; x++) {
      for (int d = 0; d < NSPEEDS; d++) {
          cells[(d * (rows_v + 2) * cols_v) + (x + (rows_v + 1)*cols_v)] = recv_buffer[x + d*cols_v];
      }
  }

  // --- Send to right, receive from left ---
  for (int x = 0; x < cols_v; x++) {
      for (int d = 0; d < NSPEEDS; d++) {
          send_buffer[x + d*cols_v] = cells[(d * (rows_v + 2) * cols_v) + (x + rows_v*cols_v)];
      }
  }

  MPI_Sendrecv(send_buffer, cols_v * NSPEEDS, MPI_FLOAT, right, tag,
               recv_buffer, cols_v * NSPEEDS, MPI_FLOAT, left, tag,
               MPI_COMM_WORLD, status);

  for (int x = 0; x < cols_v; x++) {
      for (int d = 0; d < NSPEEDS; d++) {
          cells[(d * (rows_v + 2) * cols_v) + x] = recv_buffer[x + d*cols_v];
      }
  }

  free(send_buffer);
  free(recv_buffer);
}


float timestep(const t_param params, float* restrict cells, float* restrict tmp_cells, const int* restrict obstacles, int rank) {

  halo_exchange(cells, rows_v, cols_v, left, right, &status, tag);


  accelerate_flow(params, cells, tmp_cells, obstacles, rank); //external acceleration to the second-to-last row in the last rank
  return collision(params, cells, tmp_cells, obstacles, rank); //calc local density,velocities, applied BGK relaxation, updates tmp_cells new values 

  
  
}


float av_velocity(const t_param params, float* cells, int* obstacles) {
  int   tot_cells  = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {

      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx]) {
        /* local density total */
        //indexing manually for each direction-SOA 
        float local_density = cells[(0 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(1 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(2 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(3 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(4 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(5 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(6 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(7 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(8 * params.ny * params.nx) + (ii + jj*params.nx)];

        /* compute x velocity component */
        float u_x = (cells[(1 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(5 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(8 * params.ny * params.nx) + (ii + jj*params.nx)]
                  - (cells[(3 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(6 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(7 * params.ny * params.nx) + (ii + jj*params.nx)]))
                   / local_density;
        /* compute y velocity component */
        float u_y = (cells[(2 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(5 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(6 * params.ny * params.nx) + (ii + jj*params.nx)]
                  - (cells[(4 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(7 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(8 * params.ny * params.nx) + (ii + jj*params.nx)]))
                   / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells ;
      }
    }
  }

  return tot_u / (float) tot_cells ;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr) {
  char  message[1024]; /* message buffer */
  FILE* fp;            /* file pointer */
  int   xx, yy;        /* generic array indices */
  int   blocked;       /* indicates whether a cell is blocked by an obstacle */
  int   retval;        /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  // main grid
  *cells_ptr = malloc(sizeof(float) * NSPEEDS * params->ny * params->nx);
  if (*cells_ptr == NULL) die("cannot allocate memory for cell speeds", __LINE__, __FILE__);

  // 'helper' grid, used as scratch space
  *tmp_cells_ptr = malloc(sizeof(float) * NSPEEDS * params->ny * params->nx);
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cell speeds", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * (params->ny * params->nx), 64);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density       / 9.f;
  float w2 = params->density       / 36.f;

  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      // centre
      (*cells_ptr)[(0 * params->ny * params->nx) + (ii + jj*params->nx)] = w0;
      // axis directions
      (*cells_ptr)[(1 * params->ny * params->nx) + (ii + jj*params->nx)] = w1;
      (*cells_ptr)[(2 * params->ny * params->nx) + (ii + jj*params->nx)] = w1;
      (*cells_ptr)[(3 * params->ny * params->nx) + (ii + jj*params->nx)] = w1;
      (*cells_ptr)[(4 * params->ny * params->nx) + (ii + jj*params->nx)] = w1;
      // diagonals
      (*cells_ptr)[(5 * params->ny * params->nx) + (ii + jj*params->nx)] = w2;
      (*cells_ptr)[(6 * params->ny * params->nx) + (ii + jj*params->nx)] = w2;
      (*cells_ptr)[(7 * params->ny * params->nx) + (ii + jj*params->nx)] = w2;
      (*cells_ptr)[(8 * params->ny * params->nx) + (ii + jj*params->nx)] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  total_cells = params->nx * params->ny;

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;

    total_cells--;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*) _mm_malloc(sizeof(float) * params->maxIters, 64);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr) {


     /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

             
  return EXIT_SUCCESS;

}

float calc_reynolds(const t_param params, float* cells, int* obstacles) {
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells) {
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      total = cells[(0 * params.ny * params.nx) + (ii + jj*params.nx)]
            + cells[(1 * params.ny * params.nx) + (ii + jj*params.nx)]
            + cells[(2 * params.ny * params.nx) + (ii + jj*params.nx)]
            + cells[(3 * params.ny * params.nx) + (ii + jj*params.nx)]
            + cells[(4 * params.ny * params.nx) + (ii + jj*params.nx)]
            + cells[(5 * params.ny * params.nx) + (ii + jj*params.nx)]
            + cells[(6 * params.ny * params.nx) + (ii + jj*params.nx)]
            + cells[(7 * params.ny * params.nx) + (ii + jj*params.nx)]
            + cells[(8 * params.ny * params.nx) + (ii + jj*params.nx)];
    }
  }

  return total;
}

int write_values(const t_param params, float* cells, int* obstacles, float* av_vels) {
  FILE* fp;                     // file pointer
  const float c_sq = 1.f / 3.f; // sq. of speed of sound
  float local_density;          // per grid cell sum of densities
  float pressure;               // fluid pressure in grid cell
  float u_x;                    // x-component of velocity in grid cell
  float u_y;                    // y-component of velocity in grid cell
  float u;                      // norm--root of summed squares--of u_x and u_y

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx]) {

        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;

      } else { /* no obstacle */

        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++) {
          local_density += cells[(kk * params.ny * params.nx) + (ii + jj*params.nx)];
        }

      

        /* compute x velocity component */
        u_x = (cells[(1 * params.ny * params.nx) + (ii + jj*params.nx)]  // Speed 1 (east)
               + cells[(5 * params.ny * params.nx) + (ii + jj*params.nx)] // Speed 5 (north-east)
               + cells[(8 * params.ny * params.nx) + (ii + jj*params.nx)]  // Speed 8 (south-east)
               - (cells[(3 * params.ny * params.nx) + (ii + jj*params.nx)]   // Speed 3 (west)
                  + cells[(6 * params.ny * params.nx) + (ii + jj*params.nx)]  // Speed 6 (north-west)
                  + cells[(7 * params.ny * params.nx) + (ii + jj*params.nx)]))  // Speed 7 (south-west)
              / local_density; 
        /* compute y velocity component */
        u_y = (cells[(2 * params.ny * params.nx) + (ii + jj*params.nx)]   // Speed 2 (north)
               + cells[(5 * params.ny * params.nx) + (ii + jj*params.nx)]  // Speed 5 (north-east)
               + cells[(6 * params.ny * params.nx) + (ii + jj*params.nx)]  // Speed 6 (north-west)
               - (cells[(4 * params.ny * params.nx) + (ii + jj*params.nx)]  // Speed 4 (south)
                  + cells[(7 * params.ny * params.nx) + (ii + jj*params.nx)]  // Speed 7 (south-west)
                  + cells[(8 * params.ny * params.nx) + (ii + jj*params.nx)]))   // Speed 8 (south-east)
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++) {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}


void die(const char* message, const int line, const char* file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe) {
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}


//evenly divide rows if 2d grid 
int AssignedRows(int rank, int size, int ny) {
  int t_rows;

    int base = ny / size;
    int extra = ny % size;
    t_rows = base;

    if (extra && rank == size - 1) {
        t_rows += extra;
    }

    return t_rows;
}