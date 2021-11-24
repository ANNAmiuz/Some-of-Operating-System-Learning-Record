#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NAME_MAX_LENGTH 20            // bytes

#define STATUS_POSITION 31            
#define SIZE_POSITION 30              
#define ADDR_POSITION 28              // block address, 16 bit
#define DIRECT_POSITION 16
#define MODIFY_TIME_POSITION 23
#define CREATE_TIME_POSITION 20

#define MAX_CHILD_NUM 50
#define DIRECT_SIZE 1024
#define MAX_DEPTH 3

#define CHLD_NUM_OFFSET_IN_BLOCK 2
#define PARENT_OFFSET_IN_BLOCK 0


#define DEBUGx


__device__ __managed__ u32 gtime = 0;




__device__ u32 get_direct_parent(uchar * file_addr);

// global compaction
__device__ void compact(FileSystem * fs);

// get the absolute FCB addr from FCB index
__device__ uchar * get_FCB_addr_from_FCB_idx(FileSystem *fs, int FCB);

// get the absolute FILE addr from FCB index
__device__ uchar * get_file_addr_from_FCB_idx(FileSystem *fs, int FCB);

// clean SUPER & STORAGE & FCB related to the FCB
__device__ void clean_all_from_FCB_idx(FileSystem *fs, int FCB);

// clean a child record from parent
__device__ void delete_child_record_from_parent(FileSystem *fs, int child, int parent);

// doing RMRF recursively
__device__ void perform_rmrf(FileSystem *fs, int child, int parent);

__device__ void clean_from_super_block(uchar * super_block_base, u32 SUPERBLOCK_SIZE, int block_start, u32 block_num);




/* FCB helper function: use the absolute address of the FCB entry */

// return 1 if a valid FCB, else return 0
__device__ bool get_FCB_status(uchar * FCB_start) {
  return (FCB_start[ADDR_POSITION] & 0b10000000) >> 7;
}

__device__ void set_FCB_status(uchar * FCB_start, int status) {
  FCB_start[ADDR_POSITION] = (FCB_start[ADDR_POSITION] & 0b01111111) | (status << 7);
}

// return 1 if the FCB represent a directory
__device__ bool get_FCB_isDirect(uchar * FCB_start) {
  return (FCB_start[DIRECT_POSITION] & 0b10000000) >> 7;
}

__device__ void set_FCB_isDirect(uchar * FCB_start, int status) {
  FCB_start[DIRECT_POSITION] = (FCB_start[DIRECT_POSITION] & 0b01111111) | (status << 7);
}


// return the actual size in blocks (bytes) in current FCB entry
__device__ u32 get_FCB_size(uchar * FCB_start) {
  u32 c1 = (u32)FCB_start[SIZE_POSITION];
  u32 c2 = (u32)FCB_start[SIZE_POSITION+1];
  u32 size = (c1<<8) | c2;
  return size;
}

__device__ void set_FCB_size(uchar * FCB_start, int size) {
  uchar c1 = size >> 8;
  uchar c2 = size & 0xff;
  FCB_start[SIZE_POSITION] = c1;
  FCB_start[SIZE_POSITION+1] = c2;
}

// return the size blocks for directory
__device__ u32 get_FCB_Dsize(uchar * FCB_start) {
  u32 c1 = (u32)FCB_start[DIRECT_POSITION] & 0b01111111; // the first bit is the directory indicatior
  u32 c2 = (u32)FCB_start[DIRECT_POSITION+1];
  u32 size = (c1 << 8) | c2;
  return size;
}

__device__ void set_FCB_Dsize(uchar * FCB_start, int size) {
  uchar c1 = size >> 8;
  uchar c2 = size & 0xff;
  FCB_start[DIRECT_POSITION] = (FCB_start[DIRECT_POSITION] & 0b10000000) | c1;
  FCB_start[DIRECT_POSITION+1] = c2;
}


// return the addr in current FCB entry
__device__ u32 get_FCB_addr(uchar * FCB_start) {
  u32 c1 = (u32)(FCB_start[ADDR_POSITION] & 0b01111111);
  u32 c2 = (u32)FCB_start[ADDR_POSITION+1];
  u32 addr = (c1 << 8) | c2;
  return addr;
}

__device__ void set_FCB_addr(uchar * FCB_start, u32 block_addr) {
  uchar c1 = block_addr >> 8;
  uchar c2 = block_addr & 0xff;
  FCB_start[ADDR_POSITION] = c1 | (FCB_start[ADDR_POSITION] & 0x80);
  FCB_start[ADDR_POSITION+1] = c2;
}


// return the CREATE time in current FCB entry
__device__ u32 get_FCB_create_time(uchar *FCB_start) {
  u32 c1 = (u32)FCB_start[CREATE_TIME_POSITION];
  u32 c2 = (u32)FCB_start[CREATE_TIME_POSITION+1];
  u32 c3 = (u32)FCB_start[CREATE_TIME_POSITION+2];
  u32 time = (c1<<16) | (c2<<8) | (c3<<0);
  return time;
}

__device__ void set_FCB_create_time(uchar *FCB_start, u32 time) {
  uchar c1 = (time & 0xff0000) >> 16;
  uchar c2 = (time & 0xff00) >> 8;
  uchar c3 = (time & 0xff) >> 0;
  FCB_start[CREATE_TIME_POSITION] = c1;
  FCB_start[CREATE_TIME_POSITION+1] = c2;
  FCB_start[CREATE_TIME_POSITION+2] = c3;
}


// return the MODIFY time in current FCB entry
__device__ u32 get_FCB_modify_time(uchar *FCB_start) {
  u32 c1 = (u32)FCB_start[MODIFY_TIME_POSITION];
  u32 c2 = (u32)FCB_start[MODIFY_TIME_POSITION+1];
  u32 c3 = (u32)FCB_start[MODIFY_TIME_POSITION+2];
  u32 time = (c1<<16) | (c2<<8) | (c3<<0);
  return time;
}

__device__ void set_FCB_modify_time(uchar *FCB_start, u32 time) {
  u32 c1 = (time & 0xff0000) >> 16;
  u32 c2 = (time & 0xff00) >> 8;
  u32 c3 = (time & 0xff) >> 0;
  FCB_start[MODIFY_TIME_POSITION] = c1;
  FCB_start[MODIFY_TIME_POSITION+1] = c2;
  FCB_start[MODIFY_TIME_POSITION+2] = c3;
}


// return 1 if the file names match
__device__ bool match_FCB_file_name(uchar *FCB_start, char* name) {
  char c1, c2;
  for (int i = 0; i < NAME_MAX_LENGTH; ++i)
  {
    c1 = FCB_start[i];
    c2 = name[i];
    if (c1 != c2)
      return false;
    else if (c1 == c2 && c1 == '\0')
      return true;
  }
  return false;
}

__device__ void set_FCB_file_name(uchar *FCB_start, char* name) {
  for (int i = 0; i < NAME_MAX_LENGTH; ++i) {
    FCB_start[i] = name[i];
    if (name[i] == '\0') return;
  }
}

__device__ void clean_FCB_file_name(uchar *FCB_start) {
  for (int i = 0; i < NAME_MAX_LENGTH; ++i) {
    FCB_start[i] = '\0';
  }
}

__device__ void get_FCB_file_name(uchar * FCB_start, uchar * output) {
  for (int i = 0; i < NAME_MAX_LENGTH; ++i) {
    output[i] = FCB_start[i];
    if (output[i] == '\0') return;
  }
}

__device__ void print_FCB_path_name(FileSystem * fs, int FCB) {
  int * path_buf = (int *)malloc(sizeof(int) * (MAX_DEPTH+1));
  int pwd_idx = FCB;
  int cur = 0;
  while (pwd_idx != 0) {
    path_buf[cur++] = pwd_idx;
    uchar * current_FCB = fs->volume+fs->FCB_BASE_ADDRESS+pwd_idx*fs->FCB_SIZE;
    u32 block_addr = get_FCB_addr(current_FCB);
    uchar * file_addr = fs->volume+fs->FILE_BASE_ADDRESS+block_addr*fs->STORAGE_BLOCK_SIZE;
    pwd_idx = get_direct_parent(file_addr);
  }
  path_buf[cur] = 0;
  printf("/");
  for (int i = cur-1; i >= 0; i--){
    uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + path_buf[i]*fs->FCB_SIZE;
    uchar * name = (uchar *)malloc(sizeof(uchar)*NAME_MAX_LENGTH);
    get_FCB_file_name(current_FCB, name);
    printf("%s", name);
    if (i != 0)
      printf("/");
  }
  printf("\n");
}

__device__ int get_empty_FCB(FileSystem * fs) {
  for (int i = 0; i < fs->FCB_ENTRIES; ++i) {
    uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * i;
    if (get_FCB_status(current_FCB) == 0) return i;
  }
  printf("No more valid fcb to palce\n");
  return -1;
}







/* Block manipulation helper: use the absolute address of the file */

// clean up all blocks for the current file 
__device__ void clean_file(uchar * file_addr, int block_num, int BLOCK_SIZE) {
  uchar * current_block;
  for (int i = 0; i < block_num; ++i) {
    current_block = file_addr + i * BLOCK_SIZE;
    for (int j = 0; j < BLOCK_SIZE; ++j) {
      current_block[j] = '\0';
    }
  }
}

// write to file
__device__ void write_file(uchar * file_addr, uchar * input, int size, int BLOCK_SIZE) {
  int num_of_blocks = (size - 1) / BLOCK_SIZE + 1;
  int count = 0;
  uchar * current_block;
  for (int i = 0; i < num_of_blocks; ++i) {
    current_block = file_addr + i * BLOCK_SIZE;
    for (int j = 0; j < BLOCK_SIZE; ++j) {
      current_block[j] = input[count];
      ++count;
      if (count == size) return;
    }
  }
}

// read from file
__device__ void read_file(uchar * file_addr, uchar * output, int size, int BLOCK_SIZE) {
  int num_of_blocks = (size - 1) / BLOCK_SIZE + 1;
  int count = 0;
  uchar * current_block;
  for (int i = 0; i < num_of_blocks; ++i) {
    current_block = file_addr + i * BLOCK_SIZE;
    for (int j = 0; j < BLOCK_SIZE; ++j) {
      output[count] = current_block[j];
      ++count;
      if (count == size) return;
    }
  }
}

// get the number of child in a directory block
__device__ u32 read_direct_child_num(uchar * file_addr) {
    return (u32)file_addr[CHLD_NUM_OFFSET_IN_BLOCK];
}

// set the number of child in a direct block
__device__ void set_direct_child_num(uchar * file_addr, int num) {
  file_addr[CHLD_NUM_OFFSET_IN_BLOCK] = num;
}

// match the child idx under the directory using directory storage block, i start from 0
__device__ u32 get_direct_ithChld_idx(uchar * file_addr, int i) {
  u32 c1 = (u32)file_addr[CHLD_NUM_OFFSET_IN_BLOCK+1+2*i];
  u32 c2 = (u32)file_addr[CHLD_NUM_OFFSET_IN_BLOCK+2+2*i];
  return (c1 << 8) | c2;
}

// write child idx under the directory, i start from 0
__device__ void write_direct_ithChld_idx(uchar * file_addr, int i, int child_idx) {
  file_addr[CHLD_NUM_OFFSET_IN_BLOCK+1+2*i] = (child_idx >> 8);
  file_addr[CHLD_NUM_OFFSET_IN_BLOCK+2+2*i] = (child_idx & 0xff);
}

// get parent FCB idx from direct storage block
__device__ u32 get_direct_parent(uchar * file_addr) {
  u32 c1 = (u32)file_addr[PARENT_OFFSET_IN_BLOCK];
  u32 c2 = (u32)file_addr[PARENT_OFFSET_IN_BLOCK+1];
  return (c1 << 8) | c2;
}

// set parent FCB idx
__device__ void set_direct_parent(uchar * file_addr, int parent_idx) {
  file_addr[PARENT_OFFSET_IN_BLOCK] = parent_idx >> 8;
  file_addr[PARENT_OFFSET_IN_BLOCK+1] = parent_idx & 0xff;
}

// find a child under a direct, if found, return positive
__device__ int find_child_under_PWD(FileSystem * fs, char * s) {
  uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fs->PWD_FCB_idx * fs->FCB_SIZE;
  u32 block_addr = get_FCB_addr(current_FCB);
  uchar * file_addr = fs->volume + fs->FILE_BASE_ADDRESS + block_addr * fs->STORAGE_BLOCK_SIZE;
  u32 child_num = read_direct_child_num(file_addr);
  for (int i = 0; i < child_num; i++) {
    u32 child_FCB_idx = get_direct_ithChld_idx(file_addr, i);
    uchar * current_child_FCB = fs->volume + fs->FCB_BASE_ADDRESS + child_FCB_idx * fs->FCB_SIZE;
    if (get_FCB_status(current_child_FCB)==true && match_FCB_file_name(current_child_FCB, s)) {
      // if (get_FCB_isDirect(current_child_FCB) == true) 
      //   printf("Invalid operation to a directory file.\n");
      #ifdef DEBUG
      printf("[DEBUG] File %s exist, with the FCB %d \n", s, child_FCB_idx);
      #endif
      return child_FCB_idx;
    }
  }
  return -1;
}


__device__ void delete_child_record_from_parent(FileSystem *fs, int child, int parent) {
  #ifdef DEBUG
  printf("[DEBUG] delete child record %d in parent %d\n", child, parent);
  #endif
  #ifdef DEBUG
  printf("before delete: all the child FCB is:\n");
  for (int i = 0; i < read_direct_child_num(get_file_addr_from_FCB_idx(fs, parent)); ++i) {
      printf("[--DEBUG] child %d FCB %d \n", i, get_direct_ithChld_idx(get_file_addr_from_FCB_idx(fs, parent), i));
  }
  #endif
  uchar * current_FCB = get_FCB_addr_from_FCB_idx(fs, parent);
  uchar * file_addr = get_file_addr_from_FCB_idx(fs, parent);
  u32 child_num = read_direct_child_num(file_addr);

  int i = 0;
  bool found = false;
  for (; i < child_num; i++) {
    u32 cur_child = get_direct_ithChld_idx(file_addr, i);
    if (cur_child == child) {
      found = true;
      break;
    }
  }
  if (found == false) return;
  #ifdef DEBUG
  printf("[DEBUG] %d is the %d th child of %d (deleted) \n", child, i, parent);
  #endif

  // move all child idx up
  for (int j = i; j < child_num; j++) {
    write_direct_ithChld_idx(file_addr, j, get_direct_ithChld_idx(file_addr, j+1));
  }
  write_direct_ithChld_idx(file_addr, child_num, 0);

  
  uchar * name_buf = (uchar *)malloc(sizeof(char) * NAME_MAX_LENGTH);
  get_FCB_file_name(get_FCB_addr_from_FCB_idx(fs, child), name_buf);
  #ifdef DEBUG
  printf("filename is %s, FCB IDX is %d\n", name_buf, child);
  #endif

  int file_name_length = 0, k = 0;
  while (name_buf[k] != '\0') {
    file_name_length++;
    ++k;
  }
  file_name_length++;
  set_FCB_Dsize(current_FCB, get_FCB_Dsize(current_FCB)-file_name_length);

  #ifdef DEBUG
  printf("filename is %s, length is %d\n", name_buf, file_name_length);
  printf("[DEBUG] fcb dsize reduce from %d to %d\n", get_FCB_Dsize(current_FCB) + file_name_length, get_FCB_Dsize(current_FCB));
  #endif

  set_FCB_modify_time(current_FCB, gtime++);
  set_direct_child_num(file_addr, child_num-1); // reduce number of child

  #ifdef DEBUG
  printf("before delete: all the child FCB is:\n");
  for (int i = 0; i < read_direct_child_num(get_file_addr_from_FCB_idx(fs, parent)); ++i) {
      printf("[--DEBUG] child %d FCB %d \n", i, get_direct_ithChld_idx(get_file_addr_from_FCB_idx(fs, parent), i));
  }
  #endif
}


__device__ void perform_rmrf(FileSystem *fs, int child, int parent){
  uchar * current_FCB = get_FCB_addr_from_FCB_idx(fs, child);

  if (get_FCB_isDirect(current_FCB) == false) {
    delete_child_record_from_parent(fs, child, parent);
    clean_all_from_FCB_idx(fs, child);
    compact(fs);
    return;
  }
  
  else if (get_FCB_isDirect(current_FCB) == true) {
    uchar * file_addr = get_file_addr_from_FCB_idx(fs, child);
    u32 child_num = read_direct_child_num(file_addr);
    
    for (int i = 0; i < child_num; i++) {
      #ifdef DEBUG
      printf("\n");
      printf("[xxx]child num %d of parent %d\n", child_num, child);
      printf("[xxx]get child %d\n", get_direct_ithChld_idx(file_addr, i));
      #endif
      u32 cur_child = get_direct_ithChld_idx(file_addr, 0);
      perform_rmrf(fs, cur_child, child);
    }
    delete_child_record_from_parent(fs, child, parent);
    clean_all_from_FCB_idx(fs, child);
    compact(fs);
    return;
  }
}


// get the absolute FCB addr from FCB index
__device__ uchar * get_FCB_addr_from_FCB_idx(FileSystem *fs, int FCB){
  uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + FCB * fs->FCB_SIZE;
  return current_FCB;
}

// get the absolute FILE addr from FCB index
__device__ uchar * get_file_addr_from_FCB_idx(FileSystem *fs, int FCB){
  uchar * current_FCB = get_FCB_addr_from_FCB_idx(fs, FCB);
  u32 block_addr = get_FCB_addr(current_FCB);
  return fs->volume + fs->FILE_BASE_ADDRESS + block_addr * fs->STORAGE_BLOCK_SIZE;
}



__device__ void clean_all_from_FCB_idx(FileSystem *fs, int FCB){
  #ifdef DEBUG
  printf("[DEBUG] clean everything of FCB %d\n", FCB);
  #endif
  uchar * current_child_FCB = get_FCB_addr_from_FCB_idx(fs, FCB);
  u32 block_addr = get_FCB_addr(current_child_FCB);
  u32 file_size = get_FCB_size(current_child_FCB);
  int cur_num_of_block = (file_size-1) / fs->STORAGE_BLOCK_SIZE + 1;
  uchar * file_addr = get_file_addr_from_FCB_idx(fs, FCB);
  // storage block
  clean_file(file_addr, cur_num_of_block, fs->STORAGE_BLOCK_SIZE);
  // super block
  clean_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, block_addr, cur_num_of_block);
  // FCB
  clean_FCB_file_name(current_child_FCB);
  set_FCB_create_time(current_child_FCB, 0);
  set_FCB_modify_time(current_child_FCB, 0);
  set_FCB_status(current_child_FCB, 0);
  set_FCB_addr(current_child_FCB, 0);
  set_FCB_size(current_child_FCB, 0);
  set_FCB_Dsize(current_child_FCB, 0);
  set_FCB_isDirect(current_child_FCB, 0);
  #ifdef DEBUG
  printf("[DEBUG] finish clean everything of FCB %d\n", FCB);
  #endif
}









/* Sorting helper function */
__device__ void sort_by_size(struct SortingUnit ** array, int size) {
  if (size == 0 || size == 1) return;
  struct SortingUnit * tmp = (struct SortingUnit *)malloc(sizeof(struct SortingUnit));
  for (int i = 1; i < size; i++) {
    if (array[i]->file_size > array[i-1]->file_size || 
      (array[i]->file_size == array[i-1]->file_size && array[i]->create_time < array[i-1]->create_time)) {
      tmp->FCB_idx = array[i]->FCB_idx;
      tmp->file_size = array[i]->file_size;
      tmp->create_time = array[i]->create_time;
      tmp->modify_time = array[i]->modify_time;
      int j = i;
      while (j > 0 && 
        (tmp->file_size > array[j-1]->file_size || (tmp->file_size == array[j-1]->file_size && tmp->create_time < array[j-1]->create_time))
      ) {
        array[j]->FCB_idx = array[j-1]->FCB_idx;
        array[j]->file_size = array[j-1]->file_size;
        array[j]->create_time = array[j-1]->create_time;
        array[j]->modify_time = array[j-1]->modify_time;
        j--;
      }
      array[j]->FCB_idx = tmp->FCB_idx;
      array[j]->file_size = tmp->file_size;
      array[j]->create_time = tmp->create_time;
      array[j]->modify_time = tmp->modify_time;
    }
  }
  free(tmp);
}

__device__ void sort_by_time(struct SortingUnit ** array, int size) {
  #ifdef DEBUG
  printf("begin insertion sort of time\n");
  #endif
  if (size == 0 || size == 1)  return;
  struct SortingUnit * tmp = (struct SortingUnit *)malloc(sizeof(struct SortingUnit));
  for (int i = 1; i < size; i++) {
    if (array[i]->modify_time > array[i-1]->modify_time) {
      tmp->FCB_idx = array[i]->FCB_idx;
      tmp->file_size = array[i]->file_size;
      tmp->create_time = array[i]->create_time;
      tmp->modify_time = array[i]->modify_time;
      int j = i;
      while (j > 0 && tmp->modify_time > array[j-1]->modify_time) {
        array[j]->FCB_idx = array[j-1]->FCB_idx;
        array[j]->file_size = array[j-1]->file_size;
        array[j]->create_time = array[j-1]->create_time;
        array[j]->modify_time = array[j-1]->modify_time;
        j--;
      }
      array[j]->FCB_idx = tmp->FCB_idx;
      array[j]->file_size = tmp->file_size;
      array[j]->create_time = tmp->create_time;
      array[j]->modify_time = tmp->modify_time;
    }
  }
  #ifdef DEBUG
  printf("end insertion sort of time\n");
  #endif
  free(tmp);
}





/* Super Block manipulation helper */

// find a group of blocks that can have such size
__device__ u32 get_from_super_block(uchar * super_block_base, u32 SUPERBLOCK_SIZE, u32 block_num) {
  int count = 0, global = -1, start = -1;
  for (int i = 0; i < SUPERBLOCK_SIZE; ++i) {
    uchar cur = super_block_base[i];
    for (int j = 0; j < 8; ++j) {
      global++;
      if ((cur & 0x80) == 0) {
        if (start == -1) start = global;
        count++;
        if (count == block_num) return start;
      }
      else if ((cur & 0x80) != 0) {
        count = 0;
        start = -1;
      }
      cur = cur << 1;
    }
  }
  return start;
}

__device__ void set_from_super_block(uchar * super_block_base, u32 SUPERBLOCK_SIZE, u32 block_addr, u32 block_num){
  int global = -1;
  for (int i = 0; i < SUPERBLOCK_SIZE; ++i) {
    for (int j = 7; j >= 0; --j) {
      global++;
      if (global >= block_addr + block_num) return;
      else if (global >= block_addr && global < block_addr+block_num) super_block_base[i] = super_block_base[i] | (1 << j);
    }
  }
}

__device__ void clean_from_super_block(uchar * super_block_base, u32 SUPERBLOCK_SIZE, int block_start, u32 block_num) {
  int global = -1;
  for (int i = 0; i < SUPERBLOCK_SIZE; ++i) {
    for (int j = 7; j >= 0; --j) {
      global++;
      if (global >= block_start + block_num) return;
      else if (global >= block_start && global < block_start+block_num) super_block_base[i] &= ~(1 << j);
    }
  }
}






/* Global compaction helper */

__device__ void compact(FileSystem * fs) {
  int start = -1, end = -1, global = -1; // the start and end of '0'
  int used = 0, count = 0; // number of used blocks
  int finished = 0;
  int first = -1, last = -1; // the first and last '1'
  for (int i = 0; i < fs->SUPERBLOCK_SIZE; ++i) {
    uchar cur = *(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS + i);
    for (int j = 0; j < 8; ++j) {
      global++;
      if ((cur & 0x80) == 0) {
        if (start == -1) start = global;
        if (end < global && finished == 0) end = global;
      }
      else if ((cur & 0x80) != 0) {
        if (first == -1) first = global;
        if (last < global) last = global;
        if (start != -1) {
          finished = 1;
        }
        ++used;
      }
      cur = cur << 1;
    }
  }
  if (used == last-first+1 && first == 0) {
    #ifdef DEBUG
    printf("[DEBUG] no need to compact\n");
    #endif
    return;
  }
  clean_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, end + 1, last - end);
  set_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, start, last - end);

  count = end - start + 1;
  #ifdef DEBUG
  printf("[DEBUG]compact start: %d; end: %d, size: %d, used size: %d, first %d, last: %d \n", start, end, count, used, first, last);
  #endif

  // move all the FCB block address forward [count] units
  for (int i = 0; i < fs->FCB_ENTRIES; ++i) {
    uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * i;
    if (get_FCB_status(current_FCB) == 0) continue;
    u32 block_addr = get_FCB_addr(current_FCB);
    u32 file_size = get_FCB_size(current_FCB);

    #ifdef DEBUG
    int prev_addr = get_FCB_addr(current_FCB);
    #endif

    if (block_addr < start) continue;
    else set_FCB_addr(current_FCB, get_FCB_addr(current_FCB)-count);

    #ifdef DEBUG
    printf("[compact] Move FCB %d forward from %d to %d\n", i, prev_addr, get_FCB_addr(current_FCB));
    #endif
  }

  // move all storage block after empty forward [count] units
  uchar * move_buf = (uchar *)malloc(sizeof(uchar) * (last - end) * fs->STORAGE_BLOCK_SIZE);
  uchar * src_start, * dest_start;
  src_start = fs->volume + fs->FILE_BASE_ADDRESS + (end + 1) * fs->STORAGE_BLOCK_SIZE;
  //src_end = fs->volume + fs->FILE_BASE_ADDRESS + last * fs->BLOCK_SIZE;
  dest_start = fs->volume + fs->FILE_BASE_ADDRESS + start * fs->STORAGE_BLOCK_SIZE;

  memcpy(dest_start, src_start, sizeof(uchar) * fs->STORAGE_BLOCK_SIZE * (last-end));

  #ifdef DEBUG
  printf("move %d blocks from BLOCK %d to BLOCK %d\n", last-end, end+1, start);
  #endif

  return;
}





/* File system API */

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS,
              int SUPER_BLOCK_BASE_ADDRESS, int FCB_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  fs->SUPER_BLOCK_BASE_ADDRESS = SUPER_BLOCK_BASE_ADDRESS;
  fs->FCB_BASE_ADDRESS = FCB_BASE_ADDRESS;
  fs->PWD_FCB_idx = 0;
  // initialize the FCB for root direct
  set_FCB_file_name(fs->volume + fs->FCB_BASE_ADDRESS + 0 * fs->FCB_SIZE, "/");
  set_FCB_create_time(fs->volume + fs->FCB_BASE_ADDRESS + 0 * fs->FCB_SIZE, gtime);
  set_FCB_modify_time(fs->volume + fs->FCB_BASE_ADDRESS + 0 * fs->FCB_SIZE, gtime);
  ++gtime;
  set_FCB_isDirect(fs->volume + fs->FCB_BASE_ADDRESS + 0 * fs->FCB_SIZE, 1);
  set_FCB_Dsize(fs->volume + fs->FCB_BASE_ADDRESS + 0 * fs->FCB_SIZE, 0);
  set_FCB_status(fs->volume + fs->FCB_BASE_ADDRESS + 0 * fs->FCB_SIZE, 1);
  set_FCB_addr(fs->volume + fs->FCB_BASE_ADDRESS + 0 * fs->FCB_SIZE, 0);
  set_FCB_size(fs->volume + fs->FCB_BASE_ADDRESS + 0 * fs->FCB_SIZE, DIRECT_SIZE);
  // initialize the superblock for root direct
  set_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, 0, DIRECT_SIZE/fs->STORAGE_BLOCK_SIZE);
  // initialize the storage block for root direct
  set_direct_parent(fs->volume+fs->FILE_BASE_ADDRESS, -1);
  set_direct_child_num(fs->volume+fs->FILE_BASE_ADDRESS, 0);
  #ifdef DEBUG
  printf("[DEBUG]: pwd is %d \n", fs->PWD_FCB_idx);
  #endif
}


// return the volume offset i to the FCB, the address should be (fs->volume + fs->FCB_BASE_ADDRESS + i * fs->FCB_SIZE)
__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  #ifdef DEBUG
  printf("\n");
  if (op == G_READ)
    printf("[DEBUG] %d: open file %s in read mode %d\n", gtime, s, op);
  else if (op == G_WRITE)
    printf("[DEBUG] %d open file %s in write mode %d\n", gtime, s, op);
  #endif

  // check all the children of PWD_FCB to match the file name s, if found, return the FCB idx
  int res = find_child_under_PWD(fs, s);
  #ifdef DEBUG
  printf("[DEBUG] Finish find child %d under %d\n", res, fs->PWD_FCB_idx);
  #endif
  if (res != -1 && get_FCB_isDirect(get_FCB_addr_from_FCB_idx(fs, res)) == true) {
    printf("Invalid open to a diretocy.\n");
    return -1;
  }
  else if (res != -1 && get_FCB_isDirect(get_FCB_addr_from_FCB_idx(fs, res)) == false) return res;

  if (res == -1 && op == G_READ) {
    printf("File does not exsit. Error!\n");
    return -1;
  }

  // the target not found under the PWD, iterate to find a FCB to fit in, and update the PWD blocks
  int FCB_entry_to_fill = get_empty_FCB(fs);
  
  #ifdef DEBUG
  printf("[DEBUG] to fill the FCB entry: %d\n", FCB_entry_to_fill);
  #endif
  if (FCB_entry_to_fill == -1) return -1;
  
  int file_name_length = 0, i = 0;
  while (s[i] != '\0') {
    file_name_length++;
    ++i;
  }
  file_name_length++;
  #ifdef DEBUG
  printf("[DEBUG] file name %s has length: %d\n", s, file_name_length);
  #endif
  ++gtime;
  uchar * FCB_to_fill = get_FCB_addr_from_FCB_idx(fs, FCB_entry_to_fill);
  set_FCB_status(FCB_to_fill, 1);
  set_FCB_isDirect(FCB_to_fill, 0);
  set_FCB_file_name(FCB_to_fill, s);
  set_FCB_size(FCB_to_fill, 0);
  set_FCB_create_time(FCB_to_fill, gtime);

  ++gtime;

  // add child record in parent (pwd)
  int PWD_FCB_idx = fs->PWD_FCB_idx;
  uchar * pwd_FCB = get_FCB_addr_from_FCB_idx(fs, PWD_FCB_idx);
  set_FCB_modify_time(pwd_FCB, gtime);
  set_FCB_Dsize(pwd_FCB, get_FCB_Dsize(pwd_FCB) + file_name_length);

  uchar * file_addr = get_file_addr_from_FCB_idx(fs, PWD_FCB_idx);
  set_direct_child_num(file_addr, read_direct_child_num(file_addr) + 1);
  write_direct_ithChld_idx(file_addr, read_direct_child_num(file_addr) - 1, FCB_entry_to_fill);
  

  #ifdef DEBUG
  printf("[DEBUG] File %s not exist, give the FCB %d \n", s, FCB_entry_to_fill);
  printf("[DEBUG] PWD %d actual size as %d, fake size as %d \n", PWD_FCB_idx, get_FCB_size(pwd_FCB), get_FCB_Dsize(pwd_FCB));
  printf("[DEBUG] add child %d as the %d th child of PWD %d \n", FCB_entry_to_fill, read_direct_child_num(file_addr), PWD_FCB_idx);
  #endif

  return FCB_entry_to_fill;

}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	uchar * current_FCB = get_FCB_addr_from_FCB_idx(fs, fp);
  if (get_FCB_status(current_FCB)==false || get_FCB_isDirect(current_FCB) == true) {
    #ifdef DEBUG
    printf("[DEBUG] %d: READ File pointer: %d; valid: %d, isDirect: %d\n", gtime, fp, get_FCB_status(current_FCB), get_FCB_isDirect(current_FCB));
    #endif

    printf("Ivalid read to a directory or an unexisted file!\n");
    return;
  }

  u32 file_size = get_FCB_size(current_FCB);

  #ifdef DEBUG
  printf("\n");
  u32 block_addr = get_FCB_addr(current_FCB);
  printf("[DEBUG] %d: READ FCB INDEX: %d; size: %d, block addr: %d\n", gtime, fp, size, block_addr);
  #endif

  uchar * file_addr = get_file_addr_from_FCB_idx(fs, fp);
  size = size <= file_size? size : file_size;
  read_file(file_addr, output, size, fs->STORAGE_BLOCK_SIZE);
}


// write size == 0? free all its block
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
  #ifdef DEBUG
  printf("\n");
  printf("[DEBUG] %d: write file; FCB INDEX %d; size: %d\n", gtime, fp, size);
  #endif

	uchar * current_FCB = get_FCB_addr_from_FCB_idx(fs, fp);

  if (get_FCB_status(current_FCB)==false || get_FCB_isDirect(current_FCB)==true) {
    printf("Ivalid write to a directory or an unexisted file!\n");
    return 0;
  }

  uchar * file_addr;
  u32 block_addr;
  u32 file_size = get_FCB_size(current_FCB);

  int cur_num_of_block; 

  if (size == 0 && file_size == 0) {

    #ifdef DEBUG
    printf("[DEBUG] Nothing to write in an empty file.\n");
    #endif

    ++gtime;
    set_FCB_modify_time(current_FCB, gtime);
    return 0;
  }
  // nothing to write in an non-empty file
  else if (size == 0 && file_size != 0) {

    block_addr = get_FCB_addr(current_FCB);
    // file_addr = fs->volume + fs->FILE_BASE_ADDRESS + block_addr * fs->STORAGE_BLOCK_SIZE;

    file_addr = get_file_addr_from_FCB_idx(fs, fp);
    cur_num_of_block = (file_size-1) / fs->STORAGE_BLOCK_SIZE + 1;
    clean_file(file_addr, cur_num_of_block, fs->STORAGE_BLOCK_SIZE);
    clean_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, block_addr, cur_num_of_block);
    ++gtime;
    set_FCB_modify_time(current_FCB, gtime);
    set_FCB_size(current_FCB, size);
    compact(fs);
    return 0;
  }

  int required_num_of_block = (size-1) / fs->STORAGE_BLOCK_SIZE + 1;

  // the current file has no block allocated
  if (size != 0 && file_size == 0) {
    block_addr = get_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, required_num_of_block);
    set_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, block_addr, required_num_of_block);
    set_FCB_addr(current_FCB, block_addr);
    file_addr = fs->volume + fs->FILE_BASE_ADDRESS + block_addr * fs->STORAGE_BLOCK_SIZE;
    #ifdef DEBUG
    printf("[DEBUG] No allocated block yet for FCB INDEX %d, get %d blocks from block addr %d \n", fp, required_num_of_block, block_addr);
    #endif
  }

  // the current has allocated block
  else if (size != 0 && file_size != 0) {
    block_addr = get_FCB_addr(current_FCB);

    #ifdef DEBUG
    int tmp_addr = block_addr;
    #endif

    file_addr = fs->volume + fs->FILE_BASE_ADDRESS + block_addr * fs->STORAGE_BLOCK_SIZE;
    cur_num_of_block = (file_size-1) / fs->STORAGE_BLOCK_SIZE + 1;
    clean_file(file_addr, cur_num_of_block, fs->STORAGE_BLOCK_SIZE);

    // the allocated block is not the perfect one
    if (cur_num_of_block != required_num_of_block) {
      clean_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, block_addr, cur_num_of_block);
      block_addr = get_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, required_num_of_block);
      set_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, block_addr, required_num_of_block);
      set_FCB_addr(current_FCB, block_addr);
      file_addr = fs->volume + fs->FILE_BASE_ADDRESS + block_addr * fs->STORAGE_BLOCK_SIZE;
      compact(fs);
    }
    #ifdef DEBUG
    printf("[DEBUG] Allocated block for FCB INDEX %d with %d blocks from %d previously, get %d blocks from block addr %d \n", fp, cur_num_of_block, tmp_addr ,required_num_of_block, block_addr);
    #endif
  }
  ++gtime;
  set_FCB_modify_time(current_FCB, gtime);
  write_file(file_addr, input, size, fs->STORAGE_BLOCK_SIZE);
  set_FCB_size(current_FCB, size);

  #ifdef DEBUG
  printf("[DEBUG] size: %d, block addr: %d, create: %d, modify: %d\n", get_FCB_size(current_FCB), get_FCB_addr(current_FCB), get_FCB_create_time(current_FCB), get_FCB_modify_time(current_FCB));
  #endif

  return 0;
}



__device__ void fs_gsys(FileSystem *fs, int op) 
{
  #ifdef DEBUG
  printf("\n");
  if (op == LS_D)
    printf("[DEBUG] OP: %d by list time\n", op);
  else if (op == LS_S)
    printf("[DEBUG] OP: %d by list size\n", op);
  else if (op == CD_P)
    printf("[DEBUG] OP: %d to CD TO PARENT\n", op);
  else if (op == PWD)
    printf("[DEBUG] OP: %d to get PWD\n", op);
  #endif

  if (op == PWD) {
    print_FCB_path_name(fs, fs->PWD_FCB_idx);
    return;
  }

  if (op == CD_P) {
    u32 FCB_idx = fs->PWD_FCB_idx;
    uchar * current_FCB = get_FCB_addr_from_FCB_idx(fs, FCB_idx);
    uchar * file_addr = get_file_addr_from_FCB_idx(fs, FCB_idx);
    u32 parent_FCB_idx = get_direct_parent(file_addr);

    #ifdef DEBUG
    printf("[DEBUG] %d change pwd from FCB %d to %d\n", gtime, FCB_idx, parent_FCB_idx);
    #endif
    
    fs->PWD_FCB_idx = parent_FCB_idx;
    return;
  }

  // initialize the array to sort
  int count = read_direct_child_num(get_file_addr_from_FCB_idx(fs, fs->PWD_FCB_idx));
  struct SortingUnit ** array = (struct SortingUnit**)malloc(sizeof(struct SortingUnit*) * count);

  for (int i = 0; i < count; i++) {
    int child_idx = get_direct_ithChld_idx(get_file_addr_from_FCB_idx(fs, fs->PWD_FCB_idx), i);
    uchar * current_FCB = get_FCB_addr_from_FCB_idx(fs, child_idx);
    bool isDrect = get_FCB_isDirect(current_FCB);
    array[i] = (struct SortingUnit *)malloc(sizeof(struct SortingUnit));
    array[i]->FCB_idx = child_idx;
    if (isDrect ==  false)
      array[i]->file_size = get_FCB_size(current_FCB);
    else if (isDrect == true)
      array[i]->file_size = get_FCB_Dsize(current_FCB);
    array[i]->create_time = get_FCB_create_time(current_FCB);
    array[i]->modify_time = get_FCB_modify_time(current_FCB);

    #ifdef DEBUG
    // printf("after copy into sorting array: idx: %d %d; size: %d %d; create time: %d %d, modify time: %d %d\n", i, array[count]->FCB_idx, get_FCB_size(current_FCB), array[count]->file_size, get_FCB_create_time(current_FCB),
    // array[count]->create_time, get_FCB_modify_time(current_FCB), array[count]->modify_time);
    // printf("copy finish\n");
    uchar * name_buf = (uchar *)malloc(sizeof(uchar) * NAME_MAX_LENGTH);
    get_FCB_file_name(get_FCB_addr_from_FCB_idx(fs, child_idx), name_buf);
    printf("[DEBUG] %d FCB %s size %d mtime %d ctime %d \n", i, name_buf, array[i]->file_size, array[i]->modify_time, array[i]->create_time);
    #endif
  }
  
  
  // by modify time
	if (op == LS_D) {
    sort_by_time(array, count);
    printf("===sort by modified time===\n");
    for (int i = 0; i < count; ++i) {
      uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * array[i]->FCB_idx;
      free(array[i]);
      uchar * name_buf = (uchar *)malloc(sizeof(char) * NAME_MAX_LENGTH);
      get_FCB_file_name(current_FCB, name_buf);
      if (get_FCB_isDirect(current_FCB) == false)
        printf("%s\n", name_buf);
      else if (get_FCB_isDirect(current_FCB) == true)
      printf("%s d\n", name_buf);
    }
  }

  // by size
  else if (op == LS_S) {
    sort_by_size(array, count);
    printf("===sort by file size===\n");
    for (int i = 0; i < count; ++i) {
      uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * array[i]->FCB_idx;
      free(array[i]);
      uchar * name_buf = (uchar *)malloc(sizeof(char) * NAME_MAX_LENGTH);
      get_FCB_file_name(current_FCB, name_buf);

      if (get_FCB_isDirect(current_FCB) == false) {
        u32 file_size = get_FCB_size(current_FCB);
        printf("%s %d\n", name_buf, file_size);
      }
        
      else if (get_FCB_isDirect(current_FCB) == true){
        u32 file_size = get_FCB_Dsize(current_FCB);
        printf("%s %d d\n", name_buf, file_size);
      }
    }
  }
  free(array);
}


__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  ++gtime;
  #ifdef DEBUG
  printf("\n");
  if (op == RM)
    printf("[DEBUG] %d RM op: %s\n", gtime, s);
  else if (op == MKDIR)
    printf("[DEBUG] %d MKDIR op: %s\n", gtime, s);
  else if (op == CD)
    printf("[DEBUG] %d CD op: %s\n", gtime, s);
  else if (op == RM_RF)
    printf("[DEBUG] %d RM_RF op: %s\n", gtime, s);
  #endif

  if (op!=RM && op!=MKDIR && op!=CD && op!=RM_RF)
    return;
  
  if (op == RM) {
    int res = find_child_under_PWD(fs, s);

    if (res == -1) {
      printf("File not exist under this directory. RM failure.\n");
      return;
    }

    if (get_FCB_isDirect(get_FCB_addr_from_FCB_idx(fs, res)) == true) {
      printf("Invalid operation to a diretocy.\n");
      return;
    }

    else if (res != -1) {
      delete_child_record_from_parent(fs, res, fs->PWD_FCB_idx);
      clean_all_from_FCB_idx(fs, res);
      compact(fs);
      return;
    }
  }

  else if (op == MKDIR) {
    int res = find_child_under_PWD(fs, s);
    if (res != -1) {
      printf("Already exist. MKDIR fail\n");
      return;
    }
    int fcb_idx = get_empty_FCB(fs);
    uchar * current_FCB = get_FCB_addr_from_FCB_idx(fs, fcb_idx);
    // set super block
    int block_addr = get_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, DIRECT_SIZE/fs->STORAGE_BLOCK_SIZE);
    set_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, block_addr, DIRECT_SIZE/fs->STORAGE_BLOCK_SIZE);
    // set storage block
    uchar * file_addr = fs->volume + fs->FILE_BASE_ADDRESS + block_addr * fs->STORAGE_BLOCK_SIZE;
    set_direct_child_num(file_addr, 0);
    set_direct_parent(file_addr, fs->PWD_FCB_idx);
    // set FCB
    set_FCB_file_name(current_FCB, s);
    set_FCB_create_time(current_FCB, gtime);
    set_FCB_modify_time(current_FCB, gtime);
    set_FCB_isDirect(current_FCB, 1);
    set_FCB_Dsize(current_FCB, 0);
    set_FCB_size(current_FCB, DIRECT_SIZE);
    set_FCB_addr(current_FCB, block_addr);
    set_FCB_status(current_FCB, 1);

    int file_name_length = 0, k = 0;
    while (s[k] != '\0') {
      file_name_length++;
      ++k;
    }
    file_name_length++;

    // add child record in parent (pwd)
    int PWD_FCB_idx = fs->PWD_FCB_idx;
    uchar * pwd_FCB = get_FCB_addr_from_FCB_idx(fs, PWD_FCB_idx);
    set_FCB_modify_time(pwd_FCB, gtime);
    set_FCB_Dsize(pwd_FCB, get_FCB_Dsize(pwd_FCB) + file_name_length);

    uchar * file_addr_p = get_file_addr_from_FCB_idx(fs, PWD_FCB_idx);
    set_direct_child_num(file_addr_p, read_direct_child_num(file_addr_p) + 1);
    write_direct_ithChld_idx(file_addr_p, read_direct_child_num(file_addr_p) - 1, fcb_idx);
    return;
  }

  else if (op == CD) {
    int res = find_child_under_PWD(fs, s);
    if (res == -1) {
      printf("Does not exist. Error\n");
      return;
    }
    fs->PWD_FCB_idx = res;
    return;
  }

  else if (op == RM_RF) {
    int res = find_child_under_PWD(fs, s);
    if (get_FCB_isDirect(fs->volume + fs->FCB_BASE_ADDRESS + res * fs->FCB_SIZE) == false) {
      printf("Invalid open to a file.\n");
      return;
    }
    perform_rmrf(fs, res, fs->PWD_FCB_idx);
    return;
  }
}
