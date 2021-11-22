#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NAME_MAX_LENGTH 20            // bytes

#define STATUS_POSITION 31            
#define SIZE_POSITION 30              
#define ADDR_POSITION 28              // block address, 16 bit
#define MODIFY_TIME_POSITION 24
#define CREATE_TIME_POSITION 20

#define DEBUGx


__device__ __managed__ u32 gtime = 0;


/* FCB helper function: use the absolute address of the FCB entry */

// return 1 if a valid FCB, else return 0
__device__ bool get_FCB_status(uchar * FCB_start) {
  return (FCB_start[ADDR_POSITION] & 0b10000000) >> 7;
}

__device__ void set_FCB_status(uchar * FCB_start, int status) {
  FCB_start[ADDR_POSITION] = (FCB_start[ADDR_POSITION] & 0b01111111) | (status << 7);
}


// return the size (bytes) in current FCB entry
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
  u32 c4 = (u32)FCB_start[CREATE_TIME_POSITION+3];
  u32 time = (c1<<24) | (c2<<16) | (c3<<8) | (c4<<0);
  return time;
}

__device__ void set_FCB_create_time(uchar *FCB_start, u32 time) {
  uchar c1 = (time & 0xff000000) >> 24;
  uchar c2 = (time & 0xff0000) >> 16;
  uchar c3 = (time & 0xff00) >> 8;
  uchar c4 = (time & 0xff) >> 0;
  FCB_start[CREATE_TIME_POSITION] = c1;
  FCB_start[CREATE_TIME_POSITION+1] = c2;
  FCB_start[CREATE_TIME_POSITION+2] = c3;
  FCB_start[CREATE_TIME_POSITION+3] = c4;
}


// return the MODIFY time in current FCB entry
__device__ u32 get_FCB_modify_time(uchar *FCB_start) {
  u32 c1 = (u32)FCB_start[MODIFY_TIME_POSITION];
  u32 c2 = (u32)FCB_start[MODIFY_TIME_POSITION+1];
  u32 c3 = (u32)FCB_start[MODIFY_TIME_POSITION+2];
  u32 c4 = (u32)FCB_start[MODIFY_TIME_POSITION+3];
  u32 time = (c1<<24) | (c2<<16) | (c3<<8) | (c4<<0);
  return time;
}

__device__ void set_FCB_modify_time(uchar *FCB_start, u32 time) {
  uchar c1 = (time & 0xff000000) >> 24;
  uchar c2 = (time & 0xff0000) >> 16;
  uchar c3 = (time & 0xff00) >> 8;
  uchar c4 = (time & 0xff) >> 0;
  FCB_start[MODIFY_TIME_POSITION] = c1;
  FCB_start[MODIFY_TIME_POSITION+1] = c2;
  FCB_start[MODIFY_TIME_POSITION+2] = c3;
  FCB_start[MODIFY_TIME_POSITION+3] = c4;
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
      if (count == size - 1) return;
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
      if (count == size - 1) return;
    }
  }
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

}


// return the volume offset i to the FCB, the address should be (fs->volume + fs->FCB_BASE_ADDRESS + i * fs->FCB_SIZE)
__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  #ifdef DEBUG
  printf("\n");
  printf("open file %s in mode %d\n", s, op);
  #endif
  // check all the FCB to match the file name s, FCB sized of 32 bytes
  uchar * current_FCB;
  bool isValid = false, found = false;
  int FCB_entry_to_fill = -1;
  u32 i = 0;
  for (; i < fs->FCB_ENTRIES; ++i) 
  {
    current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * i;
    isValid = get_FCB_status(current_FCB);
    if (isValid == true) {
      found = match_FCB_file_name(current_FCB, s);
      if (found == true) 
        break;
    }
    else if (isValid == false) {
      if (FCB_entry_to_fill == -1) FCB_entry_to_fill = i;
    }
  }

  if (found == true) {
    #ifdef DEBUG
    printf("\n");
    printf("File %s exist, with the FCB %d \n", s, i);
    #endif
    return i;
  }
  else if (found == false) {
    // cannot get the file to read: error
    if (op == G_READ)
      printf("Invalid write to an unexisted file %s!\n", s);
    else if (op == G_WRITE) {
      ++gtime;
      set_FCB_status(fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * FCB_entry_to_fill, 1);
      set_FCB_file_name(fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * FCB_entry_to_fill, s);
      set_FCB_size(fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * FCB_entry_to_fill, 0);
      set_FCB_create_time(fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * FCB_entry_to_fill, gtime);
      #ifdef DEBUG
      printf("File %s not exist, give the FCB %d \n", s, FCB_entry_to_fill);
      #endif
      return FCB_entry_to_fill;
    }
  }
  return 0;
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fp * fs->FCB_SIZE;
  u32 file_size = get_FCB_size(current_FCB);
  u32 block_addr = get_FCB_addr(current_FCB);

  #ifdef DEBUG
  printf("READ File pointer: %d; size: %d, block addr: %d\n", fp, size, block_addr);
  #endif

  uchar * file_addr = fs->volume + fs->FILE_BASE_ADDRESS + block_addr * fs->STORAGE_BLOCK_SIZE;
  size = size <= file_size? size : file_size;
  read_file(file_addr, output, size, fs->STORAGE_BLOCK_SIZE);
}


// write size == 0?
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
  #ifdef DEBUG
  printf("\n");
  printf("write file; FCB INDEX %d; size: %d\n", fp, size);
  #endif
	uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fp * fs->FCB_SIZE;
  uchar * file_addr;
  u32 block_addr;
  u32 file_size = get_FCB_size(current_FCB);

  int cur_num_of_block; 
  int required_num_of_block = (size-1) / fs->STORAGE_BLOCK_SIZE + 1;

  // the current file has no block allocated
  if (file_size == 0) {
    block_addr = get_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, required_num_of_block);
    set_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, block_addr, required_num_of_block);
    set_FCB_addr(current_FCB, block_addr);
    file_addr = fs->volume + fs->FILE_BASE_ADDRESS + block_addr * fs->STORAGE_BLOCK_SIZE;
    #ifdef DEBUG
    printf("No allocated block yet for FCB INDEX %d, get %d blocks from block addr %d \n", fp, required_num_of_block, block_addr);
    #endif
  }

  // the current has allocated block
  else if (file_size != 0) {
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
    }
    #ifdef DEBUG
    printf("Allocated block for FCB INDEX %d with %d blocks from %d previously, get %d blocks from block addr %d \n", fp, cur_num_of_block, tmp_addr ,required_num_of_block, block_addr);
    #endif
  }
  ++gtime;
  set_FCB_modify_time(current_FCB, gtime);
  write_file(file_addr, input, size, fs->STORAGE_BLOCK_SIZE);
  set_FCB_size(current_FCB, size);
  #ifdef DEBUG
  printf("size: %d, block addr: %d, create: %d, modify: %d\n", get_FCB_size(current_FCB), get_FCB_addr(current_FCB), get_FCB_create_time(current_FCB), get_FCB_modify_time(current_FCB));
  #endif

  return 0;
}



__device__ void fs_gsys(FileSystem *fs, int op)
{
  #ifdef DEBUG
  printf("\n");
  printf("list file\n");
  #endif
  // initialize the array to sort
  struct SortingUnit ** array = (struct SortingUnit**)malloc(sizeof(struct SortingUnit*) * fs->FCB_ENTRIES);

  int count = 0;
  for (int i = 0; i < fs->FCB_ENTRIES; ++i) {
    uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * i;
    bool isValid = get_FCB_status(current_FCB);
    #ifdef DEBUG
    //printf("block: %d valid: %d\n", i, isValid);
    #endif
    if (isValid) {
      array[count] = (struct SortingUnit *)malloc(sizeof(struct SortingUnit));
      array[count]->FCB_idx = i;
      array[count]->file_size = get_FCB_size(current_FCB);
      array[count]->create_time = get_FCB_create_time(current_FCB);
      array[count]->modify_time = get_FCB_modify_time(current_FCB);
      #ifdef DEBUG
      printf("after copy into sorting array: idx: %d %d; size: %d %d; create time: %d %d, modify time: %d %d\n", i, array[count]->FCB_idx, get_FCB_size(current_FCB), array[count]->file_size, get_FCB_create_time(current_FCB),
      array[count]->create_time, get_FCB_modify_time(current_FCB), array[count]->modify_time);
      printf("copy finish\n");
      #endif
      count++;
    }
  }
  #ifdef DEBUG
  if (op == LS_D)
    printf("OP: %d by time\n", op);
  else if (op == LS_S)
    printf("OP: %d by size\n", op);
  #endif
  // by modify time
	if (op == LS_D) {
    sort_by_time(array, count);
    printf("== sort by modify time == \n");
    for (int i = 0; i < count; ++i) {
      uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * array[i]->FCB_idx;
      free(array[i]);
      uchar * name_buf = (uchar *)malloc(sizeof(char) * NAME_MAX_LENGTH);
      get_FCB_file_name(current_FCB, name_buf);
      printf("%s\n", name_buf);
    }
  }

  // by size
  else if (op == LS_S) {
    sort_by_size(array, count);
    printf("== sort by file size == \n");
    for (int i = 0; i < count; ++i) {
      uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * array[i]->FCB_idx;
      free(array[i]);
      uchar * name_buf = (uchar *)malloc(sizeof(char) * NAME_MAX_LENGTH);
      get_FCB_file_name(current_FCB, name_buf);
      u32 file_size = get_FCB_size(current_FCB); 
      printf("%s %d\n", name_buf, file_size);
    }
  }
  free(array);
}


__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  #ifdef DEBUG
  printf("\n");
  printf("remove file op: %s\n", s);
  #endif
  if (op != RM) return;

  uchar * current_FCB;
  bool isValid = false, found = false;
  for (u32 i = 0; i < fs->FCB_ENTRIES; ++i) 
  {
    current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * i;
    isValid = get_FCB_status(current_FCB);
    if (isValid == true) {
      found = match_FCB_file_name(current_FCB, s);
      if (found == true) 
        break;
    }
  }

  if (found == true) {
    u32 block_addr = get_FCB_addr(current_FCB);
    u32 file_size = get_FCB_size(current_FCB);
    int cur_num_of_block = (file_size-1) / fs->STORAGE_BLOCK_SIZE + 1;
    uchar * file_addr = fs->volume + fs->FILE_BASE_ADDRESS + block_addr * fs->STORAGE_BLOCK_SIZE;
    clean_file(file_addr, cur_num_of_block, fs->STORAGE_BLOCK_SIZE);
    clean_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, block_addr, cur_num_of_block);

    clean_FCB_file_name(current_FCB);
    set_FCB_create_time(current_FCB, 0);
    set_FCB_modify_time(current_FCB, 0);
    set_FCB_status(current_FCB, 0);
    set_FCB_addr(current_FCB, 0);
    set_FCB_size(current_FCB, 0);
  }
  else if (found == false) 
    printf("REMOVE FAILURE: file %s does not exite. \n", s);

  compact(fs);
}
