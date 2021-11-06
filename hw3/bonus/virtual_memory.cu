#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


#define PAGE_SIZE (1 << 5)

enum direction {
  BufferToStorage,
  StorageToBuffer
};

// helper function: for page swapping
__device__ void page_swap(VirtualMemory *vm, int dest, int src, int size, direction drt) {
  int i;
  if (drt == BufferToStorage) {
    for (i = 0; i < size; ++i) 
      vm->storage[dest+i] = vm->buffer[src+i];
  }
  else {
    for (i = 0; i < size; ++i)
      vm->buffer[dest+i] = vm->storage[src+i];
  }
}

// helper function: find LRU target
__device__ int get_LRU_target(VirtualMemory *vm) {
  int count = 0, max_count = 0;
  int res = 0;   // physical page number for swap
  for (int i = 0; i < vm->PAGE_ENTRIES; ++i) {
    count = (vm->invert_page_table[i] & 0x0ffff000) >> 12;
    if (count > max_count) {
      max_count = count;
      res = i;
    }
    count++;
    vm->invert_page_table[i] = (vm->invert_page_table[i] & 0xf0000fff) | (count<<12);
  }
  return res;
}

// for a page table entry:
// |valid bit 1 |dirty bit 1 | tid 2 |count 16 |virtual page number 12|
// single thread: tid == 00

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr, int pid) {
  /* Complate vm_read function to read single element from data buffer */
  bool page_hit = false;
  bool is_full = true;
  
  int offset = addr & 0x0000001f;
  int virtual_page_number = addr >> 5;
  int page_number_in_PT, pid_in_PT;
  int physical_page_number = 0;

  uchar res;

  for (int i = 0; i < vm->PAGE_ENTRIES; ++i)
  {
    // the current page table entry is invalid
    if (((vm->invert_page_table[i]) & 0x80000000) != 0)
    {
      if (is_full)
      {
        is_full = false;
        physical_page_number = i;
      }
      continue;
    }
    page_number_in_PT = (vm->invert_page_table[i]) & 0xfff;
    pid_in_PT = (vm->invert_page_table[i] & 0x30000000) >> 28;
    
    // page hit: get the physical page number
    if (page_number_in_PT == virtual_page_number && pid_in_PT == pid)
    {
      page_hit = true;
      physical_page_number = i;
      break;
    }
  }

  // page fault
  if (page_hit == false)
  {
    (*(vm->pagefault_num_ptr))++;
    // the physical memory is not full: insert a page, updata the page table
    if (is_full == false)
    {
      page_swap(vm, physical_page_number<<5, virtual_page_number<<5, vm->PAGESIZE, StorageToBuffer);       // insert in
      vm->invert_page_table[physical_page_number] = (0 << 31) | (pid << 28) | (0 << 12) | virtual_page_number;
      res = vm->buffer[(physical_page_number<<5) |  offset];
    }
    // the physical memory is full --> page swap, update the page table
    else
    {
      physical_page_number = get_LRU_target(vm);
      // dirty page: write back
      if ((vm->invert_page_table[physical_page_number] & 0x40000000) != 0)
        page_swap(vm, ((vm->invert_page_table[physical_page_number])&0xfff)<<5, physical_page_number<<5, vm->PAGESIZE, BufferToStorage);   // swap out
      page_swap(vm, physical_page_number<<5, virtual_page_number<<5, vm->PAGESIZE, StorageToBuffer);       // swap in
      vm->invert_page_table[physical_page_number] = (0 << 31) | (pid << 28) | (0 << 12) | virtual_page_number;
      res = vm->buffer[(physical_page_number<<5) | offset];
    }
  }

  // page hit: increase the counter, return the result
  else
  {
    vm->invert_page_table[physical_page_number] = vm->invert_page_table[physical_page_number] & 0xf0000fff;
    res = vm->buffer[(physical_page_number<<5) | offset];
  }

  return res;
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value, int pid) {
  /* Complete vm_write function to write value into data buffer */
  bool page_hit = false;
  bool is_full = true;
  
  int offset = addr & 0x0000001f;
  int virtual_page_number = addr >> 5;
  int page_number_in_PT, pid_in_PT;
  int physical_page_number = 0;
  
  // if (threadIdx.x == 0 || threadIdx.x == 1)
  //   printf("page table start from %x\n", &(vm->invert_page_table[0]));
  for (int i = 0; i < vm->PAGE_ENTRIES; ++i)
  {
    // the current page table entry is invalid (1)
    if (((vm->invert_page_table[i]) & 0x80000000) != 0)
    {
      if (is_full)
      {
        is_full = false;
        physical_page_number = i;
      }
      continue;
    }
    page_number_in_PT = vm->invert_page_table[i] & 0xfff;
    pid_in_PT = (vm->invert_page_table[i] & 0x30000000) >> 28;
    
    // page hit: get the physical page number
    if (page_number_in_PT == virtual_page_number && pid_in_PT == pid)
    {
      page_hit = true;
      physical_page_number = i;
      break;
    }
  }

  // page fault
  if (page_hit == false)
  {
    (*(vm->pagefault_num_ptr))++;
    // the physical memory is not full: insert a page, updata the page table
    if (is_full == false)
    {
      page_swap(vm, physical_page_number<<5, virtual_page_number<<5, vm->PAGESIZE, StorageToBuffer);       // insert in
      vm->invert_page_table[physical_page_number] = (0 << 31) | (1 << 30) | (pid << 28) | (0 << 12) | virtual_page_number;
      vm->buffer[(physical_page_number<<5) |  offset] = value;
    }
    // the physical memory is full --> page swap, update the page table
    else
    {
      physical_page_number = get_LRU_target(vm);
      // dirty page: write back
      if ((vm->invert_page_table[physical_page_number] & 0x40000000) != 0)
        page_swap(vm, ((vm->invert_page_table[physical_page_number])&0xfff)<<5, physical_page_number<<5, vm->PAGESIZE, BufferToStorage);   // swap out
      page_swap(vm, physical_page_number<<5, virtual_page_number<<5, vm->PAGESIZE, StorageToBuffer);       // swap in
      vm->invert_page_table[physical_page_number] = (0 << 31) | (1 << 30) | (pid << 28) | (0 << 12) | virtual_page_number;
      vm->buffer[(physical_page_number<<5) | offset] = value;
    }
  }

  // page hit: increase the counter, return the result
  else
  {
    vm->invert_page_table[physical_page_number] = (0 << 31) | (1 << 30) | (pid << 28) | (0 << 12) | virtual_page_number;
    vm->buffer[(physical_page_number<<5) | offset] = value;
  }
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size, int pid) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  uchar res;
  for (int i = offset; i < offset + input_size; ++i){
    res = vm_read(vm, i, pid);
    results[i] = res;
  }
}

