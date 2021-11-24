# Assignment4 Report



## File Tree

**NOTICE:** the user program.cu is modified, which is different from the provided version. Please add your test code for test.

![image-20211124143003136](report.assets/image-20211124143003136.png)

```shell
.
├── Source
│   ├── data.bin
│   ├── file_system.cu
│   ├── file_system.h
│   ├── main.cu
│   ├── Makefile
│   ├── snapshot.bin
│   └── user_program.cu
├── Bonus
│   ├── data.bin
│   ├── file_system.cu
│   ├── file_system.h
│   ├── main.cu
│   ├── Makefile
│   ├── snapshot.bin
│   └── user_program.cu
└── report.pdf
```

Under **Source** directory:

> data.bin: input data to the virtual memory
>
> main.cu: main file for the program
>
> Makefile: help compile the program
>
> snapshot.bin: data output from the virtual memory
>
> user_program.cu: you can change it for another program reading or writing to the virtual memory
>
> file_system.h: header file for the virtual_memory.cu
>
> file_system.cu: our implementation of read / write operations on the memory



Under **bonus** directory:

> data.bin: input data to the virtual memory
>
> main.cu: main file for the program
>
> Makefile: help compile the program
>
> snapshot.bin: data output from the virtual memory
>
> user_program.cu: you can change it for another program reading or writing to the virtual memory
>
> file_system.h: header file for the virtual_memory.cu
>
> file_system.cu: our implementation of read / write operations on the memory





## 1. Running Environment

For the **homework & bonus** program:

**Operating System**

```shell
cat /etc/redhat-release
```

![image-20211124141455772](report.assets/image-20211124141455772.png)

**CUDA Version**

```shell
nvcc --version
```

![image-20211124141523887](report.assets/image-20211124141523887.png)

```shell
cat /usr/local/cuda/version.txt
```

![image-20211124141546991](report.assets/image-20211124141546991.png)

**GPU Information**

```shell
lspci | grep -i vga
```

![image-20211124141614323](report.assets/image-20211124141614323.png)



**NOTICE**: the machine is located at TC301 room, the $1th$​ machine.





## 2. Execution Steps

#### Basic Task

```shell
cd <DIRCT_TO_PROJ>                        		 						 # come to the project directory
cd Source                                  		  						 # go to the source file
make all                                   	      						 # compile all the files
./test > result.txt                        		  						 # execute the program
make clean                                        						 # clean all the compiled files
diff snapshot_correct.bin snapshot.bin             						 # compare the correct and output snapshot bin
diff <(xxd snapshot_correct.bin)<(xxd snapshot.bin)   		             # compare to view dirty data
diff result_correct.txt result.txt                                       # compare with the correct output
```

#### Bonus Task

```shell
cd <DIRCT_TO_PROJ>                        		 						 # come to the project directory
cd Bonus                               		  						     # go to the source file
make all                                   	      						 # compile all the files
./test > result.txt                        		  						 # execute the program
make clean                                        						 # clean all the compiled files
diff snapshot_correct.bin snapshot.bin             						 # compare the correct and output snapshot bin
diff <(xxd snapshot_correct.bin)<(xxd snapshot.bin)   		             # compare to view dirty data
diff result_correct.txt result.txt                                       # compare with the correct output
```





## 3.Progame Design

### Homework Program

#### Overall Ideas

​	There are 4KB for super block. Each bit represents the status of one block. Totally, it represents 32K storage block.

​	There are 32KB for FCB, each FCB with 32 bytes. Totally it can control 1K files.

​	There are 1024KB for block storage, each block of 32 bytes. Totally there are 32K storage block.

​	When the user **opens** a file, it searches the whole file system, compares all FCB blocks. It the filename match, it returns the index of FCB. Otherwise, if the open is in the write mode, it finds an empty FCB, assign to this file, and return the index of this newly assigned FCB.

​	When the user **reads** to a file, it gets the FCB by the FP, and then gets the storage address from FCB. After that, it reads from the storage address.

​	When the user **writes** to a file, it gets the FCB by the FP, and then gets the storage address from FCB. If the file has no storage block, assign some blocks to it based on the write size and update in the super block. If the current block storage is the same as the needed, it simply cleans the origin content and writes new things in. If the current block storage is the more or less than the needed, it cleans the origin blocks and assign new blocks to it, and then writes to the newly assigned blocks. If there are blocks cleaned, we do file compactions.

​	When the users **remove** a file, it searches the whole file system, compares all FCB blocks. It the filename match, it gets the index of FCB. With this FCB index, it can clean all content of this FCB and its related storage & super block. After that, we do file compactions.

​	When the users perform **LS** operation, it maintain an array outsides the file system, in which FCB index, size, and modify & create time is recorded. Then we perform **insertion sort** based on the sorting standard. After insertion sort, output the array in sorted order.

​	

#### How to arrange the 32 bytes in a FCB?

```cpp
#define STATUS_POSITION 31            
#define SIZE_POSITION 30              
#define ADDR_POSITION 28              // block address, 16 bit
#define MODIFY_TIME_POSITION 24
#define CREATE_TIME_POSITION 20
```

​	 In this single level file system, there is no directory to be created. Therefore, we only records the file information, omitting the directory information. And here is the structure of the FCB:

**0-19 bytes**: file name

**20-23 bytes**: file create time

**24-27 bytes**: file modify time

**28-29 bytes**: the first bit is the **valid** bit, 1 to indicate it as a working FCB. The remained **15** bits store the block address information.

**30-31bytes**: the last **10** bits store the size of file.



#### Which should be the file pointer?

​	There are many choices for the returned file pointer: FCB index, super block index, and storage block index. We choose to return the FCB index, since we can easily get to the storage block address and superblock based on the FCB index.

```cpp
// return the volume offset i to the FCB, the address should be (fs->volume + fs->FCB_BASE_ADDRESS + i * fs->FCB_SIZE)
__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
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
      return FCB_entry_to_fill;
    }
  }
  return 0;
}
```



#### How to do file compaction?

​	We ensure that each time there is **at most one** fragmentation in the storage. To achieve this, we do compactions each time there are blocks freed.

​	In two conditions that there are blocks freed: (1) A file is removed. All its storage blocks freed. (2) The write file size occupies more / less blocks than the original storage block. We free the original blocks and assign new blocks.

​	To do compaction, we check the position of hollow blocks  in the storage. If found, we move all the storage contents forward to fill the hollow blocks, and update the corresponding FCB information.

```c++
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
  if (used == last-first+1 && first == 0) 
    return;
  clean_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, end + 1, last - end);
  set_from_super_block(fs->volume + fs->SUPER_BLOCK_BASE_ADDRESS, fs->SUPERBLOCK_SIZE, start, last - end);
  count = end - start + 1;
  // move all the FCB block address forward [count] units
  for (int i = 0; i < fs->FCB_ENTRIES; ++i) {
    uchar * current_FCB = fs->volume + fs->FCB_BASE_ADDRESS + fs->FCB_SIZE * i;
    if (get_FCB_status(current_FCB) == 0) continue;
    u32 block_addr = get_FCB_addr(current_FCB);
    u32 file_size = get_FCB_size(current_FCB);

    if (block_addr < start) continue;
    else set_FCB_addr(current_FCB, get_FCB_addr(current_FCB)-count);
  }

  // move all storage block after empty forward [count] units
  uchar * move_buf = (uchar *)malloc(sizeof(uchar) * (last - end) * fs->STORAGE_BLOCK_SIZE);
  uchar * src_start, * dest_start;
  src_start = fs->volume + fs->FILE_BASE_ADDRESS + (end + 1) * fs->STORAGE_BLOCK_SIZE;
  dest_start = fs->volume + fs->FILE_BASE_ADDRESS + start * fs->STORAGE_BLOCK_SIZE;

  memcpy(dest_start, src_start, sizeof(uchar) * fs->STORAGE_BLOCK_SIZE * (last-end));
  return;
}
```



### Bonus Program

#### Overall Ideas

​	We maintain a tree structure on the provided file system structure. We modify the information format in the FCB, and make some adjustment on the directory storage, which should be considered as normal file.

​	To perform **MKDIR**, we add a FCB to this directory, assign the corresponding storage blocks for it.

​	To perform **CD**, we change the value the **pwd_fcb_idx** member in the **File System** struct to the FCB index of target directory.

​	To perform **CD_P**, we get into the FCB and storage block of PWD FCB to get its parent's FCB index. And update the  **pwd_fcb_idx** member in the **File System** struct to the FCB index of parent directory.

​	To perform **RM_RF**, we do remove in recursion. If the child to remove is a file, we simply clear all its content and update the information in its parent directory. If the child to remove is a directory, we remove all its child in recursion way firstly and remove it at last.

​	To perform **PWD**, we firstly get the value of **pwd_fcb_idx** member in the **File System** struct. Then we can follow its parent to find its path until the root. Then we output the path.

​	To perform **LS**, wemaintain an array outsides the file system, in which all children of PWD have their FCB index, size, and modify & create time recorded. Then we perform **insertion sort** based on the sorting standard. After insertion sort, output the array in sorted order.

​	

#### How to simulate a tree structure

​	The FCB format is modified into this way:

```
#define STATUS_POSITION 31            
#define SIZE_POSITION 30              
#define ADDR_POSITION 28              // block address, 16 bit
#define DIRECT_POSITION 16
#define MODIFY_TIME_POSITION 23
#define CREATE_TIME_POSITION 20
```

**0-19 bytes**: file name

**20-22 bytes**: file create time

**23-25 bytes**: file modify time

**26-27 bytes**: the first bit to indicate the file as a directory. The remained **15** bits to store the actual size of directory (sum of all its children name length).

**28-29 bytes**: the first bit is the **valid** bit, 1 to indicate it as a working FCB. The remained **15** bits store the block address information.

**30-31bytes**: the last **10** bits store the size of file. (1024 fixed for directory).

​	The storage block format for directory:

```
#define CHLD_NUM_OFFSET_IN_BLOCK 2
#define PARENT_OFFSET_IN_BLOCK 0
```

**0-1 bytes**: the FCB index of its parent directory.

**2-2 bytes:** number of its children.

The remain space is divided into many **2-byte unit**, each of which stores a child's FCB index.



#### How to do recursive remove

​	Here is the implementation of recursive remove of a directory.

```cpp
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
```





## 4. Test Results

### Required Homework - Provided User Program

#### Compilation

```shell
cd <DIREC_TO_PROJ>
cd Source
make all
```

![image-20211124143555675](report.assets/image-20211124143555675.png)

#### Execution & Result Validation

##### Test1

```shell
./test # user test1
```

###### Output

![image-20211124143630255](report.assets/image-20211124143630255.png)

```
===sort by modified time===
t.txt
b.txt
===sort by file size===
t.txt 32
b.txt 32
===sort by file size===
t.txt 32
b.txt 12
===sort by modified time===
b.txt
t.txt
===sort by file size===
b.txt 12
```

###### Snapshot

```shell
xxd snapshot.bin > sp01.txt
```

<img src="report.assets/image-20211124144311527.png" alt="image-20211124144311527" style="zoom: 80%;" />

Compare with others' results.

```shell
diff snapshot.bin snapshot01.bin
```

![image-20211124145640324](report.assets/image-20211124145640324.png)

Identical!



##### Test2

```shell
./test # user test2
```

###### Output

<img src="report.assets/image-20211124144820138.png" alt="image-20211124144820138" style="zoom: 80%;" />

```
===sort by modified time===
t.txt
b.txt
===sort by file size===
t.txt 32
b.txt 32
===sort by file size===
t.txt 32
b.txt 12
===sort by modified time===
b.txt
t.txt
===sort by file size===
b.txt 12
===sort by file size===
*ABCDEFGHIJKLMNOPQR 33
)ABCDEFGHIJKLMNOPQR 32
(ABCDEFGHIJKLMNOPQR 31
'ABCDEFGHIJKLMNOPQR 30
&ABCDEFGHIJKLMNOPQR 29
%ABCDEFGHIJKLMNOPQR 28
$ABCDEFGHIJKLMNOPQR 27
#ABCDEFGHIJKLMNOPQR 26
"ABCDEFGHIJKLMNOPQR 25
!ABCDEFGHIJKLMNOPQR 24
b.txt 12
===sort by modified time===
*ABCDEFGHIJKLMNOPQR
)ABCDEFGHIJKLMNOPQR
(ABCDEFGHIJKLMNOPQR
'ABCDEFGHIJKLMNOPQR
&ABCDEFGHIJKLMNOPQR
b.txt
```

###### Snapshot

```shell
xxd snapshot.bin > sp02.txt
```

<img src="report.assets/image-20211124144902367.png" alt="image-20211124144902367" style="zoom:67%;" />

Compare with others' results.

```shell
diff snapshot.bin snapshot02.bin
```

![image-20211124145547188](report.assets/image-20211124145547188.png)

Identical!





##### Test3

```shell
./test # user test3
```

###### Output (partial)

<img src="report.assets/image-20211124145031252.png" alt="image-20211124145031252" style="zoom:67%;" />

```
===sort by modified time===
t.txt
b.txt
===sort by file size===
t.txt 32
b.txt 32
===sort by file size===
t.txt 32
b.txt 12
===sort by modified time===
b.txt
t.txt
===sort by file size===
b.txt 12
===sort by file size===
*ABCDEFGHIJKLMNOPQR 33
)ABCDEFGHIJKLMNOPQR 32
(ABCDEFGHIJKLMNOPQR 31
'ABCDEFGHIJKLMNOPQR 30
&ABCDEFGHIJKLMNOPQR 29
%ABCDEFGHIJKLMNOPQR 28
$ABCDEFGHIJKLMNOPQR 27
#ABCDEFGHIJKLMNOPQR 26
"ABCDEFGHIJKLMNOPQR 25
!ABCDEFGHIJKLMNOPQR 24
b.txt 12
===sort by modified time===
*ABCDEFGHIJKLMNOPQR
)ABCDEFGHIJKLMNOPQR
(ABCDEFGHIJKLMNOPQR
'ABCDEFGHIJKLMNOPQR
&ABCDEFGHIJKLMNOPQR
b.txt
```

###### Snapshot

Compare with others' results.

```shell
diff snapshot.bin snapshot03.bin
```

![image-20211124145428356](report.assets/image-20211124145428356.png)

Identical!



#### Clean the Compiled Files

```shell
make clean
```

![image-20211124150056262](report.assets/image-20211124150056262.png)







### Bonus Task

#### Compilation

```shell
cd <DIREC_TO_PROJ>
cd Bonus
make all
```

![image-20211124150236891](report.assets/image-20211124150236891.png)

#### Execution

```shell
./test
```

###### Output

![image-20211124150641951](report.assets/image-20211124150641951.png)

```
===sort by modified time===
t.txt
b.txt
===sort by file size===
t.txt 32
b.txt 32
===sort by modified time===
app d
t.txt
b.txt
===sort by file size===
t.txt 32
b.txt 32
app 0 d
===sort by file size===
===sort by file size===
a.txt 64
b.txt 32
soft 0 d
===sort by modified time===
soft d
b.txt
a.txt
/app/soft
===sort by file size===
B.txt 1024
C.txt 1024
D.txt 1024
A.txt 64
===sort by file size===
a.txt 64
b.txt 32
soft 24 d
/app
===sort by file size===
t.txt 32
b.txt 32
app 17 d
===sort by file size===
a.txt 64
b.txt 32
===sort by file size===
t.txt 32
b.txt 32
app 12 d
```

###### Snapshot

```
xxd snapshot.bin > sp.txt
```

![image-20211124151638338](report.assets/image-20211124151638338.png)

#### Clean the Compiled Files

```shell
make clean
```

![image-20211124150907417](report.assets/image-20211124150907417.png)





## 5. Conclusion

​		According to the tests results, we can conclude that the design and code implementation of these two programs (**basic and bonus**) are successful.

​	In these tasks, I learnt:

* The FCB (file control block) is an important and useful method to maintain a well-functioned file system.
* The space for file system is valuable. We should make good use of each bit in some file system structure. And compaction is a good way to solve external fragmentation with contiguous block allocation design. 



