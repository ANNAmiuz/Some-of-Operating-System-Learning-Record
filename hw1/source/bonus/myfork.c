#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#define MAXBUF 512

typedef struct
{
    int pid;
    char info[MAXBUF];
} process_t;

pid_t do_nested_fork(int argc, char *argv[], process_t *container);
process_t *create_shared_memory(int argc);


process_t *create_shared_memory(int argc)
{
    int protection = PROT_READ | PROT_WRITE;
    int visibility = MAP_ANONYMOUS | MAP_SHARED;
    return mmap(NULL, sizeof(process_t) * argc, protection, visibility, -1, 0);
}

pid_t do_nested_fork(int argc, char *argv[], process_t *container)
{
    int current = argc - 1;
    if (current == 0)
    {
        // char buf[MAXBUF];
        return 0;
    }
    else
    {
        pid_t pid;
        int status;
        char *arg[argc];
        arg[argc - 1] = NULL;
        for (int i = 0; i < argc - 1; ++i)
        {
            arg[i] = argv[i + 1];
        }
        pid = fork();
        if (pid == 0)
        {
            do_nested_fork(current, arg, container);
            execve(argv[1], arg, NULL);
        }
        else
        {
            if (waitpid(pid, &status, WUNTRACED) == -1)
            {
                perror("waitpid() failed\n");
                exit(EXIT_FAILURE);
            };

            if (WIFEXITED(status))
            {
                container[current-1].pid = pid;
                sprintf(container[current-1].info, "Child [%d] of Parent [%d] terminate normally with EXIT STATUS = %d\n", pid, getpid(), WEXITSTATUS(status));
                //printf("Child [%d] of Parent [%d] terminate normally with EXIT STATUS = %d\n", pid, getpid(), WEXITSTATUS(status));
            }
            else if (WIFSIGNALED(status))
            {
                container[current-1].pid = pid;
                sprintf(container[current-1].info, "Child [%d] of Parent [%d] is terminated by signal %d (%s)\n", pid, getpid(), WTERMSIG(status), strsignal(WTERMSIG(status)));
                //printf("Child [%d] of Parent [%d] is terminated by signal %d (%s)\n", pid, getpid(), WTERMSIG(status), strsignal(WTERMSIG(status)));
            }
            else if (WIFSTOPPED(status))
            {
                container[current-1].pid = pid;
                sprintf(container[current-1].info, "Child [%d] of Parent [%d] is terminated by signal %d (%s)\n", pid, getpid(), WSTOPSIG(status), strsignal(WSTOPSIG(status)));
                //printf("Child [%d] of Parent [%d] is terminated by signal %d (%s)\n", pid, getpid(), WSTOPSIG(status), strsignal(WSTOPSIG(status)));
            }
        }
    }
}

int main(int argc, char *argv[])
{

    /* Implement the functions here */
    process_t *container = create_shared_memory(argc);

    printf("=======================EXECUTION RESULT=======================\n");

    do_nested_fork(argc, argv, container);

    printf("=======================EXECUTION RESULT END=======================\n");

    printf("=======================SUMMARY=======================\n");

    printf("Process Tree: %d", getpid());

    for (int i=argc-2; i>=0; --i){
        printf("->%d", container[i].pid);
    }
    printf("\n\n");
    for (int i=0; i<argc-1; ++i){
        printf("%s\n", container[i].info);
    }
    printf("=======================SUMMARY END=======================\n");
    printf("%s process (%d) terminated normally\n", argv[0], getpid());
}
