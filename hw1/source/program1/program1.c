#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int status;
    pid_t pid;
    printf("Process start to fork.\n");
    printf("I'm the Parent Process, my pid = %d\n", getpid());
    /* fork a child process */
    if (pid = fork() == 0)
    {
        printf("I'm the Child Process, my pid = %d\n", getpid());
        printf("Child process start to execute test program\n");
        execve(argv[1], NULL, NULL);
        //execve("./stop", NULL, NULL);
    }

    if (waitpid(pid, &status, WUNTRACED) == -1)
    {
        perror("waitpid() failed\n");
        exit(EXIT_FAILURE);
    };
    printf("Parent process receiving the SIGCHLD signal\n");
    if (WIFEXITED(status))
    {
        printf("Normal termination with EXIT STATUS = %d\n", WEXITSTATUS(status));
    }
    else if (WIFSIGNALED(status))
    {
        switch (WTERMSIG(status))
        {
        case 1:
        {
            printf("child process get SIGHUP signal\n");
            printf("child process is hungup by hangup signal\n");
            break;
        }
        case 2:
        {
            printf("child process get SIGINT signal\n");
            printf("child process is interrupted by interrupt signal\n");
            break;
        }
        case 3:
        {
            printf("child process get SIGQUIT signal\n");
            printf("child process is terminated by quit signal\n");
            break;
        }
        case 4:
        {
            printf("child process get SIGILL signal\n");
            printf("child process is terminated due to the illegal instructions\n");
            break;
        }
        case 5:
        {
            printf("child process get SIGTRAP signal\n");
            printf("child process is trapped by trap signal\n");
            break;
        }
        case 6:
        {
            printf("child process get SIGABRT signal\n");
            printf("child process is abort by abort signal\n");
            break;
        }
        case 7:
        {
            printf("child process get SIGBUS signal\n");
            printf("child process is terminated by bus signal\n");
            break;
        }
        case 8:
        {
            printf("child process get SIGFPE signal\n");
            printf("child process is terminated due to float point error signal\n");
            break;
        }
        case 9:
        {
            printf("child process get SIGKILL signal\n");
            printf("child process is killed by kill signal\n");
            break;
        }
        case 11:
        {
            printf("child process get SIGSEGV signal\n");
            printf("child process is terminated due to segmentation fault\n");
            break;
        }
        case 13:
        {
            printf("child process get SIGPIPE signal\n");
            printf("child process is terminated by pipe signal\n");
            break;
        }
        case 14:
        {
            printf("child process get SIGALRM signal\n");
            printf("child process is alarmed by alarm signal\n");
            break;
        }
        case 15:
        {
            printf("child process get SIGTERM signal\n");
            printf("child process is terminated by terminate signal\n");
            break;
        }
        default:
            printf("child process get %d signal\n", WTERMSIG(status));
        }

        printf("CHILD EXECUTION FAILED\n");
    }
    else if (WIFSTOPPED(status))
    {
        printf("child process get SIGSTOP signal\n");
        printf("child process stopped\n");
        printf("CHILD PROCESS STOPPED\n");
    }
}
