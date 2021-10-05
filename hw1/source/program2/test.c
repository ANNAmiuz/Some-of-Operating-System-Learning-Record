#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>




int main(int argc,char* argv[]){
	int signum = 13; //normal = 10

	printf("--------USER PROGRAM--------\n");

	switch (signum){
		case 1:
		{
			raise(SIGHUP);
			sleep(5);
			break;
		}
		case 2:
		{
			raise(SIGINT);
			sleep(5);
			break;
		}
		case 3:
		{
			raise(SIGQUIT);
			sleep(5);
			break;
		}
		case 4:
		{
			raise(SIGILL);
			sleep(5);
			break;
		}
		case 5:
		{
			raise(SIGTRAP);
			sleep(5);
			break;
		}
		case 6:
		{
			abort();
			sleep(5);
			break;
		}
		case 7:
		{
			raise(SIGBUS);
			sleep(5);
			break;
		}
		case 8:
		{
			raise(SIGFPE);
			sleep(5);
			break;
		}
		case 9:
		{
			raise(SIGKILL);
			sleep(5);
			break;
		}
		case 10:
		{
			printf("This is the normal program\n\n");
			break;
		}
		case 11:
		{
			raise(SIGSEGV);
			sleep(5);
			break;
		}
		case 19:
		{
			raise(SIGSTOP);
			sleep(5);
			break;
		}
		case 13:
		{
			raise(SIGPIPE);
			sleep(5);
			break;
		}
		case 14:
		{
			alarm(2);
			sleep(5);
			break;
		}
		case 15:
		{
			raise(SIGTERM);
			sleep(5);
			break;
		}
	}

//	alarm(2);
	//printf("user process success!!\n");
	printf("--------USER PROGRAM--------\n");
	return 100;
}
