#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <math.h>

#define ROW 10
#define COLUMN 50
#define TIME 20000
#define LOGLEN 20
#define MODE1  // One thread used to update the screen
#define MODE2x // Many threads used to update the screen

struct Node
{
	int x, y;
	Node(int _x, int _y) : x(_x), y(_y){};
	Node(){};
} frog;

pthread_mutex_t end_mutex, frog_mutex;
#ifdef MODE2
pthread_mutex_t clear_mutex;
#endif
char map[ROW + 10][COLUMN];
int end = 0; // quit = 1, win = 2, lose = 3
bool is_clear = false;

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0.
int kbhit(void)
{
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if (ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

void *check_status(void *)
{
	while (1)
	{
		/*  Check keyboard hits, to change frog's position or quit the game. */
		if (kbhit())
		{
			char dir = getchar();
			if (dir == 'w' || dir == 'W')
			{
				pthread_mutex_lock(&frog_mutex);
				if (frog.x != 0)
					frog.x--;
				pthread_mutex_unlock(&frog_mutex);
			}

			if (dir == 'a' || dir == 'A')
			{
				pthread_mutex_lock(&frog_mutex);
				if (!(frog.x == ROW && frog.y == 0))
					frog.y--;
				pthread_mutex_unlock(&frog_mutex);
			}

			if (dir == 'd' || dir == 'D')
			{
				pthread_mutex_lock(&frog_mutex);
				if (!(frog.x == ROW && frog.y == COLUMN - 2))
					frog.y++;
				pthread_mutex_unlock(&frog_mutex);
			}

			if (dir == 's' || dir == 'S')
			{
				pthread_mutex_lock(&frog_mutex);
				if (frog.x != ROW)
					frog.x++;
				pthread_mutex_unlock(&frog_mutex);
			}

			if (dir == 'q' || dir == 'Q')
			{
				end = 1;
				pthread_exit(NULL);
			}
		}

		/*  Check game's status  */
		if (frog.x == 0)
		{
			end = 2;
			pthread_exit(NULL);
		}
		else if (frog.y < 0 || frog.y >= COLUMN - 1)
		{
			end = 3;
			pthread_exit(NULL);
		}
		else if (map[frog.x][frog.y] == ' ')
		{
			end = 3;
			pthread_exit(NULL);
		}
		usleep(TIME / 4);
	}
	pthread_exit(NULL);
}

#ifdef MODE1
void *display_screen(void *)
{
	while (!end)
	{
		printf("\033[H\033[2J");
		for (int i = 0; i <= ROW; ++i)
			puts(map[i]);
		usleep(TIME / 4);
	}
}
#endif

#ifdef MODE2
void *display_top(void *)
{
	while (!end)
	{
		pthread_mutex_lock(&clear_mutex);
		if (!is_clear)
		{
			is_clear = true;
			printf("\033[H\033[2J");
		}

		for (int i = 0; i < ceil(((float)ROW - 0) / 3); ++i)
			puts(map[i]);
		usleep(TIME / 4);
		is_clear = false;
	}
}

void *display_mid(void *)
{
	while (!end)
	{
		pthread_mutex_lock(&clear_mutex);
		if (!is_clear)
		{
			is_clear = true;
			printf("\033[H\033[2J");
		}
		for (int i = 0; i < ceil(((float)ROW - 1) / 3); ++i)
			puts(map[i]);
		usleep(TIME / 4);
		is_clear = false;
	}
}

void *display_botm(void *)
{
	while (!end)
	{
		pthread_mutex_lock(&clear_mutex);
		if (!is_clear)
		{
			is_clear = true;
			printf("\033[H\033[2J");
		}
		for (int i = 0; i < ceil(((float)ROW - 0) / 3); ++i)
			puts(map[i]);
		usleep(TIME / 4);
		is_clear = false;
	}
}
#endif

void *logs_move(void *)
{
	int i, j, p1, p2;
	int left[ROW - 1];
	// initialize the position of all logs （left end point)
	for (i = 0; i <= ROW; ++i)
	{
		left[i] = rand() % (COLUMN + 1);
	}
	while (!end)
	{
		/*  Move the logs  */
		for (i = 1; i < ROW; ++i)
		{
			for (j = 0; j < COLUMN - 1; ++j)
				map[i][j] = ' ';
		}

		for (i = 1; i <= ROW; ++i)
		{
			p1 = left[i];
			p2 = p1 + LOGLEN;
			if (p1 < 0 && p2 > 0)
			{
				for (j = 0; j < p2; ++j)
					map[i][j] = '=';
				for (j = COLUMN - 1 + p1; j < COLUMN - 1; ++j)
					map[i][j] = '=';
			}
			else if (p1 > 0 && p2 >= COLUMN - 1)
			{
				for (j = p1; j < COLUMN - 1; ++j)
					map[i][j] = '=';
				for (j = 0; j < p2 - (COLUMN - 1); ++j)
					map[i][j] = '=';
			}
			else if (p1 >= 0 && p2 < COLUMN - 1)
			{
				for (j = p1; j < p2; ++j)
					map[i][j] = '=';
			}
			if (i % 2)
			{
				left[i]--;
				if (left[i] == -LOGLEN)
					left[i] = COLUMN - 1 - LOGLEN;
			}
			else
			{
				left[i]++;
				if (left[i] == COLUMN - 1)
					left[i] = 0;
			}
		}

		for (j = 0; j < COLUMN - 1; ++j)
			map[ROW][j] = map[0][j] = '|';

		for (j = 0; j < COLUMN - 1; ++j)
			map[0][j] = map[0][j] = '|';

		pthread_mutex_lock(&frog_mutex);
		map[frog.x][frog.y] = '0';
		if (frog.x != 0 && frog.x != ROW)
		{
			if (frog.x % 2)
				frog.y--;
			else
				frog.y++;
		}
		pthread_mutex_unlock(&frog_mutex);

		/*  Sleep before the next loop  */
		usleep(TIME * 10);
	}
	pthread_exit(NULL);
}

int main(int argc, char *argv[])
{

	// Initialize the river map and frog's starting position
	memset(map, 0, sizeof(map));
	int i, j;
	for (i = 1; i < ROW; ++i)
	{
		for (j = 0; j < COLUMN - 1; ++j)
			map[i][j] = ' ';
	}

	for (j = 0; j < COLUMN - 1; ++j)
		map[ROW][j] = map[0][j] = '|';

	for (j = 0; j < COLUMN - 1; ++j)
		map[0][j] = map[0][j] = '|';

	frog = Node(ROW, (COLUMN - 1) / 2);
	map[frog.x][frog.y] = '0';

	// Print the map into screen
	for (i = 0; i <= ROW; ++i)
		puts(map[i]);

	// Mutex initialization
	//pthread_mutex_init(&end_mutex, NULL);
	pthread_mutex_init(&frog_mutex, NULL);
#ifdef MODE2
	pthread_mutex_init(&clear_mutex, NULL);
#endif

// The first thread control the input status from keyboard, the other control the move
#ifdef MODE1
	pthread_t threads[3];
#endif
#ifdef MODE2
	pthread_t threads[5];
#endif

	/*  Create pthreads for wood move and frog control.  */
	pthread_create(&threads[0], NULL, logs_move, NULL);
	pthread_create(&threads[1], NULL, check_status, NULL);
#ifdef MODE1
	pthread_create(&threads[2], NULL, display_screen, NULL);
#endif
#ifdef MODE2
	pthread_create(&threads[2], NULL, display_top, NULL);
	pthread_create(&threads[3], NULL, display_mid, NULL);
	pthread_create(&threads[4], NULL, display_botm, NULL);
#endif

	/*  Display the output for user: win, lose or quit.  */
	pthread_join(threads[0], NULL);
	pthread_join(threads[1], NULL);
	pthread_join(threads[2], NULL);
#ifdef MODE2
	pthread_join(threads[3], NULL);
	pthread_join(threads[4], NULL);
#endif

	//pthread_mutex_destroy(&end_mutex);
	pthread_mutex_destroy(&frog_mutex);
#ifdef MODE2
	pthread_mutex_destroy(&clear_mutex);
#endif
	printf("\033[H\033[2J");
	if (end == 1)
		printf("You exit the game.\n");
	else if (end == 2)
		printf("You win the game!!\n");
	else if (end == 3)
		printf("You lose the game!!\n");

	return 0;
}