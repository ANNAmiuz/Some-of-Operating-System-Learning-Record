#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <new> 
#include <iostream>
#define ROW 10
#define COLUMN 50
enum STATUS {RUNNING,WIN,LOSE,QUIT};
//pthread_mutex_t mutex;
//pthread_mutex_t put_mutex;
int move_per_sec = 2;
int refresh_time_ms = 100;
struct Node{
	int x , y;
	STATUS status;
	Node( int _x , int _y ) : x( _x ) , y( _y ), status(RUNNING) {}; 
	Node(){} ;
	void go_up(){
		//pthread_mutex_lock(&mutex);
		if(x) x--;
		//pthread_mutex_unlock(&mutex);
	}
	void go_down(){
		//pthread_mutex_lock(&mutex);
		if(x<ROW) x++;
		//pthread_mutex_unlock(&mutex);
	}
	void go_left(){
		//pthread_mutex_lock(&mutex);
		if(y) y--;
		//pthread_mutex_unlock(&mutex);
	}
	void go_right(){
		//pthread_mutex_lock(&mutex);
		if(y<COLUMN) y++;
		//pthread_mutex_unlock(&mutex);
	}

};

struct Log{
	int length;
	int start;
};

struct Arguments{
	Log* logs;
	Node* frog;
	Arguments( Log* _logs , Node* _frog ) : logs( _logs ) , frog( _frog ) {}; 
};

char map[ROW+10][COLUMN] ; 
void init();
void* render(void* arg);
void* capture(void* arg);

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
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

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


void msleep(int * pmilliseconds)  
{
  struct timespec ts_sleep = 
  {
    *pmilliseconds / 1000,
    (*pmilliseconds % 1000) * 1000000L
  };

  nanosleep(&ts_sleep, NULL);
}

void *logs_move( void *t ){

	struct Log* logs = ((struct Arguments*) t)->logs;
	struct Node* frog = ((struct Arguments*) t)->frog;
	while(1) {
		//pthread_mutex_lock(&mutex);
		for(int i = 1;i<ROW;i++) {
			bool frog_move = (frog->x==i &&
			 frog->y>=logs[i-1].start && frog->y<logs[i-1].start+logs[i-1].length);
			//logs[i-1].start += i%2?-1:1;
			logs[i-1].start += 1;
			logs[i-1].start = logs[i-1].start%COLUMN;
			//if(frog_move) frog->y = (frog->y+(i%2?-1:1))%COLUMN;
			if(frog_move) frog->y = (frog->y+1)%COLUMN;
		}
		//pthread_mutex_unlock(&mutex);
		int sleep_time = 1000/move_per_sec;
		msleep(&sleep_time);
		render(t);
	}
	
}

int main( int argc, char *argv[] ){
	int res; enum THREAD_INDEX {CAPTURE=0, MOVE} tid;
	pthread_t* threads = new pthread_t[3];
	Node* frog = new Node( ROW, (COLUMN-1) / 2 );
	Log* logs = new Log[ROW];
	frog->status = RUNNING;
	srand (time(NULL));
	for(int i=0;i<ROW;i++) {
		logs[i].start = rand()%(COLUMN-1);
		//printf("%d\n",logs[i].start);
		logs[i].length =COLUMN/3;
	}
	struct Arguments *args = new Arguments(logs, frog);
	// Initialize the river map and frog's starting position
	//pthread_mutex_init(&mutex, NULL);
	//pthread_mutex_init(&put_mutex,NULL);
	capture(args);
	// res = pthread_create(threads+CAPTURE, NULL, &capture, args);
	// if(res){
	// 	printf("ERROR: return code from pthread_create() is %d", res);
	// 	exit(1);
	// }

	// 	res = pthread_create(threads+MOVE, NULL, &logs_move, args);
	// if(res){
	// 	printf("ERROR: return code from pthread_create() is %d", res);
	// 	exit(1);
	// }
	//Print the map into screen


	    
	/*  Create pthreads for wood move and frog control.  */

	
	/*  Display the output for user: win, lose or quit.  */
	for(int i=0;i<2;i++){
		pthread_join(threads[i],NULL);
	}
	////pthread_exit(NULL);

}

void * capture(void* t){
	struct Log* logs = ((struct Arguments*) t)->logs;
	struct Node* frog = ((struct Arguments*) t)->frog;
	while(1)
    {
        if(kbhit())
        {	// W up S Down A Left D Right Q Quit
			int ch = getchar();
            if(ch=='W' or ch=='w') {
				frog->go_up();
				render(t);
				continue;
			}
			if(ch=='S' or ch=='s') {
				frog->go_down();
				render(t);
				continue;
			}
	
			if(ch=='A' or ch=='a') {
				frog->go_left();
				render(t);
				continue;
			}

			if(ch=='D' or ch=='d') {
				frog->go_right();
				render(t);
				continue;
			}

			if(ch=='Q' or ch=='q') break;
        }
    }
	frog->status = QUIT;
	//pthread_exit(NULL);
}


void init(){
	memset( map , 0, sizeof( map ) ) ;
	for(int i = 1; i < ROW; ++i ){	
		for(int j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for(int j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = '|' ;

	for(int j = 0; j < COLUMN - 1; ++j )	
		map[0][j]  = '|' ;
}

void* render (void* render_args) {
	Log* logs = ((struct Arguments *)render_args)->logs;
	Node* frog = ((struct Arguments *)render_args)->frog;
	//pthread_mutex_lock(&mutex);
	init();
	bool lose = (frog->x!=ROW);
	for(int i = 1;i<ROW;i++) {
		for(int j = 0; j<logs[i-1].length; j++){
			int col = (logs[i-1].start + j + COLUMN)%COLUMN;
			map[i][col] = '=';
			if(i==frog->x && col==frog->y && col!=0 && col!=COLUMN-1)
			lose = false;
		}
	}
	map[frog->x][frog->y] = '0';
	if(!frog->x) frog->status = WIN;
	else if(lose) frog->status = LOSE;
	//pthread_mutex_unlock(&mutex);
	//std::cout << "\033[2J\033[1;1H";
	system("clear");
	for(int i = 0; i <= ROW; ++i)	puts( map[i] );
	// for(int j=0;j<COLUMN-1;j++){
	// 	putchar(map[ROW-1][j]);
	// }
	//puts( map[ROW-1] );
	//pthread_mutex_unlock(&mutex);
	if(frog->status== RUNNING) return 0;
	nanosleep((const struct timespec[]){{0, 500000000L}}, NULL);
	system("clear");
	switch(frog->status) {
		case QUIT:
			printf("You quit the game.\n");
			break;
		case WIN:
			printf("You win.\n");
			break;
		case LOSE:
			printf("You lose the game.\n");
			break;
		default:
			printf("Unknown error occurred.\n");
	}
	////pthread_exit(NULL);
}

