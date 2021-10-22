# Simple pygame frog program

# Import and initialize the pygame library
import pygame
import threading
import time

# Run until the user asks to quit or the game gets a result
# 0: nothing
# 1: win
# 2: lose
# 3: quit
is_end = 0
running  = True
in_loop = False

def create_thread(tar):
    thread = threading.Thread(target=tar)
    thread.daemon = True
    thread.start()

def print_end_status():

    while (1):
        if (is_end == 1):
            print("You win the game.\n")
            break
        elif (is_end == 2):
            print("You lose the game!\n")
            break
        elif (is_end == 3):
            print("You quit the game.\n")
            break
        time.sleep(0.3)

def maintain_move():

    while (1):
        
        
        time.sleep(0.3)

        
# Frog
class Frog:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def turn_up(self):
        self.y -= 50

    def turn_down(self):
        self.y += 50

    def turn_left(self):
        self.x -= 10
    
    def turn_right(self):
        self.x += 10

class Log:
    def __init__(self, idx, start, length, height, Frog):
        self.idx = idx
        self.start = start
        self.length = length
        self.height = height
        self.frog_on = False
        self.frog = Frog

    def move(self):
        if (self.idx % 2):
            self.start += 10
            if (self.frog_on):
                self.frog.turn_right()
        else:
            self.start -= 10
            if (self.frog_on):
                self.frog.turn_left()


# Set up the drawing window
pygame.init()
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1000
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
frog = Frog(SCREEN_HEIGHT, SCREEN_WIDTH/2)
while running:
    in_loop = True
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            is_end = 3
            print("You quit the game.\n")

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                is_end = 3
                running = False
                print("You quit the game.\n")
            elif event.key == pygame.K_w:
                frog.turn_up
            elif event.key == pygame.K_a:
                frog.turn_left
            elif event.key == pygame.K_d:
                frog.turn_right
            elif event.key == pygame.K_s:
                frog.turn_down

    # Fill the background with white
    screen.fill((86, 168, 177))

    # Draw a solid blue circle in the center
    pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()