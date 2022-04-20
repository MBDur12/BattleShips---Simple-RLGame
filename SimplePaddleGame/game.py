import pygame
import os
from random import randint
pygame.font.init()


"""
DEFINE CONSTANTS
"""
WIDTH, HEIGHT = 450, 500
FPS = 60
SCORE_FONT = pygame.font.SysFont("comicsans", 20)
GAMEOVER_FONT = pygame.font.SysFont("comicsans", 80)
# Player Paddle
PADDLE_WIDTH, PADDLE_HEIGHT = 60, 10
PADDLE_VELOCITY = 4
# Ball to keep up
BALL_WIDTH, BALL_HEIGHT = 20, 40
# Define custom events to call
HIT_PADDLE = pygame.USEREVENT + 1 # event to call every second to update the score count
INCREASE_BALL_SPEED = pygame.USEREVENT + 2
# Define colours
BLACK = (0, 0 ,0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class PaddleGame():
    def __init__(self):
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Paddle Ball")

       
        self.paddle = pygame.Rect(
            WIDTH//2 - PADDLE_WIDTH//2, 
            HEIGHT - 20, 
            PADDLE_WIDTH, 
            PADDLE_HEIGHT)
        
        x,y = self._set_ball_pos()
        self.ball = pygame.Rect(x, y, BALL_WIDTH, BALL_HEIGHT)
        self.ball_speed = [5,5]

        self.score = 0

    def reset():
        pass
    
    def _set_ball_pos(self):
        start_x, start_y = randint(0, WIDTH), randint(0, HEIGHT//4)
        return start_x, start_y


    
    def take_step(self):
        #1 get user information on key presses
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
            
            if event.type == HIT_PADDLE:
                self.score += 10

            if event.type == INCREASE_BALL_SPEED:
                self.ball_speed = [1.1*val for val in self.ball_speed]
        #2 make a move based on this information
        self._handle_ball()
        self.keys_pressed = pygame.key.get_pressed()
        self._move_paddle()
        #3 check if game is over (paddle missed)
        game_over = self._is_game_over()
        if game_over:
            return game_over, self.score
        #4 check if ball hit surface or collided with paddle
        self._handle_collision()
        # Move ball based on the ball_speed vector
        self.ball = self.ball.move(self.ball_speed)
        #5 update window by calling function and update tick
        self._update_win()
        self.clock.tick(FPS)
        #6 return status and score
        return game_over, self.score

    def _update_win(self):
        self.window.fill(BLACK)
        score_text = SCORE_FONT.render(f"Score: {self.score}", 1, WHITE)
        self.window.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 20))

        pygame.draw.circle(self.window, RED, (self.ball.x, self.ball.y), 15)
        pygame.draw.rect(self.window, WHITE, self.paddle)
        pygame.display.update()

    def _handle_ball(self):
        if self.ball.x <= 0 or self.ball.x >= WIDTH:
            self.ball_speed[0] = -self.ball_speed[0]
        
        if self.ball.y <= 0:
            self.ball_speed[1] = -self.ball_speed[1]

    def _move_paddle(self):
        if self.keys_pressed[pygame.K_LEFT] and self.paddle.x - PADDLE_VELOCITY > 0:
            self.paddle.x -= PADDLE_VELOCITY
        if self.keys_pressed[pygame.K_RIGHT] and self.paddle.x + PADDLE_VELOCITY + PADDLE_WIDTH < WIDTH:
            self.paddle.x += PADDLE_VELOCITY

    def _handle_collision(self):
        if (self.ball.x >= self.paddle.x and 
        self.ball.x <= self.paddle.x + PADDLE_WIDTH and 
        self.ball.y >= HEIGHT - 20 - PADDLE_HEIGHT):

            self.ball_speed[1] = -self.ball_speed[1]
            pygame.event.post(pygame.event.Event(HIT_PADDLE))

    def _is_game_over(self):
        if self.ball.y >= HEIGHT:
            return True
        return False
    
    def display_loss(self):
        game_over_text = GAMEOVER_FONT.render("Game Over", 1, WHITE)
        self.window.blit(game_over_text, 
        (WIDTH//2 - game_over_text.get_width()//2, 
        HEIGHT//2 - game_over_text.get_height()//2))
        pygame.display.update()
        pygame.time.delay(500)
        self.score = 0
    

def main():
    game = PaddleGame()

    pygame.time.set_timer(INCREASE_BALL_SPEED, 5000)

    run = True
    while run:
        game_over, score = game.take_step()

        if game_over:
            game.display_loss()
            break

    print(f"Score: {score}")
    if game_over:
        main()
    else:
        pygame.quit()

if __name__ == "__main__":
    main()
