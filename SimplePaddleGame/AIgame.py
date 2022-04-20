import pygame
import numpy as np
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
PADDLE_WIDTH, PADDLE_HEIGHT = 50, 10
PADDLE_VELOCITY = 5
# Ball to keep up
BALL_WIDTH, BALL_HEIGHT = 10, 5
# Define custom events to call
HIT_PADDLE = pygame.USEREVENT + 1 # event to call every second to update the score count
INCREASE_BALL_SPEED = pygame.USEREVENT + 2
# Define colours
BLACK = (0, 0 ,0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class PaddleGame():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.window = pygame.display.set_mode((self.width,self.height))
        self.clock = pygame.time.Clock()
        pygame.time.set_timer(INCREASE_BALL_SPEED, 5000)
        pygame.display.set_caption("Paddle Ball")

        self.reset()
       

    def reset(self):
        self.paddle = pygame.Rect(
            WIDTH//2 - PADDLE_WIDTH//2, 
            HEIGHT - 20, 
            PADDLE_WIDTH, 
            PADDLE_HEIGHT)
        
        x,y = self._set_ball_pos()
        self.ball = pygame.Rect(x, y, BALL_WIDTH, BALL_HEIGHT)
        self.ball_speed = [3,3]


        self.score = 0
    
    def _set_ball_pos(self):
        start_x, start_y = randint(5, WIDTH), 5
        return start_x, start_y


    
    def take_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
            
            if event.type == HIT_PADDLE:
                self.score += 10

            if event.type == INCREASE_BALL_SPEED:
                self.ball_speed = [1.1*val for val in self.ball_speed]
        
        reward = 0
        if np.array_equal(action, [1,0,0]) or np.array_equal(action, [0,0,1]):
            reward -= 0.1
        #2 make a move based on this information
        self._handle_ball()
        self._move_paddle(action)

        # if paddle is closer to ball reward +3, otherwise reward -3


        #3 check if game is over (paddle missed)
        game_over = self._is_game_over()
        if game_over:
            reward -= 10
            return reward, game_over, self.score
        #4 check if ball hit surface or collided with paddle - in which reward the action.
        if self._handle_collision():
            reward += 10
        # Move ball based on the ball_speed vector
        self.ball = self.ball.move(self.ball_speed)
        #5 update window by calling function and update tick
        self._update_win()
        self.clock.tick(FPS)
        #6 return status and score
        return reward, game_over, self.score

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

    def _move_paddle(self, action):
        # Move left
        if np.array_equal(action, [1,0,0]) and self.paddle.x >= 0:
            self.paddle.x -= PADDLE_VELOCITY  
        elif np.array_equal(action, [0,0,1]) and self.paddle.x + self.paddle.width <= self.width:
            self.paddle.x += PADDLE_VELOCITY

    def _handle_collision(self):
        if self.ball.colliderect(self.paddle):
            self.ball_speed[1] = -self.ball_speed[1]
            pygame.event.post(pygame.event.Event(HIT_PADDLE))
            return True
        
        return False
        

    def _is_game_over(self):
        if self.ball.y >= HEIGHT:
            return True
        return False
    
    """def display_loss(self):
        game_over_text = GAMEOVER_FONT.render("Game Over", 1, WHITE)
        self.window.blit(game_over_text, 
        (WIDTH//2 - game_over_text.get_width()//2, 
        HEIGHT//2 - game_over_text.get_height()//2))
        pygame.display.update()
        pygame.time.delay(500)
        self.reset()"""
    

def main():
    game = PaddleGame(WIDTH, HEIGHT)

    pygame.time.set_timer(INCREMENT_SCORE, 1000)
    pygame.time.set_timer(INCREASE_BALL_SPEED, 5000)

    run = True
    while run:
        reward, game_over, score = game.take_step([1,0,0])

        if game_over:
            #game.display_loss()
            break

    print(f"Score: {game.score}")
    """if game_over:
        main()
    else:"""
    pygame.quit()

if __name__ == "__main__":
    main()
