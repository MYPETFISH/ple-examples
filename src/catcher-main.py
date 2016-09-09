from ple.games.catcher import Catcher
from ple import PLE

import pygame
import numpy as np
import NaiveAgent

if __name__ == '__main__':
    pygame.init()
    game = Catcher(width=256, height=256)
    game.rng = np.random.RandomState(24)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.init()
    
    ''' create learning environment '''
    p = PLE(game, fps=30, display_screen=True, force_fps=False)
    p.init()


    ''' set my agent actions and rewards '''
    myAgent = NaiveAgent(p.getActionSet())
    reward = 0.0

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.reset()

        game.step(dt)
        pygame.display.update()
        
        ''' my agent actions '''
        obs = p.getScreenRGB()
        action = myAgent.pickAction(reward, obs)
        reward = p.act(action)

    pass