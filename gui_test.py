
import pygame
import sys

def main():
    
    pygame.init()

  
    WIDTH, HEIGHT = 400, 400
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pygame Test")

    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        window.fill((0, 128, 0))  
        pygame.display.flip()


    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
