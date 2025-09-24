"""
This is a simple test script for Pygame.

It initializes Pygame, creates a window with a green background, and runs a
basic event loop. This can be used to quickly test if Pygame is installed and
working correctly in the environment.
"""
import pygame
import sys

def main():
    """
    Initializes Pygame, creates a window, and runs the main event loop.
    """
    pygame.init()

    # Set up window
    WIDTH, HEIGHT = 400, 400
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pygame Test")

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        window.fill((0, 128, 0))  # A solid green background
        pygame.display.flip()

    # Cleanly exit Pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
