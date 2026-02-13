#!/usr/bin/env python3
"""
Snake Game - Cached WASM Edition

Uses pre-compiled function library for INSTANT execution.
No API calls, no waiting - pure speed!

Usage:
    python demos/snake_cached.py
"""

import sys
import os
import time
import random
import select
import termios
import tty
from typing import List, Tuple, Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zerolang import ZeroLangEngine

# Game constants
GRID_WIDTH = 20
GRID_HEIGHT = 15
INITIAL_LENGTH = 3
GAME_SPEED = 0.12  # seconds per frame (faster now!)

# Direction constants
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3

# Characters
CHAR_EMPTY = ' '
CHAR_SNAKE_HEAD = '█'
CHAR_SNAKE_BODY = '▓'
CHAR_FOOD = '●'
CHAR_BORDER_H = '═'
CHAR_BORDER_V = '║'


class SnakeGame:
    """Snake game powered by cached WASM functions."""
    
    def __init__(self, engine: ZeroLangEngine):
        self.engine = engine
        
        # Pre-load WAT for all needed functions
        self.wat_cache = {
            'get_dx': engine.library.get('get_dx'),
            'get_dy': engine.library.get('get_dy'),
            'in_bounds': engine.library.get('in_bounds'),
            'pack_point': engine.library.get('pack_point'),
        }
        
        # Game state
        self.snake: List[Tuple[int, int]] = []
        self.food: Tuple[int, int] = (0, 0)
        self.direction: int = DIR_RIGHT
        self.score: int = 0
        self.alive: bool = True
        self.high_score: int = 0
    
    def reset(self):
        """Reset game state."""
        start_x = GRID_WIDTH // 2
        start_y = GRID_HEIGHT // 2
        
        self.snake = [(start_x - i, start_y) for i in range(INITIAL_LENGTH)]
        self.food = self._spawn_food()
        self.direction = DIR_RIGHT
        self.score = 0
        self.alive = True
    
    def _spawn_food(self) -> Tuple[int, int]:
        """Spawn food at random position."""
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in self.snake:
                return (x, y)
    
    def _wasm_get_dx(self, direction: int) -> int:
        """Get X delta using WASM."""
        return self.engine.execute(self.wat_cache['get_dx'], 'get_dx', [direction])
    
    def _wasm_get_dy(self, direction: int) -> int:
        """Get Y delta using WASM."""
        return self.engine.execute(self.wat_cache['get_dy'], 'get_dy', [direction])
    
    def _wasm_in_bounds(self, x: int, y: int) -> bool:
        """Check bounds using WASM."""
        result = self.engine.execute(
            self.wat_cache['in_bounds'], 
            'in_bounds', 
            [x, y, GRID_WIDTH, GRID_HEIGHT]
        )
        return result == 1
    
    def update(self) -> bool:
        """Update game state. Returns False if game over."""
        if not self.alive:
            return False
        
        # Get movement delta from WASM
        dx = self._wasm_get_dx(self.direction)
        dy = self._wasm_get_dy(self.direction)
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        new_x = head_x + dx
        new_y = head_y + dy
        
        # Check wall collision using WASM
        if not self._wasm_in_bounds(new_x, new_y):
            self.alive = False
            if self.score > self.high_score:
                self.high_score = self.score
            return False
        
        # Check self collision
        if (new_x, new_y) in self.snake[:-1]:
            self.alive = False
            if self.score > self.high_score:
                self.high_score = self.score
            return False
        
        # Move snake
        self.snake.insert(0, (new_x, new_y))
        
        # Check food
        if (new_x, new_y) == self.food:
            self.score += 10
            self.food = self._spawn_food()
        else:
            self.snake.pop()
        
        return True
    
    def set_direction(self, new_dir: int):
        """Set direction (can't reverse)."""
        opposites = {DIR_UP: DIR_DOWN, DIR_DOWN: DIR_UP,
                     DIR_LEFT: DIR_RIGHT, DIR_RIGHT: DIR_LEFT}
        if opposites.get(self.direction) != new_dir:
            self.direction = new_dir
    
    def render(self) -> str:
        """Render game to string."""
        lines = []
        
        # Title
        title = f" 🐍 SNAKE - Score: {self.score} | High: {self.high_score} "
        padding = (GRID_WIDTH + 2 - len(title)) // 2
        lines.append(f"╔{'═' * padding}{title}{'═' * (GRID_WIDTH + 2 - len(title) - padding)}╗")
        
        # Grid
        for y in range(GRID_HEIGHT):
            row = "║ "
            for x in range(GRID_WIDTH):
                if (x, y) == self.snake[0]:
                    row += CHAR_SNAKE_HEAD
                elif (x, y) in self.snake:
                    row += CHAR_SNAKE_BODY
                elif (x, y) == self.food:
                    row += CHAR_FOOD
                else:
                    row += CHAR_EMPTY
            row += " ║"
            lines.append(row)
        
        # Footer
        lines.append(f"╠{'═' * (GRID_WIDTH + 2)}╣")
        controls = " WASD/Arrows: Move | Q: Quit | R: Restart "
        lines.append(f"║{controls:^{GRID_WIDTH + 2}}║")
        lines.append(f"╚{'═' * (GRID_WIDTH + 2)}╝")
        
        # WASM indicator
        lines.append(f"\n  ⚡ Powered by WASM (63 cached functions)")
        
        return '\n'.join(lines)


class Terminal:
    """Terminal input/output handler."""
    
    def __init__(self):
        self.old_settings = None
    
    def setup(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    
    def cleanup(self):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self) -> Optional[str]:
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == '\x1b':
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    key += sys.stdin.read(2)
            return key
        return None
    
    def clear(self):
        print('\033[2J\033[H', end='')
    
    def hide_cursor(self):
        print('\033[?25l', end='')
    
    def show_cursor(self):
        print('\033[?25h', end='')


def main():
    # Initialize engine with cached functions
    print("🐍 Snake Game - WASM Cached Edition\n")
    print("[*] Loading function library...")
    
    engine = ZeroLangEngine()
    lib_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "functions.json"
    )
    engine.load_library(lib_path)
    
    print("[✓] Ready!\n")
    print("Press any key to start...")
    
    game = SnakeGame(engine)
    terminal = Terminal()
    
    terminal.setup()
    try:
        # Wait for key
        while not terminal.get_key():
            time.sleep(0.1)
        
        game.reset()
        terminal.clear()
        terminal.hide_cursor()
        
        last_update = time.time()
        
        while True:
            # Input
            key = terminal.get_key()
            if key:
                if key.lower() == 'q':
                    break
                elif key in ('w', '\x1b[A'):
                    game.set_direction(DIR_UP)
                elif key in ('s', '\x1b[B'):
                    game.set_direction(DIR_DOWN)
                elif key in ('a', '\x1b[D'):
                    game.set_direction(DIR_LEFT)
                elif key in ('d', '\x1b[C'):
                    game.set_direction(DIR_RIGHT)
                elif key.lower() == 'r':
                    game.reset()
            
            # Update
            current = time.time()
            if current - last_update >= GAME_SPEED:
                if not game.update():
                    terminal.clear()
                    print(game.render())
                    print(f"\n  💀 GAME OVER!")
                    print(f"  Press R to restart, Q to quit")
                    
                    while True:
                        key = terminal.get_key()
                        if key and key.lower() == 'q':
                            return
                        elif key and key.lower() == 'r':
                            game.reset()
                            break
                        time.sleep(0.1)
                
                last_update = current
            
            # Render
            terminal.clear()
            print(game.render())
            
            time.sleep(0.016)  # ~60fps input polling
    
    finally:
        terminal.show_cursor()
        terminal.cleanup()
        print("\n\nThanks for playing! 🎮")
        engine.print_stats()


if __name__ == "__main__":
    main()
