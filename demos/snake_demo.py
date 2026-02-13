#!/usr/bin/env python3
"""
Snake Demo - WASM Game Logic + Python Host

Demonstrates ZeroLang by running Snake game logic in WebAssembly.
The AI-generated WASM handles: movement, collision, scoring
Python handles: input, rendering, game loop
"""

import sys
import os
import time
import random
import select
import termios
import tty
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradio_client import Client

# Game constants
GRID_WIDTH = 20
GRID_HEIGHT = 15
INITIAL_LENGTH = 3
GAME_SPEED = 0.15  # seconds per frame

# Direction constants (matching trained model)
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3

# Characters for rendering
CHAR_EMPTY = '·'
CHAR_SNAKE_HEAD = '█'
CHAR_SNAKE_BODY = '▓'
CHAR_FOOD = '●'
CHAR_WALL = '▒'


@dataclass
class GameState:
    snake: List[Tuple[int, int]]  # List of (x, y) positions, head first
    food: Tuple[int, int]
    direction: int
    score: int
    alive: bool


class WASMGameLogic:
    """Handles WASM function generation and caching."""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.client = None
        self.runtime = None
        self.wat_cache = {}
        
    def connect(self):
        """Connect to the API."""
        print("[*] Connecting to ZeroLang API...")
        try:
            self.client = Client(self.api_url, verbose=False)
            print("[✓] Connected!")
        except Exception as e:
            print(f"[✗] Failed to connect: {e}")
            return False
        
        print("[*] Initializing WASM runtime...")
        try:
            from zrun.runtime import ZeroLangRuntime
            self.runtime = ZeroLangRuntime()
            print("[✓] Runtime ready!")
        except Exception as e:
            print(f"[✗] Runtime failed: {e}")
            return False
        
        return True
    
    def get_wat(self, instruction: str) -> Optional[str]:
        """Get WAT code, using cache if available."""
        if instruction in self.wat_cache:
            return self.wat_cache[instruction]
        
        try:
            wat = self.client.predict(instruction, api_name="/predict")
            self.wat_cache[instruction] = wat
            return wat
        except Exception as e:
            print(f"[!] WAT generation failed: {e}")
            return None
    
    def call(self, instruction: str, func_name: str, args: List[int]) -> Optional[int]:
        """Generate WAT and execute function."""
        wat = self.get_wat(instruction)
        if not wat:
            return None
        
        try:
            return self.runtime.execute_wat(wat, func_name, args)
        except Exception as e:
            print(f"[!] Execution failed: {e}")
            return None


class SnakeGame:
    """Snake game using WASM for game logic."""
    
    def __init__(self, wasm: WASMGameLogic):
        self.wasm = wasm
        self.state = None
        self.preload_functions()
    
    def preload_functions(self):
        """Pre-generate commonly used WAT functions."""
        print("\n[*] Pre-loading game functions...")
        
        functions = [
            "Implement: int pack_point(int x, int y) - packs 2 coords into 1 int",
            "Implement: int get_dx(int direction) - x delta for direction (0=UP,1=RIGHT,2=DOWN,3=LEFT)",
            "Implement: int get_dy(int direction) - y delta for direction",
            "Implement: int in_bounds(int x, int y, int w, int h) - check if in bounds",
            "Implement: int point_equal(int p1, int p2) - returns 1 if equal",
        ]
        
        for i, func in enumerate(functions):
            print(f"  [{i+1}/{len(functions)}] Loading...", end='\r')
            self.wasm.get_wat(func)
        
        print(f"  [✓] Loaded {len(functions)} functions    ")
    
    def reset(self):
        """Reset game to initial state."""
        # Snake starts in the middle, going right
        start_x = GRID_WIDTH // 2
        start_y = GRID_HEIGHT // 2
        
        self.state = GameState(
            snake=[(start_x - i, start_y) for i in range(INITIAL_LENGTH)],
            food=self.spawn_food([]),
            direction=DIR_RIGHT,
            score=0,
            alive=True
        )
    
    def spawn_food(self, snake: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Spawn food at random position not on snake."""
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in snake:
                return (x, y)
    
    def update(self) -> bool:
        """Update game state. Returns False if game over."""
        if not self.state.alive:
            return False
        
        # Get direction deltas using WASM
        dx = self.wasm.call(
            "Implement: int get_dx(int direction) - x delta for direction (0=UP,1=RIGHT,2=DOWN,3=LEFT)",
            "get_dx",
            [self.state.direction]
        )
        dy = self.wasm.call(
            "Implement: int get_dy(int direction) - y delta for direction",
            "get_dy",
            [self.state.direction]
        )
        
        if dx is None or dy is None:
            # Fallback to Python if WASM fails
            dx = [0, 1, 0, -1][self.state.direction]
            dy = [-1, 0, 1, 0][self.state.direction]
        
        # Calculate new head position
        head_x, head_y = self.state.snake[0]
        new_x = head_x + dx
        new_y = head_y + dy
        
        # Check bounds using WASM
        in_bounds = self.wasm.call(
            "Implement: int in_bounds(int x, int y, int w, int h) - check if in bounds",
            "in_bounds",
            [new_x, new_y, GRID_WIDTH, GRID_HEIGHT]
        )
        
        if in_bounds is None:
            # Fallback
            in_bounds = 1 if (0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT) else 0
        
        if not in_bounds:
            self.state.alive = False
            return False
        
        # Check self collision
        if (new_x, new_y) in self.state.snake[:-1]:
            self.state.alive = False
            return False
        
        # Move snake
        self.state.snake.insert(0, (new_x, new_y))
        
        # Check food collision
        if (new_x, new_y) == self.state.food:
            self.state.score += 10
            self.state.food = self.spawn_food(self.state.snake)
        else:
            self.state.snake.pop()
        
        return True
    
    def set_direction(self, new_dir: int):
        """Set direction if not opposite."""
        # Can't reverse direction
        opposites = {DIR_UP: DIR_DOWN, DIR_DOWN: DIR_UP, 
                     DIR_LEFT: DIR_RIGHT, DIR_RIGHT: DIR_LEFT}
        if opposites.get(self.state.direction) != new_dir:
            self.state.direction = new_dir
    
    def render(self) -> str:
        """Render game to string."""
        lines = []
        
        # Header
        lines.append(f"╔{'═' * (GRID_WIDTH + 2)}╗")
        lines.append(f"║ Score: {self.state.score:<{GRID_WIDTH - 6}} ║")
        lines.append(f"╠{'═' * (GRID_WIDTH + 2)}╣")
        
        # Grid
        for y in range(GRID_HEIGHT):
            row = "║ "
            for x in range(GRID_WIDTH):
                if (x, y) == self.state.snake[0]:
                    row += CHAR_SNAKE_HEAD
                elif (x, y) in self.state.snake:
                    row += CHAR_SNAKE_BODY
                elif (x, y) == self.state.food:
                    row += CHAR_FOOD
                else:
                    row += CHAR_EMPTY
            row += " ║"
            lines.append(row)
        
        # Footer
        lines.append(f"╠{'═' * (GRID_WIDTH + 2)}╣")
        lines.append(f"║ {'WASD/Arrows to move, Q to quit':<{GRID_WIDTH}} ║")
        lines.append(f"╚{'═' * (GRID_WIDTH + 2)}╝")
        
        return '\n'.join(lines)


class Terminal:
    """Handle terminal input/output."""
    
    def __init__(self):
        self.old_settings = None
    
    def setup(self):
        """Set up terminal for raw input."""
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    
    def cleanup(self):
        """Restore terminal settings."""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self) -> Optional[str]:
        """Get key press if available (non-blocking)."""
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            # Handle arrow keys (escape sequences)
            if key == '\x1b':
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    key += sys.stdin.read(2)
            return key
        return None
    
    def clear(self):
        """Clear screen."""
        print('\033[2J\033[H', end='')
    
    def hide_cursor(self):
        """Hide cursor."""
        print('\033[?25l', end='')
    
    def show_cursor(self):
        """Show cursor."""
        print('\033[?25h', end='')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Snake Demo - ZeroLang WASM Game")
    parser.add_argument("--api", "-a", type=str, required=True,
                        help="Gradio API URL")
    args = parser.parse_args()
    
    # Initialize
    wasm = WASMGameLogic(args.api)
    if not wasm.connect():
        print("Failed to initialize. Exiting.")
        return
    
    game = SnakeGame(wasm)
    terminal = Terminal()
    
    print("\n" + "="*50)
    print("   🐍 SNAKE DEMO - Powered by ZeroLang WASM")
    print("="*50)
    print("\nControls: WASD or Arrow keys to move, Q to quit")
    print("\nPress any key to start...")
    
    terminal.setup()
    try:
        # Wait for key to start
        while not terminal.get_key():
            time.sleep(0.1)
        
        game.reset()
        terminal.clear()
        terminal.hide_cursor()
        
        last_update = time.time()
        
        while True:
            # Handle input
            key = terminal.get_key()
            if key:
                if key.lower() == 'q':
                    break
                elif key in ('w', '\x1b[A'):  # W or Up arrow
                    game.set_direction(DIR_UP)
                elif key in ('s', '\x1b[B'):  # S or Down arrow
                    game.set_direction(DIR_DOWN)
                elif key in ('a', '\x1b[D'):  # A or Left arrow
                    game.set_direction(DIR_LEFT)
                elif key in ('d', '\x1b[C'):  # D or Right arrow
                    game.set_direction(DIR_RIGHT)
            
            # Update game at fixed interval
            current_time = time.time()
            if current_time - last_update >= GAME_SPEED:
                if not game.update():
                    # Game over
                    terminal.clear()
                    print(game.render())
                    print(f"\n   💀 GAME OVER! Final Score: {game.state.score}")
                    print("\n   Press R to restart, Q to quit")
                    
                    while True:
                        key = terminal.get_key()
                        if key and key.lower() == 'q':
                            return
                        elif key and key.lower() == 'r':
                            game.reset()
                            break
                        time.sleep(0.1)
                
                last_update = current_time
            
            # Render
            terminal.clear()
            print(game.render())
            
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
    
    finally:
        terminal.show_cursor()
        terminal.cleanup()
        print("\nThanks for playing! 🎮")


if __name__ == "__main__":
    main()
