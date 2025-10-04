import tkinter as tk
import logging
from go_game import GoGame

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('go_game.log'),
        logging.StreamHandler()
    ]
)


def main():
    """Main function to start the Go game"""
    logging.info("=" * 60)
    logging.info("Starting Deep Q-Learning Go Game Application")
    logging.info("=" * 60)

    root = tk.Tk()
    game = GoGame(root)

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    logging.info("Application window created and centered")

    root.mainloop()

    logging.info("Application closed")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
