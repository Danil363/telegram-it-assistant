import asyncio
import logging
from src.bot.dispetcher import main

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    try:
        print("Starting Telegram bot...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user")
    # except Exception as e:
    #     print(f"Error occurred: {e}")