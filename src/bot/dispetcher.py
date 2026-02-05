from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
import asyncio
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import API_TOKEN
from bot.handlers import router, storage



bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=storage)

async def main():
    dp.include_router(router)
    await dp.start_polling(bot, skip_updates=True)
