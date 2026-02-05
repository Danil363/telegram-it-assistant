from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from rag_llm.RAGAgent import RAGAgent
from bot.keyboards import main, mini
from configs.config import model, tokenizer, sentence_transformer, INDEX_PATH, DATA_PATH
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup



router = Router()
storage = MemoryStorage()
agent = RAGAgent(model, tokenizer, sentence_transformer, DATA_PATH, INDEX_PATH)
user_urls = {}



@router.message(Command("start"))
async def start_handler(message: Message):
    await message.answer(f"Hello, {message.from_user.full_name}! I`m RAG-bot about IT. Please enter your question.")


@router.message(F.text)
async def handle_user_query(message: Message):
    query = message.text

    waiting_msg = await message.answer("⏳ Searching online and generating response, please wait...")

    answer, urls = await agent.generate_answer(
        query=query,
        user_message=message,
        max_tokens=5,
        k=3,
        search_online=True
    )

    if urls:
        user_urls[message.from_user.id] = urls

    is_off_topic = answer in agent.OFF_TOPIC_RESPONSES
    reply_markup = None if is_off_topic else main

    await waiting_msg.delete()
    print(message.from_user.id, message.message_id)
    await message.answer(answer, reply_markup=reply_markup)



@router.callback_query(F.data == 'explain')
async def explain(callback: CallbackQuery):
    await callback.answer('You chose to expand the answer')
    query = callback.message.text



    answer, urls = await agent.generate_answer(
        query=query,
        user_message=callback.message,
        max_tokens=150,
        k=5,
        search_online=True,
        detail=True
    )

    if urls:
        user_urls[callback.message.from_user.id] = urls
    
    await callback.message.answer(answer, reply_markup=mini)

@router.callback_query(F.data == 'resources')
async def get_resources(callback: CallbackQuery):
    await callback.answer("Fetching resources...")

    urls = user_urls.get(callback.from_user.id)

    if not urls:
        await callback.message.answer("No resources available for this answer.")
        return

    buttons = []
    for i, url in enumerate(urls, 1):
        buttons.append([InlineKeyboardButton(text=f"Resource {i}", url=url)]) 
    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)

    await callback.message.answer("Here are the resources:", reply_markup=keyboard)

@router.message(F.photo)
async def get_photo(message: Message):
    await message.answer("Sorry, but I can't work with photos yet.")

@router.message(F.audio)
async def get_audio(message: Message):
    await message.answer("Sorry, but I can't work with audio yet.")