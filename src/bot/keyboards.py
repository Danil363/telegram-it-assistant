from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

main = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='📝 Explain in more detail', callback_data='explain')],
    [InlineKeyboardButton(text="📚 View Resources", callback_data="resources")]
])

mini  = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="📚 View Resources", callback_data='resources')]
])