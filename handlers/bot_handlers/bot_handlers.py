import glob
import os.path
import random
import string

import cv2
import tensorflow as tf
from aiogram import types
from aiogram.types import ContentType
from ultralytics import YOLO
import numpy as np

from config import ALLOWED_FORMATS, MODEL_PATH
from keyboards.keyboards import rate_kb
from main import bot, dp

model_y = YOLO(MODEL_PATH)
with open('Model/MobileNetV2/network.json', 'r') as json_file:
    json_saved_model = json_file.read()
network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('Model/MobileNetV2/weights4.hdf5')

os.makedirs("Temp", exist_ok=True)
os.makedirs("Processed", exist_ok=True)


def generate_random_idx(length=8):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choices(letters_and_digits, k=length))


@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    await message.answer("Hi! Please send me a photo with cats, dogs or raccoons and I will try to define it. ")


@dp.message_handler(content_types=[ContentType.PHOTO, ContentType.DOCUMENT])
async def handle_img_and_files(message: types.Message):
    photo_id = generate_random_idx()
    file_path = f"Temp/{photo_id}.jpg"
    # Handle photos
    if message.photo:
        photo = message.photo[-1]
        try:
            await photo.download(file_path)
            results_yolo = model_y(file_path)
            class_name_yolo = (results_yolo[0].names)[np.argmax(results_yolo[0].probs).item()]

            network_loaded.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
            image = cv2.imread(file_path)
            image = cv2.resize(image, (640, 640))
            image = image / 255
            image = image.reshape(-1, 640, 640, 3)
            result = network_loaded(image)
            result = np.argmax(result)
            class_nt = ''
            if result == 0:
                class_nt ='Cat'
            elif result == 2:
                class_nt ="Raccoon"
            else:
                class_nt = 'Dog'
            await message.answer(f"YOLO says it is: {class_name_yolo.upper()}\n\nMobileNetV2 says it is: {class_nt.upper()}")
            # await message.reply(f"")
            await message.answer("Please, rate the answer", reply_markup=rate_kb(photo_id))
        except Exception as e:
            await message.reply(f"Error processing file: {e}")

    # Handle documents
    elif message.document and message.document.mime_type.split("/")[0] == "image" and \
            message.document.file_name.split(".")[-1] in ALLOWED_FORMATS:
        photo = message.document
        try:
            await photo.download(file_path)
            results_yolo = model_y(file_path)
            class_name_yolo = (results_yolo[0].names)[np.argmax(results_yolo[0].probs).item()]

            network_loaded.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
            image = cv2.imread(file_path)
            image = cv2.resize(image, (640, 640))
            image = image / 255
            image = image.reshape(-1, 640, 640, 3)
            result = network_loaded(image)
            result = np.argmax(result)
            class_nt = ''
            if result == 0:
                class_nt = 'Cat'
            elif result == 2:
                class_nt = "Raccoon"
            else:
                class_nt = 'Dog'
            await message.answer(
                f"YOLO says it is: {class_name_yolo.upper()}\n\nMobileNetV2 says it is: {class_nt.upper()}")
            # await message.reply(f"")
            await message.answer("Please, rate the answer", reply_markup=rate_kb(photo_id))
        except Exception as e:
            await message.reply(f"Error processing file: {e}")
    else:
        await message.reply("Sorry, only image files in JPEG, JPG, PNG, and GIF formats are allowed.")

@dp.callback_query_handler(lambda c: c.data and c.data.startswith('rate_'))
async def handle_rate_callback(callback_query: types.CallbackQuery):
    await bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)
    photo_id = callback_query.data.split('|')[-1]
    rate = callback_query.data.split('_')[1]
    if rate == "40" or rate == "60":
        path = "Answer_gifs/Low_rate/"
        video_files = glob.glob(path + "*.mp4")
        random_video = random.choice(video_files)
        if os.path.exists(f"Temp/{photo_id}.jpg"):
            os.remove(f"Temp/{photo_id}.jpg")


    elif rate == "80" or rate == "100":
        path = "Answer_gifs/High_rate/"
        video_files = glob.glob(path + "*.mp4")
        random_video = random.choice(video_files)
        file_path = f"Temp/{photo_id}.jpg"
        if os.path.exists(file_path):
            os.remove(file_path)

    await bot.send_message(callback_query.message.chat.id, "Thanks for rating!")
    await bot.send_video(callback_query.message.chat.id, open(random_video, 'rb'))


@dp.message_handler(
    content_types=[ContentType.GAME, ContentType.STICKER, ContentType.VIDEO, ContentType.VIDEO_NOTE, ContentType.VOICE,
                   ContentType.CONTACT, ContentType.LOCATION, ContentType.AUDIO])
async def handle_other_types(message: types.Message):
    await message.answer("Sorry, only image files in JPEG, JPG, PNG, and GIF formats are allowed.")
