import json
import asyncio
import websockets
import logging
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình bot Telegram
BOT_TOKEN = "8053826988:AAFlCP-OPKJdr9XegaryaRkX8gWmnEknwLg"
CHAT_ID = "-1002596735298"
ADMIN_IDS = [6020088518]

# URL WebSocket
WS_URL = "ws://163.61.110.10:8000/game_sunwin/ws?id=duy914c&key=dduy1514nsadfl"

# Trạng thái bot
bot_running = False
websocket_task = None

# Khởi tạo bot Telegram
bot = Bot(token=BOT_TOKEN)

async def send_message_to_group(message):
    """Hàm gửi tin nhắn đến nhóm Telegram"""
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message)
        logger.info(f"Đã gửi tin nhắn: {message}")
    except TelegramError as e:
        logger.error(f"Lỗi khi gửi tin nhắn Telegram: {e}")

async def process_websocket_message(message):
    """Hàm xử lý dữ liệu WebSocket"""
    try:
        # Parse dữ liệu JSON
        data = json.loads(message)
        required_fields = ["Phien", "Xuc_xac_1", "Xuc_xac_2", "Xuc_xac_3", "Tong", "Ket_qua"]
        if not all(field in data for field in required_fields):
            logger.warning("Lỗi: Thiếu trường dữ liệu JSON")
            return

        phien = data["Phien"]
        xuc_xac_1 = data["Xuc_xac_1"]
        xuc_xac_2 = data["Xuc_xac_2"]
        xuc_xac_3 = data["Xuc_xac_3"]
        tong = data["Tong"]
        ket_qua = data["Ket_qua"]

        # Chuyển số thành emoji
        num_to_emoji = {
            1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣", 
            6: "6️⃣", 7: "7️⃣", 8: "8️⃣", 9: "9️⃣", 10: "🔟",
            11: "1️⃣1️⃣", 12: "1️⃣2️⃣", 13: "1️⃣3️⃣", 14: "1️⃣4️⃣", 
            15: "1️⃣5️⃣", 16: "1️⃣6️⃣", 17: "1️⃣7️⃣", 18: "1️⃣8️⃣"
        }

        # Định dạng tin nhắn
        msg = (
            f"Kết quả mới nhất sun.win\n"
            f"=====================\n"
            f"🎲 Phiên: #{phien}\n"
            f"Xúc xắc: {num_to_emoji.get(xuc_xac_1, str(xuc_xac_1))}•{num_to_emoji.get(xuc_xac_2, str(xuc_xac_2))}•{num_to_emoji.get(xuc_xac_3, str(xuc_xac_3))}\n"
            f"Tổng: {num_to_emoji.get(tong, str(tong))}\n"
            f"Kết quả: {ket_qua}"
        )

        # Gửi tin nhắn đến nhóm
        await send_message_to_group(msg)
    except json.JSONDecodeError:
        logger.error("Lỗi: Dữ liệu không phải JSON hợp lệ")
    except Exception as e:
        logger.error(f"Lỗi khi xử lý dữ liệu: {e}")

async def websocket_client():
    """Hàm chạy WebSocket với reconnect"""
    global bot_running
    while True:
        if not bot_running:
            await asyncio.sleep(1)
            continue
        try:
            logger.info("Đang kết nối đến WebSocket...")
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as ws:
                logger.info("Đã kết nối thành công đến WebSocket")
                while bot_running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=60)
                        if message:
                            await process_websocket_message(message)
                    except asyncio.TimeoutError:
                        logger.info("WebSocket timeout, kiểm tra kết nối...")
                        try:
                            await ws.ping()
                        except:
                            logger.warning("Ping failed, kết nối bị lỗi")
                            break
                    except websockets.ConnectionClosed as e:
                        logger.warning(f"Kết nối WebSocket bị đóng: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Lỗi khi nhận dữ liệu WebSocket: {e}")
                        break
        except Exception as e:
            logger.error(f"Lỗi kết nối WebSocket: {e}")
            if bot_running:
                logger.info("Thử kết nối lại sau 5 giây...")
                await asyncio.sleep(5)

async def handle_telegram_updates():
    """Hàm xử lý updates từ Telegram"""
    global bot_running, websocket_task
    
    offset = None
    while True:
        try:
            updates = await bot.get_updates(offset=offset, timeout=10)
            for update in updates:
                offset = update.update_id + 1
                logger.info(f"Nhận update: {update.update_id}")
                
                # Xử lý tin nhắn
                if update.message:
                    user_id = update.message.from_user.id
                    username = update.message.from_user.username or "Unknown"
                    message_text = update.message.text or ""
                    
                    logger.info(f"Tin nhắn từ {username} (ID: {user_id}): {message_text}")
                    
                    if message_text == "/start":
                        logger.info(f"Lệnh /start từ user ID: {user_id}, Admin IDs: {ADMIN_IDS}")
                        if user_id in ADMIN_IDS:
                            logger.info("User là admin, gửi menu")
                            keyboard = [
                                [
                                    InlineKeyboardButton("Bật bot", callback_data="start_bot"),
                                    InlineKeyboardButton("Tắt bot", callback_data="stop_bot")
                                ]
                            ]
                            reply_markup = InlineKeyboardMarkup(keyboard)
                            await bot.send_message(
                                chat_id=update.message.chat_id,
                                text="Chọn hành động:",
                                reply_markup=reply_markup
                            )
                            logger.info("Đã gửi menu thành công")
                        else:
                            logger.info("User không phải admin")
                            await bot.send_message(
                                chat_id=update.message.chat_id,
                                text=f"Xin chào! ID của bạn là: {user_id}\nChỉ admin mới có thể sử dụng bot này."
                            )
                
                # Xử lý callback query
                elif update.callback_query:
                    query = update.callback_query
                    user_id = query.from_user.id
                    
                    if user_id not in ADMIN_IDS:
                        await bot.answer_callback_query(
                            callback_query_id=query.id,
                            text="Bạn không có quyền sử dụng bot này!"
                        )
                        continue
                    
                    await bot.answer_callback_query(callback_query_id=query.id)
                    
                    if query.data == "start_bot":
                        if bot_running:
                            await bot.send_message(
                                chat_id=query.message.chat_id,
                                text="Bot đã đang chạy!"
                            )
                        else:
                            bot_running = True
                            if websocket_task is None or websocket_task.done():
                                websocket_task = asyncio.create_task(websocket_client())
                            await bot.send_message(
                                chat_id=query.message.chat_id,
                                text="Bot đã được bật!"
                            )
                    
                    elif query.data == "stop_bot":
                        if not bot_running:
                            await bot.send_message(
                                chat_id=query.message.chat_id,
                                text="Bot đã đang tắt!"
                            )
                        else:
                            bot_running = False
                            if websocket_task and not websocket_task.done():
                                websocket_task.cancel()
                                try:
                                    await websocket_task
                                except asyncio.CancelledError:
                                    pass
                            await bot.send_message(
                                chat_id=query.message.chat_id,
                                text="Bot đã được tắt!"
                            )
                            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý Telegram updates: {e}")
            await asyncio.sleep(1)

async def main():
    """Hàm chính"""
    global websocket_task
    
    logger.info("Đang khởi động bot...")
    logger.info(f"Bot Token: {BOT_TOKEN[:10]}...")
    logger.info(f"Chat ID: {CHAT_ID}")
    logger.info(f"Admin IDs: {ADMIN_IDS}")
    logger.info(f"WebSocket URL: {WS_URL}")
    
    try:
        # Kiểm tra bot token
        me = await bot.get_me()
        logger.info(f"Bot đã sẵn sàng: @{me.username}")
        
        # Chạy handler cho Telegram updates
        await handle_telegram_updates()
        
    except KeyboardInterrupt:
        logger.info("Bot đang dừng...")
    except Exception as e:
        logger.error(f"Lỗi khi chạy bot: {e}")
    finally:
        # Dọn dẹp
        if websocket_task and not websocket_task.done():
            websocket_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())