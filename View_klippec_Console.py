import asyncio
import websockets
import json

# MoonrakerのWebSocket URL (デフォルトのMoonrakerのアドレスを使用)
MOONRAKER_URL = "ws://127.0.0.1:7125/websocket"

async def listen_to_klipper():
    async with websockets.connect(MOONRAKER_URL) as ws:
        # Klipperのコンソール出力の購読をリクエスト
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "printer.console/list", "id": 1}))
        
        while True:
            message = await ws.recv()
            data = json.loads(message)
            
            if "result" in data:
                # 初回のコンソールメッセージの取得
                for line in data["result"]:
                    print(line)
            elif "method" in data and data["method"] == "notify_console_output":
                # 新しいコンソールメッセージの通知
                for line in data["params"][0]:
                    print(line)

# asyncioループを起動
loop = asyncio.get_event_loop()
loop.run_until_complete(listen_to_klipper())
