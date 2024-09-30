"""
获取游戏状态
"""

import asyncio
import websockets
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import threading
import torch

matplotlib.use('TkAgg')

from .pacman_map import map_cfg
key = [39, 40, 37, 38, 13]

# WebSocket 服务器类
class WebSocketServer():
    def __init__(self, port=8765):
        self.port = port
        self.active_connections = None

    # 处理 WebSocket 消息并传递给回调函数
    async def handle_message(self, websocket, on_message_callback):
        print("Client connected")
        self.active_connections = websocket
        try:
            async for message in websocket:
                data = json.loads(message)  # 接收 JSON 格式数据

                # 调用回调函数处理消息
                if on_message_callback:
                    on_message_callback(data)

                # 响应给客户端
                # response = {"status": "Data received"}
                # await websocket.send(json.dumps(response))  # 将响应发回客户端
            
        except websockets.ConnectionClosed:
            print("Client disconnected")

    # 启动 WebSocket 服务器
    async def start_server(self, on_message_callback):
        async with websockets.serve(lambda ws: self.handle_message(ws, on_message_callback), "localhost", self.port):
            print(f"WebSocket server started on ws://localhost:{self.port}")
            await asyncio.Future()  # 无限运行

    async def send_message_to_client(self, message):
        if self.active_connections:  # 检查是否有活动连接
            # print("Send message.")
            await self.active_connections.send(json.dumps(message))
        else:
            print("No active connections to send message.")

# 单个PACMAN游戏
class PACMAN():
    def __init__(self, port = 8765):

        self.port = port
        self.stage = 0 # 第几关
        self.websocket = WebSocketServer(port)

        # 任务位置
        self.pacman, self.npc, self.npc_status = None, None, None
        self.init_status()

        self.score_ = 0
        self.reward = 0
        self.life = 5
        self.done = True # 记录游戏是否开始，因为开始需要按一下enter

        self.map = self.init_map()

    def init_status(self):
        # 设置初始位置
        self.pacman = [23, 13.5]
        self.npc = [[14, 12], [14, 13], [14, 14], [14, 15]]

        # 当前状态 1 正常 3 虚弱 4 被吃
        self.npc_status = np.array([1, 1, 1, 1])

    def init_map(self):
        temp_map = np.array(map_cfg[self.stage]['map'])
        temp_map = np.where(temp_map == 0, 1, -1) # 將map中的非0值（墻壁）變爲-1，豆子變爲1

        goods = map_cfg[self.stage]['goods'] # 超级豆位置
        
        goods = [tuple(map(int, key.split(','))) for key in goods.keys()]
        for good in goods:
            temp_map[good[1], good[0]] = 2  # 超級豆设为2

        self.init_status()

        if self.pacman[1] == 13.5: # 因为游戏初始位置时13.5，所以初始的时候会直接吃掉13、14的豆子
            temp_map[self.pacman[0], 13] = 0
            temp_map[self.pacman[0], 14] = 0
            self.score_ += 2
        else:
            temp_map[self.pacman[0], self.pacman[1]] = 0
            self.score_ += 1

        return temp_map
    
    # 处理从 WebSocket 收到的消息
    def load_message(self, data):
        # print(f"port:{self.port} {data}")
        if data['type'] == 'pacman':
            self.pacman = [data['position']['y'], data['position']['x']]
            # 吃豆加分
            if self.map[self.pacman[0], self.pacman[1]]:
                self.score_ += 1 
                if self.map[self.pacman[0], self.pacman[1]] == 2: # 吃了超级豆
                    self.npc_status = np.where(self.npc_status == 1, 3, self.npc_status)
                    # print("self.npc_status", self.npc_status)
                
                self.map[self.pacman[0], self.pacman[1]] = 0

            # 没吃豆，走重复路
            else:
                self.reward -= 2

        elif data['type'] == 'npc':
            self.npc[data['id']] = [data['position']['y'], data['position']['x']]

        elif data['type'] == 'status':
            # print(f"port:{self.port} {data}")
            self.npc_status[data['id']] = data['value']

        # 有东西被吃了
        elif data['type'] == 'eat':
            # 吃豆人变强后吃掉怪物
            if self.npc_status[data['id']] == 3:
                print("eat eat", data['id'])
                self.npc_status[data['id']] = 4
                self.score_ += 10
            # 吃豆人被吃
            elif self.npc_status[data['id']] == 1:
                self.life -= 1
                self.reward -= 50
                if self.life == 0:
                    self.stage = 0
                    self.score_ = 0
                    self.life = 5
                    self.done = True
                
                self.init_status()

        # 通过一关
        elif data['type'] == 'nextStage':
            self.stage += 1
            self.reward += 100
            self.init_map()

    async def display_game(self):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.fig.patch.set_facecolor('black')

        while True:
            self.ax.clear()
            temp_map = np.copy(self.map)
            self.ax.imshow(temp_map, cmap="gray")  # -1为墙，0为空地，1为豆子，2为超级豆

            # 绘制Pacman位置
            pacman_x, pacman_y = self.pacman
            self.ax.scatter(pacman_y, pacman_x, color='#ffe600', label='Pacman', s=80, marker='o')  # Pacman

            # 绘制NPC位置
            for idx, npc_pos in enumerate(self.npc):
                npc_x, npc_y = npc_pos
                npc_color = '#00ccff' if self.npc_status[idx] == 1 else '#ff9933' if self.npc_status[idx] == 3 else 'gray'
                self.ax.scatter(npc_y, npc_x, color=npc_color, label=f'NPC {idx+1}', s=80, marker='h')

            # 显示得分和生命
            self.ax.set_title(f'Score: {self.score_} | Lives: {self.life} | Reward: {self.reward}', fontdict={'color': 'white', 'weight': 'bold', 'fontsize': 16})
            
            # 强制绘图并暂停以更新
            plt.draw()
            plt.pause(0.1)  # 短暂暂停，以便实时刷新

            await asyncio.sleep(0.1)  # 异步等待，以避免阻塞其他任务

    async def start(self):
        await self.websocket.start_server(self.load_message)

    # key = [39, 40, 37, 38, 13] 右 下 左 上 回车
    # 输入Enter 开始游戏或继续游戏
    async def enter(self):
        await self.websocket.send_message_to_client({"keyCode": key[4]})
        self.done = not self.done   # 改变游戏状态 暂停 <-> 继续

    async def turn(self, direction):
        assert (direction >= 0 and direction <= 3)
        await self.websocket.send_message_to_client({"keyCode": key[direction]})

    # 获取状态
    def get_reward(self):
        return self.score_ + self.reward
    
    def get_frame(self):
        # 创建一个形状为 (2, height, width) 的 NumPy 数组
        frame = np.array([self.map.copy(), np.zeros_like(self.map)], dtype=np.float32)

        # 在图2中标上人物坐标和状态
        frame[1][int(self.pacman[0]), int(self.pacman[1])] = 2
        for status, npc in zip(self.npc_status, self.npc):
            frame[1][npc[0], npc[1]] = status

        return torch.tensor(frame)

    def get_possible_direction(self):
        dirc = []
        x, y = int(self.pacman[0]), int(self.pacman[1])

        if self.map[x][y + 1] != -1:
            dirc.append(0)

        if self.map[x+1, y] != -1:
            dirc.append(1)

        if self.map[x, y-1] != -1:
            dirc.append(2)

        if self.map[x-1, y] != -1:
            dirc.append(3)
        
        return dirc

# 多PACMAN游戏
class Multi_PACMAN():
    def __init__(self, ports = [8765]):
        self.pacmans = [PACMAN(port=port) for port in ports]
        self.loop = asyncio.new_event_loop()
        
    # 定义一个运行多个 PACMAN 实例的函数        
    async def run_multiple_pacman(self):
        tasks = [
            self.pacmans[0].display_game(),               # 异步展示游戏状态, 调试用, 不需要的时候可以注释掉, 关闭先关游戏窗口
            *[pacman.start() for pacman in self.pacmans]  # 异步运行游戏逻辑
        ]
        await asyncio.gather(*tasks)

    def start_pacman_thread(self):
        # 将异步任务提交到事件循环中运行
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.run_multiple_pacman())

    def run(self):
        PACMAN_thread = threading.Thread(target=self.start_pacman_thread)
        PACMAN_thread.daemon = True
        PACMAN_thread.start()

# 运行多个 PACMAN 实例
if __name__ == "__main__":
    ports = [8765] # 定义不同的端口
    pacmans = Multi_PACMAN(ports)
    pacmans.main()