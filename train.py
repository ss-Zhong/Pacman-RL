import agent
import pacman_state, pacman_web
import torch
import argparse
import utils
import asyncio

# 设置训练参数
def get_args():
    parser = argparse.ArgumentParser(description='Parameters for Training Pacman Bot')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers')
    parser.add_argument('--url', type=str, default='./pacman/index.html', help='the url of index')
    return parser.parse_args()

args = get_args()

async def main():
    # ports = utils.find_unused_ports(args.num_workers)
    ports = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015] # 直接输入节省时间
    ports = ports[:args.num_workers]
    utils.log_message("Ports: ",ports)

    # 初始化全局模型和优化器
    global_model = agent.DQN()  # 定义 DQN 模型
    optimizer = torch.optim.Adam(global_model.parameters(), lr=args.learning_rate)
    env = pacman_state.Multi_PACMAN(ports)

    # 启动多个工作线程
    workers = [agent.Worker(global_model, optimizer, env.pacmans[i], args) for i in range(args.num_workers)]

    task1 = asyncio.create_task(asyncio.to_thread(pacman_web.open_web, ports, zoom=0.5, index_url=args.url)) # web打开
    task2 = asyncio.create_task(asyncio.to_thread(env.run)) # 游戏记录打开

    # 开始训练
    await asyncio.sleep(5)
    await asyncio.gather(*(worker.run() for worker in workers))

    await asyncio.gather(task1, task2)


if __name__ == "__main__":
    asyncio.run(main())