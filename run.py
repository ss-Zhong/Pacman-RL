from pac_env import pacman_state, pacman_web
import asyncio
import random

########################## Settings ##########################

index_url = './pac_env/pacman/index.html' # index.html的全路径
# ports = [8765, 8766, 8767, 8768, 8769, 8770] # port list 多端口就多开
ports = [8765]

##############################################################

async def robot(env):
    possible_direction = env.pacmans[0].get_possible_direction()
    
    direction = random.choice(possible_direction)

    await env.pacmans[0].turn(direction) # 0-3 右 下 左 上
    await asyncio.sleep(1)


async def main():
    env = pacman_state.Multi_PACMAN(ports)
    # asyncio.to_thread(pacman_web.open_web, ports, zoom=0.5, index_url=index_url)
    
    task1 = asyncio.create_task(asyncio.to_thread(pacman_web.open_web, ports, zoom=0.5, index_url=index_url)) # web打开
    task2 = asyncio.create_task(asyncio.to_thread(env.run)) # 游戏记录打开

    await asyncio.sleep(5)
    
    print("Both tasks are running concurrently, continuing with other code...")
    for i in range(len(ports)):
        await env.pacmans[i].enter() # 开局按 Enter, enter可以开始、继续、暂停

    for i in range(100):
        await robot(env) # 对game1进行操作
    
    await asyncio.gather(task1, task2)

if __name__ == "__main__":
    asyncio.run(main())