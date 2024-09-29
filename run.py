import pacman_state, pacman_web
import asyncio
import random

########################## Settings ##########################

index_url = './pacman/index.html' # index.html的全路径
# ports = [8765, 8766, 8767, 8768, 8769, 8770] # port list 多端口就多开
ports = [8765]

##############################################################

key = [39, 40, 37, 38, 13] # 右 下 左 上 回车

async def robot(pacmans):
    possible_direction = pacmans.pacmans[0].get_possible_direction()
    
    direction = random.choice(possible_direction)

    await pacmans.pacmans[0].key_press(key[direction])
    await asyncio.sleep(1)


async def main():
    pacmans = pacman_state.Multi_PACMAN(ports)
    # asyncio.to_thread(pacman_web.open_web, ports, zoom=0.5, index_url=index_url)
    
    task1 = asyncio.create_task(asyncio.to_thread(pacman_web.open_web, ports, zoom=0.5, index_url=index_url)) # web打开
    task2 = asyncio.create_task(asyncio.to_thread(pacmans.run)) # 游戏记录打开

    await asyncio.sleep(5)

    print("Both tasks are running concurrently, continuing with other code...")
    for i in range(len(ports)):
        await pacmans.pacmans[i].key_press(key[4]) # 开局按 Enter

    for i in range(100):
        await robot(pacmans) # 对game1进行操作
    
    await asyncio.gather(task1, task2)

if __name__ == "__main__":
    asyncio.run(main())