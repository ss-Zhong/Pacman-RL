# Pacman-RL

强化学习在玩吃豆人（Pac-Man）中的应用是一个引人入胜的研究领域。在这个经典的游戏中，玩家控制吃豆人穿越迷宫，吃掉豆子，同时避免被鬼魂捕捉。通过强化学习，模型可以通过与环境的互动学习最优策略，以最大化其获得的奖励。

游戏环境基于 [mumuy/pacman](https://github.com/mumuy/pacman)

## 启动方法

```bash
cd Pacman-RL
python .\run.py
```

实时获取游戏状态

<img src="Img/image-20240929170314022.png" alt="image-20240929170314022" style="zoom: 50%;" />

## 训练方法(未完成)

```bash
cd Pacman-RL
python .\train.py [-h] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--url URL] [--gpu_id GPU_ID] [--unshow_windows]
``` 

## 开发

### 实时地图显示

注释掉 pac_env\pacman_state.py 文件中的run_multiple_pacman()内的

```python
self.pacmans[0].display_game()
```

即可关闭实时地图显示

### 修改浏览器console.log是否显示

修改 pac_env\pacman_web.py 文件中的 ``CONSOLE``变量即可（True 输出，False 不输出）

### 输出包含位置信息

```python
utils.log_message('...')	# 输入和 print 一样
```

得到结果如下

```bash
[ <Path> \Pacman-RL\ <File> : <line> ]...
```

