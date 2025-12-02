"""测试栅格地图功能"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sumo_rl.environment.env import SumoEnvironment
import numpy as np
import matplotlib.pyplot as plt
import sumo_rl

def test_grid_map():
    """测试栅格地图功能"""
    
    # 使用简单的单路口网络进行测试
    net_file = "sumo_rl/nets/single-intersection/single-intersection.net.xml"
    route_file = "sumo_rl/nets/single-intersection/single-intersection.rou.xml"
    
    # 检查文件是否存在
    if not os.path.exists(net_file):
        print(f"网络文件不存在: {net_file}")
        # 尝试使用其他网络文件
        net_file = "sumo_rl/nets/2way-single-intersection/single-intersection.net.xml"
        route_file = "sumo_rl/nets/2way-single-intersection/single-intersection.rou.xml"
        
        if not os.path.exists(net_file):
            print(f"网络文件不存在: {net_file}")
            return
    
    print("开始测试栅格地图功能...")
    
    # 测试1: 使用SumoEnvironment直接创建环境
    print("\n=== 测试1: 使用SumoEnvironment ===")
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=100,
        delta_time=5,
        enable_grid_map=True,  # 启用栅格地图
        grid_map_size=32,      # 32x32的栅格地图
        grid_map_type="enhanced"  # 使用增强模式
    )
    
    # 测试2: 使用parallel_env创建环境
    print("\n=== 测试2: 使用parallel_env ===")
    env_pz = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=100,
        delta_time=5,
        enable_grid_map=True,  # 启用栅格地图
        grid_map_size=32,      # 32x32的栅格地图
        grid_map_type="enhanced"  # 使用增强模式
    )
    
    # 获取栅格地图信息
    grid_info = env.get_grid_map_info()
    if grid_info:
        print(f"栅格地图信息: {grid_info}")
    
    # 重置环境
    obs = env.reset()
    
    # 执行几步仿真
    for step in range(3):
        print(f"\n=== 第 {step+1} 步 ===")
        
        if env.single_agent:
            action = env.action_space.sample()
        else:
            action = {ts: env.action_spaces(ts).sample() for ts in env.ts_ids}
        
        result = env.step(action)
        
        if env.single_agent:
            obs, reward, terminated, truncated, info = result
            print(f"单智能体模式 - 奖励: {reward}, 终止: {terminated}, 截断: {truncated}")
        else:
            obs, rewards, dones, info = result
            print(f"多智能体模式 - 奖励: {rewards}, 终止: {dones}")
        
        # 检查栅格地图信息
        if "grid_map" in info:
            grid_map = info["grid_map"]
            grid_shape = info["grid_map_shape"]
            print(f"栅格地图形状: {grid_shape}")
            print(f"栅格地图统计 - 车辆数: {np.sum(grid_map[:,:,0])}, "
                  f"信号灯数: {np.sum(grid_map[:,:,1])}, "
                  f"道路网络: {np.sum(grid_map[:,:,2])}")
            
            # 可视化栅格地图
            plt.figure(figsize=(12, 4))
            
            # 显示三个通道
            channels = ['车辆通道', '信号灯通道', '道路网络通道']
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.imshow(grid_map[:,:,i], cmap='hot', interpolation='nearest')
                plt.title(channels[i])
                plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(f"grid_map_step_{step+1}.png")
            print(f"栅格地图已保存为 grid_map_step_{step+1}.png")
            plt.close()
        else:
            print("未找到栅格地图信息")
    
    # 测试动态启用/禁用功能
    print("\n=== 测试动态启用/禁用功能 ===")
    
    # 禁用栅格地图
    env.enable_grid_map_feature(enable=False)
    
    # 再执行一步，应该没有栅格地图
    if env.single_agent:
        action = env.action_space.sample()
    else:
        action = {ts: env.action_spaces(ts).sample() for ts in env.ts_ids}
    
    result = env.step(action)
    if env.single_agent:
        obs, reward, terminated, truncated, info = result
    else:
        obs, rewards, dones, info = result
    
    if "grid_map" not in info:
        print("成功：禁用后没有栅格地图")
    else:
        print("错误：禁用后仍然有栅格地图")
    
    # 重新启用
    env.enable_grid_map_feature(enable=True, grid_map_size=16, grid_map_type="basic")
    
    # 再执行一步，应该有栅格地图
    if env.single_agent:
        action = env.action_space.sample()
    else:
        action = {ts: env.action_spaces(ts).sample() for ts in env.ts_ids}
    
    result = env.step(action)
    if env.single_agent:
        obs, reward, terminated, truncated, info = result
    else:
        obs, rewards, dones, info = result
    
    if "grid_map" in info:
        print("成功：重新启用后有栅格地图")
        print(f"新的栅格地图形状: {info['grid_map'].shape}")
    else:
        print("错误：重新启用后没有栅格地图")
    
    # 关闭环境
    env.close()
    
    # 测试parallel_env
    print("\n=== 测试parallel_env ===")
    
    # 重置parallel_env环境
    obs_pz, info_pz = env_pz.reset()
    print(f"parallel_env重置成功，观察空间: {list(obs_pz.keys())}")
    
    # 执行一步仿真
    action_pz = {ts: env_pz.action_spaces[ts].sample() for ts in env_pz.agents}
    obs_pz, rewards_pz, terminations_pz, truncations_pz, info_pz = env_pz.step(action_pz)
    
    print(f"parallel_env执行成功，奖励: {rewards_pz}")
    
    # 检查栅格地图信息
    # 在PettingZoo中，info_pz是一个字典的字典，需要检查每个智能体的info
    found_grid_map = False
    for agent_id, agent_info in info_pz.items():
        if "grid_map" in agent_info:
            grid_map_pz = agent_info["grid_map"]
            grid_shape_pz = agent_info["grid_map_shape"]
            print(f"parallel_env栅格地图形状: {grid_shape_pz}")
            print(f"parallel_env栅格地图统计 - 车辆数: {np.sum(grid_map_pz[:,:,0])}, "
                  f"信号灯数: {np.sum(grid_map_pz[:,:,1])}, "
                  f"道路网络: {np.sum(grid_map_pz[:,:,2])}")
            found_grid_map = True
            break
    
    if not found_grid_map:
        print("parallel_env未找到栅格地图信息")
        # 打印所有可用的info键来调试
        for agent_id, agent_info in info_pz.items():
            print(f"智能体 {agent_id} 的info键: {list(agent_info.keys())}")
    
    # 关闭parallel_env环境
    env_pz.close()
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    test_grid_map()