"""栅格化地图生成器模块"""

import numpy as np
import sumolib
from typing import Dict, Tuple, List, Optional
import math


class GridMapGenerator:
    """栅格化地图生成器，用于将路网和车辆位置转换为NxN的栅格地图"""
    
    def __init__(self, net_file: str, grid_size: int = 32):
        """
        初始化栅格化地图生成器
        
        Args:
            net_file: SUMO网络文件路径
            grid_size: 栅格地图大小 (NxN)
        """
        self.net_file = net_file
        self.grid_size = grid_size
        self.net = sumolib.net.readNet(net_file)
        
        # 获取路网的边界坐标
        self.net_boundary = self.net.getBoundary()
        self.min_x, self.min_y, self.max_x, self.max_y = self.net_boundary
        
        # 计算缩放比例
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y
        self.scale_x = grid_size / self.width if self.width > 0 else 1
        self.scale_y = grid_size / self.height if self.height > 0 else 1
        
        # 预计算路口位置
        self.intersection_positions = self._get_intersection_positions()
        
        # 预计算道路段位置
        self.edge_positions = self._get_edge_positions()
    
    def _get_intersection_positions(self) -> Dict[str, Tuple[int, int]]:
        """获取所有路口的位置映射"""
        intersections = {}
        for junction in self.net.getNodes():
            x, y = junction.getCoord()
            grid_x, grid_y = self._world_to_grid(x, y)
            intersections[junction.getID()] = (grid_x, grid_y)
        return intersections
    
    def _get_edge_positions(self) -> Dict[str, List[Tuple[int, int]]]:
        """获取所有道路段的位置映射"""
        edges = {}
        for edge in self.net.getEdges():
            edge_id = edge.getID()
            shape = edge.getShape()
            grid_points = []
            
            # 将道路形状转换为栅格坐标
            for point in shape:
                x, y = point
                grid_x, grid_y = self._world_to_grid(x, y)
                grid_points.append((grid_x, grid_y))
            
            edges[edge_id] = grid_points
        return edges
    
    def _world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """将世界坐标转换为栅格坐标"""
        # 归一化到[0,1]范围
        norm_x = (world_x - self.min_x) / self.width if self.width > 0 else 0
        norm_y = (world_y - self.min_y) / self.height if self.height > 0 else 0
        
        # 映射到栅格坐标
        grid_x = int(norm_x * (self.grid_size - 1))
        grid_y = int(norm_y * (self.grid_size - 1))
        
        # 确保在有效范围内
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))
        
        return grid_x, grid_y
    
    def generate_grid_map(self, vehicle_positions: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """
        生成栅格化地图
        
        Args:
            vehicle_positions: 车辆位置字典 {vehicle_id: (x, y)}
            
        Returns:
            NxN的栅格地图，1表示有车，0表示没有车
        """
        grid_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # 填充车辆位置
        for vehicle_id, (x, y) in vehicle_positions.items():
            grid_x, grid_y = self._world_to_grid(x, y)
            grid_map[grid_y, grid_x] = 1.0
        
        return grid_map
    
    def generate_enhanced_grid_map(self, vehicle_positions: Dict[str, Tuple[float, float]], 
                                  traffic_lights: List[str]) -> np.ndarray:
        """
        生成增强的栅格化地图，包含车辆位置和交通信号灯位置
        
        Args:
            vehicle_positions: 车辆位置字典
            traffic_lights: 交通信号灯ID列表
            
        Returns:
            增强的NxN栅格地图，包含车辆和交通信号灯信息
        """
        grid_map = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # 通道0: 车辆位置 (1表示有车)
        for vehicle_id, (x, y) in vehicle_positions.items():
            grid_x, grid_y = self._world_to_grid(x, y)
            grid_map[grid_y, grid_x, 0] = 1.0
        
        # 通道1: 交通信号灯位置 (1表示有交通信号灯)
        for tl_id in traffic_lights:
            if tl_id in self.intersection_positions:
                grid_x, grid_y = self.intersection_positions[tl_id]
                grid_map[grid_y, grid_x, 1] = 1.0
        
        # 通道2: 道路网络 (1表示有道路)
        for edge_id, points in self.edge_positions.items():
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i+1]
                
                # 在两点之间画线
                self._draw_line(grid_map, x1, y1, x2, y2, 2)
        
        return grid_map
    
    def _draw_line(self, grid_map: np.ndarray, x1: int, y1: int, x2: int, y2: int, channel: int):
        """在栅格地图上画线"""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx > dy:
            # 水平方向为主
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            
            for x in range(x1, x2 + 1):
                t = (x - x1) / (x2 - x1) if x2 != x1 else 0
                y = int(y1 + t * (y2 - y1))
                if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                    grid_map[y, x, channel] = 1.0
        else:
            # 垂直方向为主
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            
            for y in range(y1, y2 + 1):
                t = (y - y1) / (y2 - y1) if y2 != y1 else 0
                x = int(x1 + t * (x2 - x1))
                if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                    grid_map[y, x, channel] = 1.0
    
    def get_network_info(self) -> Dict:
        """获取路网信息"""
        return {
            'boundary': self.net_boundary,
            'grid_size': self.grid_size,
            'num_intersections': len(self.intersection_positions),
            'num_edges': len(self.edge_positions),
            'scale_factors': (self.scale_x, self.scale_y)
        }