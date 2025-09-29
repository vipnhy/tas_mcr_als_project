# mcr/constraint_config.py
"""
Constraint configuration system for MCR-ALS
Supports loading and managing constraint templates from JSON files
"""
import json
import os
from typing import Dict, List, Any, Optional, Union
import numpy as np


class ConstraintConfig:
    """MCR-ALS 约束配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化约束配置管理器
        
        Parameters:
        - config_path (str, optional): JSON配置文件路径
        """
        self.constraints = {}
        self.active_constraints = []
        
        if config_path:
            self.load_from_file(config_path)
        else:
            self._load_default_constraints()
    
    def _load_default_constraints(self):
        """加载默认约束模板"""
        self.constraints = {
            "non_negativity": {
                "name": "浓度非负性约束",
                "description": "确保浓度矩阵C和光谱矩阵S中所有元素≥0",
                "type": "non_negativity",
                "enabled": True,
                "apply_to": ["C", "S"],  # 应用到浓度矩阵和光谱矩阵
                "parameters": {}
            },
            "spectral_smoothness": {
                "name": "光谱平滑度约束",
                "description": "使用二阶导数惩罚项确保光谱平滑",
                "type": "spectral_smoothness",
                "enabled": False,
                "apply_to": ["S"],  # 仅应用到光谱矩阵
                "parameters": {
                    "lambda": 1e-3,  # 平滑度惩罚系数
                    "order": 2       # 导数阶数
                }
            },
            "component_count_range": {
                "name": "组分数量范围",
                "description": "限制组分数量在合理范围内",
                "type": "component_count_range", 
                "enabled": True,
                "apply_to": ["validation"],
                "parameters": {
                    "min_components": 1,
                    "max_components": 4,
                    "default_components": 3
                }
            }
        }
        
        # 默认激活的约束
        self.active_constraints = ["non_negativity", "component_count_range"]
    
    def load_from_file(self, config_path: str):
        """从JSON文件加载约束配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.constraints = data.get('constraints', {})
            self.active_constraints = data.get('active_constraints', [])
            
            # 验证配置
            self._validate_config()
            
        except FileNotFoundError:
            print(f"警告: 配置文件 {config_path} 未找到，使用默认配置")
            self._load_default_constraints()
        except json.JSONDecodeError as e:
            print(f"警告: 配置文件 {config_path} 格式错误: {e}，使用默认配置")
            self._load_default_constraints()
    
    def save_to_file(self, config_path: str):
        """保存约束配置到JSON文件"""
        data = {
            "constraints": self.constraints,
            "active_constraints": self.active_constraints,
            "version": "1.0",
            "description": "MCR-ALS 约束配置模板"
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _validate_config(self):
        """验证配置的有效性"""
        for constraint_name in self.active_constraints:
            if constraint_name not in self.constraints:
                print(f"警告: 激活的约束 '{constraint_name}' 未定义，将被忽略")
                self.active_constraints.remove(constraint_name)
    
    def get_constraint(self, name: str) -> Optional[Dict]:
        """获取指定约束的配置"""
        return self.constraints.get(name)
    
    def add_constraint(self, name: str, constraint_config: Dict):
        """添加自定义约束"""
        # 验证约束配置必需字段
        required_fields = ['name', 'description', 'type', 'enabled', 'apply_to']
        for field in required_fields:
            if field not in constraint_config:
                raise ValueError(f"约束配置缺少必需字段: {field}")
        
        self.constraints[name] = constraint_config
    
    def enable_constraint(self, name: str):
        """启用约束"""
        if name in self.constraints:
            self.constraints[name]['enabled'] = True
            if name not in self.active_constraints:
                self.active_constraints.append(name)
        else:
            raise ValueError(f"约束 '{name}' 不存在")
    
    def disable_constraint(self, name: str):
        """禁用约束"""
        if name in self.constraints:
            self.constraints[name]['enabled'] = False
            if name in self.active_constraints:
                self.active_constraints.remove(name)
    
    def get_active_constraints(self) -> List[str]:
        """获取当前激活的约束列表"""
        return [name for name in self.active_constraints 
                if self.constraints.get(name, {}).get('enabled', False)]
    
    def get_constraints_for_matrix(self, matrix_type: str) -> List[str]:
        """获取适用于特定矩阵类型的约束"""
        applicable_constraints = []
        for name in self.get_active_constraints():
            constraint = self.constraints[name]
            if matrix_type in constraint.get('apply_to', []):
                applicable_constraints.append(name)
        return applicable_constraints
    
    def set_constraint_parameter(self, constraint_name: str, parameter_name: str, value: Any):
        """设置约束参数"""
        if constraint_name in self.constraints:
            if 'parameters' not in self.constraints[constraint_name]:
                self.constraints[constraint_name]['parameters'] = {}
            self.constraints[constraint_name]['parameters'][parameter_name] = value
        else:
            raise ValueError(f"约束 '{constraint_name}' 不存在")
    
    def validate_component_count(self, n_components: int) -> bool:
        """验证组分数量是否在允许范围内"""
        if "component_count_range" in self.get_active_constraints():
            constraint = self.constraints["component_count_range"]
            params = constraint.get("parameters", {})
            min_comp = params.get("min_components", 1)
            max_comp = params.get("max_components", 4)
            return min_comp <= n_components <= max_comp
        return True


def create_default_constraint_templates():
    """创建默认约束模板文件"""
    # 创建模板目录
    template_dir = os.path.join(os.path.dirname(__file__), "constraint_templates")
    os.makedirs(template_dir, exist_ok=True)
    
    # 标准模板
    standard_config = ConstraintConfig()
    standard_config.save_to_file(os.path.join(template_dir, "standard_constraints.json"))
    
    # 严格约束模板
    strict_config = ConstraintConfig()
    strict_config.enable_constraint("spectral_smoothness")
    strict_config.constraints["spectral_smoothness"]["parameters"]["lambda"] = 1e-2
    strict_config.save_to_file(os.path.join(template_dir, "strict_constraints.json"))
    
    # 宽松约束模板
    relaxed_config = ConstraintConfig()
    relaxed_config.disable_constraint("component_count_range")
    relaxed_config.constraints["component_count_range"]["parameters"]["max_components"] = 6
    relaxed_config.save_to_file(os.path.join(template_dir, "relaxed_constraints.json"))


if __name__ == "__main__":
    # 创建默认模板
    create_default_constraint_templates()
    print("默认约束模板已创建")