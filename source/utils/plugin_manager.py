from typing import Dict


# from 你写过哪些真正生产可用的 Python 装饰器？ - 霍华德的回答 - 知乎
# https://www.zhihu.com/question/350078061/answer/2156821164


class PluginManager: 
    """Plugin manager""" 

    def __init__(self): 
        self.plugin_container: Dict[str:Dict[str:object]] = {} 
    
    def register(self, plugin_type: str, plugin_name: str, plugin_cls): 
        if plugin_type not in self.plugin_container: 
            self.plugin_container[plugin_type] = {} 
            self.plugin_container[plugin_type][plugin_name] = plugin_cls 
    
    def get(self, plugin_type: str, plugin_name: str): 
        if plugin_type in self.plugin_container and plugin_name in self.plugin_container[plugin_type]: 
            return self.plugin_container[plugin_type][plugin_name] 
        else: 
            return None



DefaultPluginManager = PluginManager() 

def register_plugin(plugin_type: str, plugin_name: str): 

    def decorator(cls): 
        DefaultPluginManager.register(plugin_type, plugin_name, cls) 
        return cls 
        
    return decorator 
    
def get_plugin(plugin_type: str, plugin_name: str): 
    return DefaultPluginManager.get(plugin_type, plugin_name)



# import plugin 

# @plugin.register_plugin("model_loss", 'BCELoss') 


# class BCELoss: 
#     @classmethod 
#     def build(cls, cfg): 
#         return nn.BCEWithLogitsLoss(reduction='sum')

# loss = plugin.get_plugin("model_loss", "BCELoss")