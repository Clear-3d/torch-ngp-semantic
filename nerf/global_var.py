def __init(**kwargs):
    """
    初始化全局变量 【Dict】
    :param kwargs: 初始化参数 k=v
    :return: 
    """
    global _globals_dict
    _globals_dict = kwargs if kwargs else {}
    

def set_value(key, value):
    """
    设置
    :param key: Key
    :param value: Value
    :return: 
    """
    _globals_dict[key] = value


def get_value(key, default=None):
    """
    获取单个
    :param key: Key
    :param default: None
    :return: 
    """
    try:
        return _globals_dict[key]
    except KeyError:
        return default
    
    
def get_all(default=None):
    """
    获取全部
    :param default: None
    :return: 
    """
    try:
        return _globals_dict
    except Exception:
        return default


def pop_value(key, default=None):
    """
    删除单个key
    :param key: Key
    :param default: None
    :return: 
    """
    try:
        return _globals_dict.pop(key)
    except KeyError:
        return default


def clear():
    """清空"""
    _globals_dict.clear()