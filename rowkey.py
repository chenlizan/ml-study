import redis

# Redis服务器配置
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
COUNTER_KEY = 'hbase_rowkey_counter'

# 创建Redis连接
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


def get_new_rowkey(table_name):
    # 使用Redis的INCR命令来递增计数器并获取新的值
    # INCR命令是原子性的，因此可以在并发环境中安全地使用
    new_value = r.incr(f'{COUNTER_KEY}_{table_name}')
    return f'seq_id_{new_value}'
