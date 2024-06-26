import happybase

from rowkey import get_new_rowkey

connection = happybase.Connection('127.0.0.1', 9090)

# 列出所有的表
print(connection.tables())

table_name = 'message_line_1'
table = connection.table(table_name)

row_key = get_new_rowkey(table_name)

table.put(row_key, {b'message:sender': b'user_A', b'message:content': b'hello', b'message:timestamp': b'6-25 9:00'})


row = table.row(row_key)
print(row[b'message:content'])

connection.close()
