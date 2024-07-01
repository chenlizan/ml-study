import tensorflow as tf

# 第一步：从本地读取图片文件
image_raw = tf.io.read_file("pgy.jpg")

# 第二步：解码图片数据
image = tf.image.decode_jpeg(image_raw, channels=3)

# 第三步：可选的图片预处理，例如调整大小和标准化
image = tf.image.resize(image, [180, 180])  # 假设我们需要的图片大小是299x299
# image /= 255  # 将像素值标准化到0到1之间

print(image.shape)
