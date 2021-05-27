'''
图像量化是一种有损压缩方法，可以用单色替换图像中的一系列相似颜色。
量化减少了图像文件的大小，因为表示颜色所需的位数更少。
在下面的示例中，我们将使用聚类发现包含其最重要颜色的图像的压缩调色板。
然后，我们将使用压缩的调色板重建图像。 此示例需要mahotas或skimage
图像处理库，可以使用pip install mahotas或pip install scikit-image安装
'''

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage import io  #pip install scikit-image
import mahotas as mh   #pip install mahotas

# Read and flatten the image
#original_img = np.array(mh.imread('../data/CN-wp4.jpg'), dtype=np.float64) / 255
original_img = np.array(io.imread('CN-wp4.jpg'), dtype=np.float64) / 255

original_dimensions = tuple(original_img.shape)
width, height, depth = tuple(original_img.shape)
image_flattened = np.reshape(original_img, (width * height, depth))
print(image_flattened.shape)

# Use clustering method to create n_colors (4, 8, 16, 32, 64, 128) of clusters from a sample of 1000 randomly selected colors.
# Each of the clusters will be a color in the compressed palette.
n_colors = 4
image_array_sample = shuffle(image_flattened, random_state=0)[:1000]
# 创建对象estimator并用数据image_array_sample对它进行训练
# Start your code here --------------------------------------------------
estimator=KMeans(n_clusters=64,random_state=0)
estimator.fit(image_array_sample)
# End -------------------------------------------------------------------

# Predict the cluster assignment for each of the pixels in the original image
cluster_assignments = estimator.predict(image_flattened)
compressed_palette = estimator.cluster_centers_
compressed_img = np.zeros((width, height, compressed_palette.shape[1]))
label_idx = 0
# Find new color from compressed_palette according cluster_assignments
#Start your code here----------------------------------------
for i in range(width):
    for j in range(height):
        compressed_img[i][j]=compressed_palette[cluster_assignments[label_idx]]
        label_idx+=1
#End ---------------------------------------------------------
plt.subplot(121)
plt.title('Original Image')
plt.imshow(original_img)
plt.axis('off')
plt.subplot(122)
plt.title('Compressed Image')
plt.imshow(compressed_img)
plt.axis('off')
plt.show()
