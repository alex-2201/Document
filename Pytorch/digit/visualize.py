import pandas as pd
from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from digit_transform import RandomShift, RandomRotation

train_df = pd.read_csv('data/train.csv')

n_train = len(train_df)
n_pixels = len(train_df.columns) - 1
n_class = len(set(train_df['label']))

print(f'Number of training samples: {n_train}')
print(f'Number of training pixels: {n_pixels}')
print(f'Number of classes: {n_class}')

test_df = pd.read_csv('data/test.csv')

n_test = len(test_df)
n_pixels = len(test_df.columns)  # Không trừ 1 vì không có cột labels như train_df

print(f'Number of train samples: {n_test}')
print(f'Number of test pixels: {n_pixels}')

# Hiển thị một số ảnh

random_sel = np.random.randint(n_train, size=8)
random_train_df = train_df.iloc[random_sel, 1:].values.reshape(-1, 28, 28) / 255.
grid = make_grid(torch.Tensor(random_train_df).unsqueeze(1), nrow=8)
plt.rcParams['figure.figsize'] = (16, 2)
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
print(*list(train_df.iloc[random_sel, 0].values), sep=', ')

# Hiển thị histogram

plt.rcParams['figure.figsize'] = (8, 5)
plt.bar(train_df['label'].value_counts().index, train_df['label'].value_counts())
plt.xticks(np.arange(n_class))
plt.xlabel('Class', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.grid('on', axis='y')

# Hiển thị ảnh sau khi transform

rotate = RandomRotation(20)
shift = RandomShift(3)
composed = transforms.Compose([RandomRotation(20),
                               RandomShift(3)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = transforms.ToPILImage()(train_df.iloc[65, 1:].values.reshape((28, 28)).astype(np.uint8)[:, :, None])
for i, tsfrm in enumerate([rotate, shift, composed]):
    transformed_sample = tsfrm(sample)
    ax = plt.subplot(1, 3, i + 1)
    plt.rcParams['figure.figsize'] = (6, 2)
    ax.set_title(type(tsfrm).__name__)
    ax.imshow(np.reshape(np.array(list(transformed_sample.getdata())), (-1, 28)), cmap='gray')

plt.show()
