import numpy as np
from skimage import io
from skimage.metrics import mean_squared_error, structural_similarity as ssim
from skimage.transform import resize
import os

def convert_to_rgb(image):
    """Преобразовать изображение в формат RGB."""
    if image.ndim == 3 and image.shape[-1] == 4:  # Проверка на RGBA
        return image[:, :, :3]  # Убираем альфа-канал
    return image

# Чтение изображений
image1 = io.imread(os.path.expanduser('~/Desktop/RandomRainbow.png'))
image2 = io.imread(os.path.expanduser('~/Desktop/20241101_021138.jpg'))

# Преобразование изображений в RGB
image1 = convert_to_rgb(image1)
image2 = convert_to_rgb(image2)

# Проверка формата
print(f"Shape of Image 1: {image1.shape}")
print(f"Shape of Image 2: {image2.shape}")

# Приведение ко одному размеру - делаем меньшее изображение 800x800
image2_resized = resize(image2, (800, 800), anti_aliasing=True)

# Проверка размеров после приведения
print(f"Resized Image 2 Shape: {image2_resized.shape}")

# Подсчет минимальных размеров изображений
min_height = min(image1.shape[0], image2_resized.shape[0])
min_width = min(image1.shape[1], image2_resized.shape[1])

# Оценка минимального размера
print(f"Minimum Height: {min_height}, Minimum Width: {min_width}")

# Убедимся, что размеры достаточно большие для SSIM
if min_height < 7 or min_width < 7:
    raise ValueError("Размер изображений меньше 7x7, SSIM не может быть вычислен.")

# Устанавливаем win_size как минимальный размер, если он нечетный
win_size = min(min_height, min_width, 11)
if win_size % 2 == 0:
    win_size -= 1  # Делаем его нечетным, если он четный

print(f"Using window size: {win_size}")

# Проверяем тип данных изображений и устанавливаем data_range
data_range = image1.max() - image1.min() if image1.dtype.name == 'float' else 255

# Вычисление MSE и SSIM
mse = mean_squared_error(image1, image2_resized)

# Добавляем data_range в ssim
similarity_index, _ = ssim(image1, image2_resized, full=True, win_size=win_size, channel_axis=-1, data_range=data_range)

print(f"MSE: {mse}")
print(f"SSIM: {similarity_index}")
