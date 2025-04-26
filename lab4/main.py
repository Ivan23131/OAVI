import numpy as np
import cv2
import os

# Функция для перевода изображения в оттенки серого
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

# Функция для вычисления градиента Робертса (вариант 2)
def roberts_gradient(image):
    # Ядра оператора Робертса 2x2 (вариант 2)
    kernel_x = np.array([[0, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]], dtype=np.float32)
    
    kernel_y = np.array([[0, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]], dtype=np.float32)
    
    # Применяем свертку с ядрами Робертса
    Gx = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    Gy = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    
    # Вычисляем градиент по формуле G = |Gx| + |Gy|
    G = np.abs(Gx) + np.abs(Gy)
    
    return Gx, Gy, G

# Функция нормализации изображения
def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Функция бинаризации
def binarize_image(image, threshold=50):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

# Пути к папкам
input_folder = 'lab4/pictures_src'
output_folder = 'lab4/pictures_results'
os.makedirs(output_folder, exist_ok=True)

# Обработка изображений
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Ошибка: Не удалось загрузить {image_name}")
        continue

    # Преобразование в полутоновое
    gray_image = convert_to_grayscale(image)
    
    # Вычисление градиента Робертса
    Gx, Gy, G = roberts_gradient(gray_image)
    
    # Нормализация
    Gx_norm = normalize_image(Gx)
    Gy_norm = normalize_image(Gy)
    G_norm = normalize_image(G)
    
    # Бинаризация
    binary_G = binarize_image(G_norm, threshold=50)
    
    # Сохранение результатов
    base_name = os.path.splitext(image_name)[0]
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_gray.png"), gray_image)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_roberts_gx.png"), Gx_norm)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_roberts_gy.png"), Gy_norm)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_roberts_g.png"), G_norm)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_binary.png"), binary_G)
    
    print(f"Обработано: {image_name}")

print("Все изображения обработаны с оператором Робертса (вариант 2)!")