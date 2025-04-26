import cv2
import numpy as np
import os

input_folder = 'lab3/pictures_src'
output_folder = 'lab3/pictures_results'

os.makedirs(output_folder, exist_ok=True)

def process_image(image_path, output_folder):
    # Загрузка изображения в градациях серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Не удалось загрузить изображение {image_path}.")
        return

    # Структурирующий элемент — кольцо (3x3 с 0 в центре)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Применяем эрозию
    eroded_image = cv2.erode(image, kernel, iterations=1)

    # Разница между оригиналом и эродированным изображением
    diff_image = cv2.absdiff(image, eroded_image)

    # Бинаризация (опционально)
    _, binary_image = cv2.threshold(eroded_image, 127, 255, cv2.THRESH_BINARY)

    # Сохранение результатов
    base_name = os.path.basename(image_path)
    eroded_path = os.path.join(output_folder, f"eroded_{base_name}")
    diff_path = os.path.join(output_folder, f"diff_{base_name}")
    binary_path = os.path.join(output_folder, f"binary_{base_name}")

    cv2.imwrite(eroded_path, eroded_image)
    cv2.imwrite(diff_path, diff_image)
    cv2.imwrite(binary_path, binary_image)

    print(f"Изображение {base_name} обработано (эрозия) и сохранено в {output_folder}.")


# Обработка всех изображений в папке
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    if os.path.isfile(image_path):
        process_image(image_path, output_folder)