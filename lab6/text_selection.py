from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageOps import invert


from PIL import Image
from PIL.ImageOps import invert
import numpy as np

def calculate_profiles(img):
    """Вычисление профилей изображения"""
    profile_x = np.sum(img, axis=0)  # Горизонтальный профиль (по столбцам)
    profile_y = np.sum(img, axis=1)  # Вертикальный профиль (по строкам)
    return {'x': profile_x, 'y': profile_y}

def get_symbol_boxes(img, min_symbol_width=5):
    """
    Находит границы символов с учетом минимальной ширины символа
    min_symbol_width - минимальная ожидаемая ширина символа в пикселях
    """
    profiles = calculate_profiles(img)
    borders = []
    i = 0
    width = profiles['x'].shape[0]
    
    while i < width:
        # Ищем начало символа (ненулевой столбец)
        if profiles['x'][i] > 0:
            x1 = i
            # Ищем конец символа (нулевой столбец после минимум min_symbol_width пикселей)
            x2 = i
            while x2 < width and (profiles['x'][x2] > 0 or x2 - x1 < min_symbol_width):
                x2 += 1
            
            # Проверяем, что найденный символ достаточно широк
            if x2 - x1 >= min_symbol_width:
                borders.append((x1, x2))
                i = x2  # Перескакиваем на позицию после символа
            else:
                i += 1  # Пропускаем слишком узкие участки
        else:
            i += 1
    
    return borders

if __name__ == '__main__':
    # Загрузка и подготовка изображения
    img_src = Image.open('lab6/out/sentence/1.png').convert('L')
    img_src_arr = np.array(img_src)
    
    # Инвертируем изображение (0 - фон, 1 - символ)
    img_arr = np.zeros(shape=img_src_arr.shape)
    img_arr[img_src_arr == 0] = 1  # Черные пиксели (символы) становятся 1
    img_arr[img_src_arr == 255] = 0  # Белый фон становится 0
    
    # Получаем границы символов с минимальной шириной 10 пикселей
    symbol_boxes = get_symbol_boxes(img_arr, min_symbol_width=10)
    
    # Сохраняем каждый найденный символ
    for i, (x1, x2) in enumerate(symbol_boxes):
        # Добавляем небольшие отступы вокруг символа (по 2 пикселя с каждой стороны)
        padding = 2
        x1 = max(0, x1 - padding)
        x2 = min(img_arr.shape[1], x2 + padding)
        
        symbol_img = img_src_arr[:, x1:x2]
        invert(Image.fromarray(symbol_img)).save(
            f"lab6/pictures_results/symbols/{i+1}.png")
    
    print(f"Найдено {len(symbol_boxes)} символов")