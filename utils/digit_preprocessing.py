import itertools
import torch

def extract_coordinates(json_data):
    temp_list = []
    if json_data is not None:
        objects = json_data.get("objects", [])
        for obj in objects:
            if obj.get("type") == "path" and "path" in obj:
                for point in obj["path"]:
                    if isinstance(point, dict) and "coords" in point:
                        coords = point["coords"]
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                x, y = coords[i], coords[i + 1]
                                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                                    temp_list.extend([x, y])
    return temp_list

def scale_coordinates(list_coords):
    x_vals = list_coords[::2]
    y_vals = list_coords[1::2]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    x_range = max_x - min_x or 1
    y_range = max_y - min_y or 1
    norm_x = [round((x - min_x) / x_range, 8) for x in x_vals]
    norm_y = [round((y - min_y) / y_range, 8) for y in y_vals]
    normalized = list(itertools.chain(*zip(norm_x, norm_y)))
    return normalized

def split_to_substroke(list_coords):
    nested_coords = []
    while len(list_coords) >= 16:
        nested_coords.append(list_coords[:16])
        list_coords = list_coords[16:]
    if list_coords:
        padded = list_coords + [0] * (16 - len(list_coords))
        nested_coords.append(padded)
    return nested_coords

def english_to_khmer_digit(digit):
    khmer_digits = {0: '០', 1: '១', 2: '២', 3: '៣', 4: '៤', 5: '៥', 6: '៦', 7: '៧', 8: '៨', 9: '៩'}
    if isinstance(digit, list) and len(digit) == 1:
        digit = digit[0]
    try:
        digit_int = int(digit)
        return khmer_digits.get(digit_int, "Invalid input")
    except (ValueError, TypeError):
        return "Invalid input"

def prediction_correctness_to_word(prediction):
    correctness = {0: 'incorrect', 1: 'correct'}
    if isinstance(prediction, list) and len(prediction) == 1:
        prediction = prediction[0]
    try:
        digit_int = int(prediction)
        return correctness.get(digit_int, "Invalid input")
    except (ValueError, TypeError):
        return "Invalid input"