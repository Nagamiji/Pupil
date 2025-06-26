import numpy as np
import torch

khmer_consonants = ["ក", "ខ", "គ", "ឃ", "ង", "ច", "ឆ", "ជ", "ឈ", "ញ", "ដ", "ឋ", "ឌ", "ឍ", "ណ", "ត", "ថ", "ទ", "ធ", "ន", "ប", "ផ", "ព", "ភ", "ម", "យ", "រ", "ល", "វ", "ស", "ហ", "ឡ", "អ"]
khmer_independent_vowels = ["ឥ", "ឦ", "ឧ", "ឩ", "ឪ", "ឫ", "ឬ", "ឭ", "ឮ", "ឯ", "ឰ", "ឱ", "ឳ"]
khmer_dependent_vowels = ["ា", "ិ", "ី", "ឹ", "ឺ", "ុ", "ូ", "ួ", "ើ", "ឿ", "ៀ", "េ", "ែ", "ៃ", "ោ", "ៅ", "ុំ", "ំ", "ាំ", "ះ", "ិះ", "ុះ", "េះ", "ោះ"]
khmer_symbols = ["៖", "។", "៕", "៘", "៉", "៊", "់", "៌", "៍", "៎", "៏", "័", "ឲ", "ៗ", "ៈ"]
khmer_sub_consonants = ['្ក', '្ខ', '្គ', '្ឃ', '្ង', '្ច', '្ឆ', '្ជ', '្ឈ', '្ញ', '្ឋ', '្ឌ', '្ឍ', '្ណ', '្ត', '្ថ', '្ទ', '្ធ', '្ន', '្ប', '្ផ', '្ព', '្ភ', '្ម', '្យ', '្រ', '្ល', '្វ', '្ស', '្ហ', '្ឡ', '្អ']
unsorted_map = khmer_consonants + khmer_independent_vowels + khmer_dependent_vowels + khmer_symbols + khmer_sub_consonants
if len(unsorted_map) < 119:
    unsorted_map.append("##PLACEHOLDER##")
KHMER_CHARACTER_MAP = sorted(unsorted_map)

def preprocess_drawing(json_data, input_dim=16, max_points_per_substroke=8):
    strokes = [obj["path"] for obj in json_data["objects"] if obj["type"] == "path" and obj["path"]]
    if not strokes:
        return None
    raw_points = []
    for i, stroke in enumerate(strokes):
        points = [coord for point in stroke for coord in point["coords"]]
        raw_points.extend(points)
        if i < len(strokes) - 1:
            raw_points.extend([-1, -1])
    if not raw_points:
        return None
    coords = np.array(raw_points).reshape(-1, 2)
    valid_coords = np.array([c for c in coords if not np.array_equal(c, [-1, -1])])
    if valid_coords.shape[0] < 2:
        return None
    min_x, min_y = np.min(valid_coords, axis=0)
    max_x, max_y = np.max(valid_coords, axis=0)
    x_range = max_x - min_x if max_x != min_x else 1
    y_range = max_y - min_y if max_y != min_y else 1
    scaled_list = []
    for x, y in coords:
        if x == -1 and y == -1:
            scaled_list.extend([-1, -1])
        else:
            scaled_x = (x - min_x) / x_range
            scaled_y = (y - min_y) / y_range
            scaled_list.extend([scaled_x, scaled_y])
    sub_length = max_points_per_substroke * 2
    substrokes = []
    for i in range(0, len(scaled_list), sub_length):
        chunk = scaled_list[i:i + sub_length]
        if len(chunk) < sub_length:
            chunk += [0] * (sub_length - len(chunk))
        substrokes.append(chunk)
    return torch.tensor(substrokes, dtype=torch.float32).unsqueeze(0)