
import csv
import numpy as np
import pandas as pd
import os
from collections import Counter

# Dictionary to translate labels from English to Vietnamese
label_translation = {
    "Partially cloudy": "Có mây một phần",
    "Rain, Partially cloudy": "Mưa, Có mây một phần",
    "Clear": "Trời quang",
    "Rain, Overcast": "Mưa, U ám",
    "Overcast": "U ám",
}

# Hàm để tạo đường dẫn tương đối
def get_relative_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

# Hàm kiểm tra và yêu cầu nhập lại nếu dữ liệu vượt quá giới hạn
def get_valid_value(key, min_val, max_val, default_value):
    while True:
        try:
            value = float(input(f"{key} (từ {min_val} đến {max_val}, mặc định {default_value}): "))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Giá trị phải nằm trong khoảng từ {min_val} đến {max_val}. Hãy nhập lại.")
        except ValueError:
            print("Giá trị không hợp lệ! Hãy nhập lại bằng số.")

# Hàm làm sạch và kiểm tra dữ liệu đầu vào
def clean_input_data(user_data):
    # Dictionary chứa giá trị mặc định cho các trường
    default_values = {
        "Max Temperature": 30.0,
        "Min Temperature": 20.0,
        "Wind Speed": 5.0,
        "Cloud Cover": 50.0,
        "Relative Humidity": 70.0
    }
    
    # Giới hạn cho từng trường
    limits = {
        "Max Temperature": (-10, 42),
        "Min Temperature": (-10, 42),
        "Wind Speed": (0, 60),
        "Cloud Cover": (0, 100),
        "Relative Humidity": (0, 100)
    }
    
    cleaned_data = {}
    for key, value in user_data.items():
        try:
            num_value = float(value)
            min_val, max_val = limits[key]
            # Kiểm tra điều kiện hợp lệ cho mỗi trường
            if min_val <= num_value <= max_val:
                cleaned_data[key] = num_value
            else:
                print(f"{key} vượt quá giới hạn cho phép.")
                # Yêu cầu nhập lại nếu giá trị không hợp lệ
                cleaned_data[key] = get_valid_value(key, min_val, max_val, default_values[key])
        except ValueError:
            print(f"{key} không hợp lệ! Sử dụng giá trị mặc định.")
            cleaned_data[key] = default_values[key]
        
        # Hiển thị lại các giá trị đã nhập hợp lệ sau mỗi lần kiểm tra
        print("\nDữ liệu đã nhập hợp lệ cho đến thời điểm này:")
        for k, v in cleaned_data.items():
            print(f"{k}: {v}")
        print("-" * 30)
    
    return cleaned_data

# Load dữ liệu từ CSV và phân chia thành tập train và test
def loadData(filename="ThoiTiet_dulieu.csv"):
    path = get_relative_path(filename)
    with open(path) as f:
        data = np.array(list(csv.reader(f))[1:])  # Bỏ qua header
    np.random.shuffle(data)
    return data[:362], data[362:]

# Tính khoảng cách giữa hai điểm
def calcDistance(pointA, pointB):
    return np.linalg.norm(pointA[:5].astype(float) - pointB[:5].astype(float))

# Thuật toán k-NN
def kNearestNeighbor(trainSet, point, k=5):
    distances = [{"label": item[-1], "value": calcDistance(item, point)} for item in trainSet]
    return [item["label"] for item in sorted(distances, key=lambda x: x["value"])[:k]]

# Tìm nhãn xuất hiện nhiều nhất
def findMostOccur(labels):
    return Counter(labels).most_common(1)[0][0]

# Load dữ liệu dự đoán từ file CSV
def loadDataInput(filename):
    path = get_relative_path(filename)
    with open(path, "r") as f:
        return np.array(list(csv.reader(f))[1:2])

# Main
if __name__ == "__main__":
    # Load và xử lý dữ liệu
    trainSet, testSet = loadData()
    
    # Ghi kết quả dự đoán và tính độ chính xác
    output_path = get_relative_path("ThoiTiet_daxuly.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['Label', 'Predicted'])
        
        correct_predictions = 0
        for item in testSet:
            knn = kNearestNeighbor(trainSet, item, 5)
            prediction = findMostOccur(knn)
            correct_predictions += (item[-1] == prediction)
            writer.writerow([label_translation.get(item[-1], item[-1]), label_translation.get(prediction, prediction)])
    
    print("\n\nAccuracy (Test Set) =", correct_predictions / len(testSet))

    # Nhập và làm sạch dữ liệu từ người dùng
    user_data = {
        "Max Temperature": input("Nhập nhiệt độ cao nhất: "),
        "Min Temperature": input("Nhập vào nhiệt độ thấp nhất: "),
        "Wind Speed": input("Nhập vào tốc độ gió: "),
        "Cloud Cover": input("Nhập vào lượng mây: "),
        "Relative Humidity": input("Nhập vào độ ẩm: ")
    }
    
    # Làm sạch dữ liệu đầu vào và hiển thị kết quả cuối cùng
    cleaned_data = clean_input_data(user_data)
    print("\nDữ liệu sau khi làm sạch hoàn chỉnh:")
    print("\n\nAccuracy (Test Set) =", correct_predictions / len(testSet))
    for key, value in cleaned_data.items():
        print(f"{key}: {value}")

    # Lưu dữ liệu đã làm sạch vào CSV
    df = pd.DataFrame([cleaned_data])
    input_path = get_relative_path("ThoiTiet_input.csv")
    df.to_csv(input_path, index=False)

    # Dự đoán từ dữ liệu người dùng
    item = loadDataInput(input_path)
    knn = kNearestNeighbor(trainSet, item[0], 5)
    prediction = findMostOccur(knn)
    
    # Dịch nhãn dự đoán và ghi ra kết quả
    translated_prediction = label_translation.get(prediction, prediction)
    with open(get_relative_path("ThoiTiet_testInput.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Input Label", "Predicted"])
        writer.writerow([label_translation.get(item[0][-1], item[0][-1]), translated_prediction])
    
    print("Predicted (Dự đoán):", translated_prediction)
