# csv_example.py
import csv

def write_csv(file_name, data, field_names=None):
    """ 写入 CSV 文件 """
    with open(file_name, 'w+', newline='', encoding='utf-8') as file:
        if field_names:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(data)
        else:
            writer = csv.writer(file)
            writer.writerows(data)

def read_csv(file_name):
    """ 读取 CSV 文件 """
    with open(file_name, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)

def main():
    # CSV 文件名
    file_name = 'example.csv'

    # 数据 - 字典格式
    data_dict = [
        {'Name': 'Alice', 'Age': 30, 'City': 'New York'},
        {'Name': 'Bob', 'Age': 25, 'City': 'Los Angeles'}
    ]

    # 数据 - 列表格式
    data_list = [
        ['Name', 'Age', 'City'],
        ['Alice', 30, 'New York'],
        ['Bob', 25, 'Los Angeles']
    ]

    # 写入 CSV 文件
    # write_csv(file_name, data_dict, field_names=['Name', 'Age', 'City'])
    # 或者使用列表数据
    write_csv(file_name, data_list)

    # 读取 CSV 文件
    read_csv(file_name)

if __name__ == '__main__':
    main()
