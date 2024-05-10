import csv

# 原始CSV檔案路徑
input_file = 'mitbih_ptbdb_train.csv'

# 新CSV檔案路徑
output_file = 'mitbih_ptbdb_train_sliced.csv'

# 要擷取的列索引範圍
columns_to_extract = range(0, 15)

# 要擷取的最後一列索引
last_column = 187

# 開啟原始CSV檔案
with open(input_file, 'r') as file:
    csv_reader = csv.reader(file)

    # 開啟新CSV檔案
    with open(output_file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)

        # 逐行讀取原始CSV檔案
        for row in csv_reader:
            # 擷取指定範圍的列
            extracted_columns = [row[i] for i in columns_to_extract]
            
            # 擷取最後一列
            extracted_columns.append(row[last_column])
            
            # 將擷取的列寫入新CSV檔案
            csv_writer.writerow(extracted_columns)

print("擷取完成,新的CSV檔案已生成。")