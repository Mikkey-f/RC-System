from recbole.quick_start import load_data_and_model

# 1. 加载预训练模型与数据
model_path = "saved/EchoMamba4Rec-Sep-17-2025_19-35-08.pth"
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_path)

# 2. 测试模型
test_result = model.evaluate(test_data)

# 3. 输出测试结果（如 Recall@10、MRR@10 等）
print("测试集性能：")
for metric, value in test_result.items():
    print(f"{metric}: {value:.4f}")