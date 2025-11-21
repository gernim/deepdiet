# Read the original splits
with open('./data/nutrition5k_dataset/dish_ids/splits/train.txt', 'r') as f:
    all_train_ids = [line.strip() for line in f if line.strip()]

with open('./data/nutrition5k_dataset/dish_ids/splits/test.txt', 'r') as f:
    all_test_ids = [line.strip() for line in f if line.strip()]

# Create small subsets (e.g., 10 dishes each)
subset_train = all_train_ids[:100]
subset_test = all_test_ids[:10]

# Save to new split files (or overwrite existing)
with open('./data/nutrition5k_dataset/dish_ids/splits/train.txt', 'w') as f:
    f.write('\n'.join(subset_train))

with open('./data/nutrition5k_dataset/dish_ids/splits/test.txt', 'w') as f:
    f.write('\n'.join(subset_test))

print(f"Created test splits: {len(subset_train)} train, {len(subset_test)} test dishes")
