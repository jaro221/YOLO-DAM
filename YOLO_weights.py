from main import model_dam

# Load v26 pretrained weights
path_v26 = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/YOLODAM_pretrained_v26.h5"
model_dam.load_weights(path_v26)
print(f"  Loaded: {path_v26}")

# Save v11-transferred weights
path_v11 = r"D:/Projekty/2022_01_BattPor/2025_12_Dresden/Models/YOLODAM_pretrained_v11.h5"
model_dam.save_weights(path_v11)
print(f"  Saved: {path_v11}")
