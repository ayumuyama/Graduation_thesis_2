import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

colors = ['skyblue', 'lightblue', 'lightcoral', 'mistyrose']
names = ['Sky Blue', 'Light Blue', 'Light Coral', 'Misty Rose']

# 保存先設定
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S(stripe)")
base_save_dir = Path("outputs")
current_save_dir = base_save_dir / timestamp
current_save_dir.mkdir(parents=True, exist_ok=True)
plt.bar(names, [1, 1, 1, 1], color=colors)

comb_acc_plot_path = current_save_dir / "color.png"
plt.savefig(comb_acc_plot_path, bbox_inches='tight')
plt.close()