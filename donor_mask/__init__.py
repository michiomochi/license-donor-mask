"""donor_mask: 運転免許証裏面の「臓器提供に関する意思表示」欄をローカルでマスクする（fail-closed）。"""
import cv2

cv2.setNumThreads(1)  # 並列化による浮動小数点の順序差を排除（決定性）

__version__ = "0.1.0"
