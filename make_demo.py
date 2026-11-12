"""Generate a synthetic demo; these frames are not real manufacturing footage."""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

OUTPUT = Path('data')
OUTPUT.mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT, FPS = 640, 360, 15
PRODUCT_W, PRODUCT_H = 132, 84
N_PRODUCTS = 12
DEFECTIVE_IDS = {3, 7, 10}


def make_product(defect=None):
    patch = np.full((PRODUCT_H, PRODUCT_W, 3), 195, dtype=np.uint8)
    cv2.rectangle(patch, (8, 8), (PRODUCT_W - 9, PRODUCT_H - 9), (225, 225, 225), 2)
    cv2.rectangle(patch, (39, 27), (92, 55), (125, 125, 125), -1)
    cv2.putText(patch, 'A1', (50, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 2)
    if defect == 'stain':
        cv2.circle(patch, (106, 48), 16, (35, 35, 35), -1)
    elif defect == 'corner':
        cv2.rectangle(patch, (104, 0), (131, 25), (65, 65, 65), -1)
    elif defect == 'scratch':
        cv2.line(patch, (13, 22), (110, 68), (30, 30, 30), 8)
    return patch


def make_frame(x, defect):
    frame = np.full((HEIGHT, WIDTH, 3), 42, dtype=np.uint8)
    cv2.rectangle(frame, (0, 105), (WIDTH - 1, 260), (65, 65, 65), -1)
    cv2.line(frame, (0, 104), (WIDTH, 104), (100, 100, 100), 3)
    cv2.line(frame, (0, 261), (WIDTH, 261), (100, 100, 100), 3)
    product = make_product(defect)
    y = 142
    x1, y1 = max(0, x), y
    x2, y2 = min(WIDTH, x + PRODUCT_W), y + PRODUCT_H
    if x2 > x1:
        frame[y1:y2, x1:x2] = product[:, x1 - x:x2 - x]
    return frame


def run():
    good = make_product()
    cv2.imwrite(str(OUTPUT / 'good_product.png'), good)
    writer = cv2.VideoWriter(str(OUTPUT / 'factory_demo.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError('Cannot create demo video.')

    labels = []
    defect_types = {3: 'stain', 7: 'corner', 10: 'scratch'}
    for item_id in range(1, N_PRODUCTS + 1):
        defect = defect_types.get(item_id)
        labels.append({'item_id': item_id, 'status': 'DEFECT' if defect else 'OK'})
        # Each product travels across the same fixed conveyor belt.
        for x in range(-PRODUCT_W, WIDTH + PRODUCT_W, 37):
            writer.write(make_frame(x, defect))
    writer.release()
    pd.DataFrame(labels).to_csv(OUTPUT / 'demo_ground_truth.csv', index=False)
    print('Created: data/factory_demo.mp4')
    print('Created: data/good_product.png')
    print('Created: data/demo_ground_truth.csv')
    print('Synthetic defective products:', sorted(DEFECTIVE_IDS))


if __name__ == '__main__':
    run()
