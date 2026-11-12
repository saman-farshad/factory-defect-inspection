import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def read_product(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w < 15 or h < 15:
        return None
    crop = gray[y:y+h, x:x+w]
    return cv2.resize(crop, (132, 84), interpolation=cv2.INTER_AREA)


def find_products(frame, threshold=100, min_area=2800):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    products = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        products.append({'box': (x, y, w, h), 'center': (x + w // 2, y + h // 2)})
    return sorted(products, key=lambda item: item['center'][0])


def classify_product(frame, box, reference, pixel_threshold, defect_threshold):
    x, y, w, h = box
    crop = frame[y:y+h, x:x+w]
    image = read_product(crop)
    if image is None:
        return 'DEFECT', 1.0
    diff = cv2.absdiff(reference, image)
    ratio = float(np.mean(diff > pixel_threshold))
    return ('DEFECT' if ratio >= defect_threshold else 'OK'), ratio


def compare_ground_truth(result_df, truth_file):
    truth_df = pd.read_csv(truth_file)
    merged = truth_df.merge(result_df, on='item_id', suffixes=('_true', '_pred'))
    if merged.empty:
        print('No matching IDs for evaluation.')
        return
    y_test = merged['status_true']
    y_pred = merged['status_pred']
    print('-' * 45)
    print(f'Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%')
    print(f'Precision : {precision_score(y_test, y_pred, pos_label="DEFECT", zero_division=0)*100:.2f}%')
    print(f'Recall    : {recall_score(y_test, y_pred, pos_label="DEFECT", zero_division=0)*100:.2f}%')
    print(f'F1 Score  : {f1_score(y_test, y_pred, pos_label="DEFECT", zero_division=0)*100:.2f}%')
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred, labels=['OK', 'DEFECT']))
    print(f'Evaluated {len(merged)} of {len(truth_df)} labeled demo items.')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='data/factory_demo.mp4', help='Video or camera index 0')
    parser.add_argument('--reference', default='data/good_product.png')
    parser.add_argument('--truth', default=None, help='Optional CSV with item_id,status (for demos)')
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--threshold', type=int, default=100, help='Object/background separation')
    parser.add_argument('--min-area', type=int, default=2800)
    parser.add_argument('--pixel-threshold', type=int, default=28)
    parser.add_argument('--defect-threshold', type=float, default=0.04)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    reference_image = cv2.imread(args.reference)
    if reference_image is None:
        raise FileNotFoundError('Good product reference not found. Run python make_demo.py first.')
    reference = read_product(reference_image, args.threshold)
    if reference is None:
        raise ValueError('No product visible in reference image.')

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError('Cannot open the input video.')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    if fps <= 0:
        fps = 15
    inspection_line = width // 2
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path / 'inspection_result.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('Cannot create the output video.')

    tracks = {}
    result_list = []
    next_id = 1
    frame_number = 0
    total_ok, total_defect = 0, 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            products = find_products(frame, args.threshold, args.min_area)
            assigned_ids = set()

            for product in products:
                center = product['center']
                candidates = []
                for item_id, track in tracks.items():
                    if item_id in assigned_ids or frame_number - track['last_frame'] > 4:
                        continue
                    dist = np.linalg.norm(np.array(center) - np.array(track['center']))
                    if dist < 90:
                        candidates.append((dist, item_id))

                if candidates:
                    _, item_id = min(candidates)
                    track = tracks[item_id]
                else:
                    item_id = next_id
                    next_id += 1
                    track = {'center': center, 'last_frame': frame_number,
                             'checked': False, 'status': 'WAIT', 'ratio': 0}
                    tracks[item_id] = track

                assigned_ids.add(item_id)
                old_x = track['center'][0]
                track['center'] = center
                track['last_frame'] = frame_number
                x, y, w, h = product['box']

                # Inspect once per item as its center passes the inspection line.
                if not track['checked'] and old_x < inspection_line <= center[0]:
                    status, ratio = classify_product(frame, product['box'], reference,
                                                     args.pixel_threshold, args.defect_threshold)
                    track.update({'checked': True, 'status': status, 'ratio': ratio})
                    total_defect += int(status == 'DEFECT')
                    total_ok += int(status == 'OK')
                    result_list.append({'item_id': item_id, 'time_seconds': round(frame_number / fps, 2),
                                        'status': status, 'difference_ratio': round(ratio, 4)})
                    print(f'Item {item_id}: {status}, image difference = {ratio*100:.1f}%')

                color = (0, 0, 255) if track['status'] == 'DEFECT' else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'#{item_id} {track["status"]}', (x, max(y - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Drop stale tracks to keep memory bounded on long videos.
            tracks = {i: t for i, t in tracks.items() if frame_number - t['last_frame'] < 20}
            cv2.line(frame, (inspection_line, 90), (inspection_line, height - 70), (255, 255, 0), 2)
            cv2.rectangle(frame, (0, 0), (width, 56), (25, 25, 25), -1)
            cv2.putText(frame, f'Checked: {total_ok + total_defect}    OK: {total_ok}    DEFECT: {total_defect}',
                        (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            writer.write(frame)
            if not args.no_show:
                cv2.imshow('Factory Defect Inspection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    df = pd.DataFrame(result_list, columns=['item_id', 'time_seconds', 'status', 'difference_ratio'])
    df.to_csv(output_path / 'inspection_report.csv', index=False)
    print('-' * 45)
    print(f'Products checked: {total_ok + total_defect}')
    print(f'OK: {total_ok}, DEFECT: {total_defect}')
    print('Saved outputs in:', output_path)
    if args.truth is not None:
        compare_ground_truth(df, args.truth)


if __name__ == '__main__':
    run()
