import sys
import os
import traceback
import json
import cv2
import numpy as np
import hashlib

# Global constants
CACHE_VERSION = "2.0" # Version bumped for Label Map approach

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_image(image_path):
    try:
        stream = open(image_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        stream.close()
    except Exception as e:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray[alpha < 128] = 0
        return gray
    elif len(img.shape) == 2:
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def smooth_mask(mask, radius):
    if radius <= 0: return mask
    ksize = int(radius * 2) | 1
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return smoothed

def get_component_mask(img, seed_point):
    h, w = img.shape
    x, y = int(seed_point[0]), int(seed_point[1])
    
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
        
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    flood_img = img.copy()
    lo_diff = 32
    up_diff = 32
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    
    try:
        cv2.floodFill(flood_img, flood_mask, (x, y), 255, lo_diff, up_diff, flags)
    except Exception as e:
        return None
        
    return flood_mask[1:-1, 1:-1]

def largest_rect_in_histogram(heights):
    stack = []
    max_area = 0
    best_rect = (0, 0, 0)
    h = np.append(heights, 0)
    
    for i, height in enumerate(h):
        start = i
        while stack and stack[-1][0] >= height:
            h_top, idx_top = stack.pop()
            width = i - idx_top
            area = h_top * width
            if area > max_area:
                max_area = area
                best_rect = (h_top, idx_top, i - 1)
            start = idx_top
        stack.append((height, start))
        
    return max_area, best_rect

def find_largest_inner_rectangle(mask, bbox, step=4):
    x0, y0, w, h = bbox
    roi = mask[y0:y0+h, x0:x0+w]
    rows, cols = roi.shape
    
    is_white = (roi == 255)
    h_matrix = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        if r == 0:
            h_matrix[r] = is_white[r].astype(np.int32)
        else:
            h_matrix[r] = np.where(is_white[r], h_matrix[r-1] + 1, 0)

    best_area = 0
    best_rect_global = None

    for r in range(0, rows, step):
        area, (rect_h, start_col, end_col) = largest_rect_in_histogram(h_matrix[r])
        if area > best_area:
            best_area = area
            rect_top = r - rect_h + 1
            rect_left = start_col
            rect_w = end_col - start_col + 1
            
            best_rect_global = {
                "left": rect_left + x0,
                "top": rect_top + y0,
                "width": rect_w,
                "height": rect_h,
                "area": area
            }
            
    return best_rect_global

def process_single_item(mask, item):
    smoothed_mask = smooth_mask(mask, 10)
    points = cv2.findNonZero(smoothed_mask)
    if points is None:
        return {
            "processed": False,
            "debug": ["Smoothed selection is empty"]
        }
        
    x, y, w, h = cv2.boundingRect(points)
    bbox_center_x = x + w / 2
    bbox_center_y = y + h / 2
    
    result = {
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "processed": True
    }
    
    inner_rect = find_largest_inner_rectangle(smoothed_mask, (x, y, w, h), step=2)
    
    optimal_x = bbox_center_x
    optimal_y = bbox_center_y
    
    if inner_rect:
        result["inner_rect"] = inner_rect
        inner_center_x = inner_rect["left"] + inner_rect["width"] / 2
        inner_center_y = inner_rect["top"] + inner_rect["height"] / 2
        optimal_x = (bbox_center_x + inner_center_x) / 2
        optimal_y = (bbox_center_y + inner_center_y) / 2
        result["calculation_method"] = "average_bbox_inner"
    else:
        result["calculation_method"] = "bbox_center (inner rect failed)"
        
    delta_x = optimal_x - item["x"]
    delta_y = optimal_y - item["y"]
    
    result["optimal_x"] = optimal_x
    result["optimal_y"] = optimal_y
    result["deltaX"] = int(round(delta_x))
    result["deltaY"] = int(round(delta_y))
    
    return result

def apply_watershed(img, items_in_group, group_mask):
    h, w = group_mask.shape
    markers = np.zeros((h, w), dtype=np.int32)
    
    valid_seeds = 0
    for idx, item in enumerate(items_in_group):
        seed_x, seed_y = int(item["x"]), int(item["y"])
        if 0 <= seed_x < w and 0 <= seed_y < h:
            if group_mask[seed_y, seed_x] > 0:
                cv2.circle(markers, (seed_x, seed_y), 3, idx + 1, -1)
                valid_seeds += 1
                
    if valid_seeds < 2:
        return None
    
    bg_marker_id = len(items_in_group) + 5
    markers[group_mask == 0] = bg_marker_id
    
    mask_bgr = cv2.cvtColor(group_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask_bgr, markers)
    
    sub_masks = []
    for idx in range(len(items_in_group)):
        marker_id = idx + 1
        sub_mask = np.zeros_like(group_mask)
        sub_mask[markers == marker_id] = 255
        sub_masks.append(sub_mask)
        
    return sub_masks

def save_cache_data(label_map, bubble_data, cache_base_path):
    """
    Saves Label Map (.npy) and Bubble Data (.json)
    """
    try:
        # Save compressed label map
        np.savez_compressed(cache_base_path + ".npz", label_map=label_map)
        
        # Save geometry data
        with open(cache_base_path + ".json", 'w', encoding='utf-8') as f:
            json.dump(bubble_data, f, cls=NumpyEncoder)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

def load_cache_data(cache_base_path):
    """
    Loads Label Map and Bubble Data
    """
    try:
        data_npz = np.load(cache_base_path + ".npz")
        label_map = data_npz['label_map']
        
        with open(cache_base_path + ".json", 'r', encoding='utf-8') as f:
            bubble_data = json.load(f)
            
        return label_map, bubble_data
    except Exception as e:
        print(f"Cache load failed: {e}")
        return None, None

def process_layout(image_path, json_data_path, output_path, cache_path=None):
    with open(json_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # === CACHE MODE CHECK ===
    # If image_path ends with .npz or cache_path is provided and exists as .npz
    # We prioritize using cache if available
    
    is_cache_mode = False
    label_map = None
    bubble_db = {} # id -> result dict
    
    # Try to load cache if we are in cache mode (image path is actually a cache path stub)
    if image_path == "CACHE_MODE":
        # In this mode, cache_path argument is mandatory and holds the base path
        if cache_path:
            label_map, bubble_db = load_cache_data(cache_path)
            if label_map is not None:
                is_cache_mode = True
                print("Using Label Map Cache")
    
    results = []
    
    if is_cache_mode:
        # === FAST PATH: Look up from Label Map ===
        h, w = label_map.shape
        
        # Dialate label map slightly to handle small movements near edges
        # This makes the "catchment area" of each bubble slightly larger
        # kernel = np.ones((5,5), np.uint8)
        # label_map_dilated = cv2.dilate(label_map.astype(np.uint8), kernel, iterations=1)
        # (Skip dilation for now to keep it simple and accurate)
        
        for item in data:
            cx, cy = int(item["x"]), int(item["y"])
            
            bubble_id = 0
            if 0 <= cx < w and 0 <= cy < h:
                bubble_id = int(label_map[cy, cx])
                
            if bubble_id > 0:
                # Cache Hit!
                # Retrieve geometry from DB using bubble_id (as string key)
                b_key = str(bubble_id)
                if b_key in bubble_db:
                    cached_res = bubble_db[b_key]
                    
                    # Re-calculate delta based on new text position
                    optimal_x = cached_res["optimal_x"]
                    optimal_y = cached_res["optimal_y"]
                    
                    delta_x = optimal_x - item["x"]
                    delta_y = optimal_y - item["y"]
                    
                    res = {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "processed": True,
                        "deltaX": int(round(delta_x)),
                        "deltaY": int(round(delta_y)),
                        "bbox": cached_res["bbox"],
                        "debug": ["Cache Hit"]
                    }
                    if "inner_rect" in cached_res:
                         res["inner_rect"] = cached_res["inner_rect"]
                    
                    results.append(res)
                else:
                    # ID found in map but missing in DB (should not happen)
                    results.append({"id": item["id"], "processed": False, "debug": ["Cache ID mismatch"]})
            else:
                # Background or out of bounds
                 results.append({"id": item["id"], "processed": False, "debug": ["Outside cached bubbles"]})
                 
    else:
        # === SLOW PATH: Magic Wand & Analysis ===
        img = load_image(image_path)
        h_img, w_img = img.shape
        
        # Prepare Label Map for saving
        label_map = np.zeros((h_img, w_img), dtype=np.int32)
        bubble_db = {}
        next_bubble_id = 1
        
        groups = []
        
        # 1. Grouping Phase
        for item in data:
            cx, cy = item["x"], item["y"]
            mask = get_component_mask(img, (cx, cy))
            
            if mask is None or cv2.countNonZero(mask) == 0:
                results.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "processed": False,
                    "deltaX": 0, "deltaY": 0,
                    "debug": ["Magic wand failed"]
                })
                continue
                
            found_group = False
            for group in groups:
                intersection = cv2.bitwise_and(mask, group['mask'])
                overlap_area = cv2.countNonZero(intersection)
                mask_area = cv2.countNonZero(mask)
                
                if mask_area > 0 and (overlap_area / mask_area) > 0.5:
                    group['items'].append(item)
                    group['mask'] = cv2.bitwise_or(group['mask'], mask)
                    found_group = True
                    break
            
            if not found_group:
                groups.append({
                    'items': [item],
                    'mask': mask
                })
                
        # 2. Processing Phase & Map Building
        for group in groups:
            items = group['items']
            base_mask = group['mask']
            
            current_bubble_items = [] # To store (mask, result_data)
            
            if len(items) == 1:
                item = items[0]
                res = process_single_item(base_mask, item)
                
                # Append result
                final_res = res.copy()
                final_res["id"] = item.get("id")
                final_res["name"] = item.get("name")
                results.append(final_res)
                
                # Store for Cache
                current_bubble_items.append((base_mask, res))
                
            else:
                sub_masks = apply_watershed(img, items, base_mask)
                if sub_masks:
                    for idx, item in enumerate(items):
                        res = process_single_item(sub_masks[idx], item)
                        
                        final_res = res.copy()
                        final_res["id"] = item.get("id")
                        final_res["name"] = item.get("name")
                        final_res["debug"] = ["Watershed"]
                        results.append(final_res)
                        
                        current_bubble_items.append((sub_masks[idx], res))
                else:
                    for item in items:
                        res = process_single_item(base_mask, item)
                        
                        final_res = res.copy()
                        final_res["id"] = item.get("id")
                        final_res["name"] = item.get("name")
                        final_res["debug"] = ["Watershed Failed"]
                        results.append(final_res)
                        
                        current_bubble_items.append((base_mask, res))
            
            # 3. Fill Label Map
            for mask, res_data in current_bubble_items:
                if res_data["processed"]:
                    # Assign new ID
                    bid = next_bubble_id
                    next_bubble_id += 1
                    
                    # Fill mask into Label Map
                    # mask is 0 or 255. We want to set label_map where mask > 0 to bid
                    label_map[mask > 0] = bid
                    
                    # Store geometry in DB (key is stringified ID)
                    bubble_db[str(bid)] = {
                        "optimal_x": res_data["optimal_x"],
                        "optimal_y": res_data["optimal_y"],
                        "bbox": res_data["bbox"],
                        "inner_rect": res_data.get("inner_rect")
                    }

        # 4. Save Cache if path provided
        if cache_path:
            # We strip .json or .npy extension if present to get base path
            # But the caller usually provides full path including extension.
            # Let's standardize: cache_path passed from JSX is "..../layout_hash.json"
            # We will use that base name.
            base_path = os.path.splitext(cache_path)[0]
            save_cache_data(label_map, bubble_db, base_path)
            print(f"Cache saved to {base_path}.npz/.json")

    # Output Results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"Processed {len(results)} items. Results saved to {output_path}")

if __name__ == "__main__":
    try:
        if len(sys.argv) < 4:
            print("Usage: python process.py <img_path> <in.json> <out.json> [cache_path]")
            sys.exit(1)
            
        img_path = sys.argv[1]
        in_json = sys.argv[2]
        out_json = sys.argv[3]
        cache_path = sys.argv[4] if len(sys.argv) > 4 else None
        
        process_layout(img_path, in_json, out_json, cache_path)
        
    except Exception as e:
        error_msg = str(e) + "\n" + traceback.format_exc()
        print(f"Error: {error_msg}")
        try:
            log_dir = os.path.dirname(sys.argv[3]) if len(sys.argv) >= 4 else "."
            log_path = os.path.join(log_dir, "python_error.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(error_msg)
        except:
            pass
        sys.exit(1)
