"""
Footfall Counter using Computer Vision
Detects, tracks, and counts people crossing a virtual line
"""

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Configuration
# VIDEO_PATH = 0
# VIDEO_PATH = "test.mp4"
VIDEO_PATH = "mall.mp4"
CONFIDENCE_THRESHOLD = 0.4
LINE_POSITION = 0.5  # Position of counting line (0.5 = middle of frame)
TRACK_HISTORY_LENGTH = 30

class FootfallCounter:
    def __init__(self, line_position=0.5):
        """Initialize the footfall counter with counting line position"""
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for speed
        self.line_position = line_position
        self.track_history = defaultdict(list)
        self.counted_ids = set()
        self.entry_count = 0
        self.exit_count = 0
        
    def detect_and_track(self, frame):
        """Detect people and track them across frames"""
        # Run YOLOv8 tracking
        results = self.model.track(frame, persist=True, classes=[0], 
                                   conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        detections = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = box
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'centroid': (centroid_x, centroid_y),
                    'track_id': track_id,
                    'confidence': conf
                })
        
        return detections
    
    def update_counts(self, detections, frame_height):
        """Update entry and exit counts based on line crossing"""
        counting_line_y = int(frame_height * self.line_position)
        
        for detection in detections:
            track_id = detection['track_id']
            centroid = detection['centroid']
            
            # Store track history
            self.track_history[track_id].append(centroid)
            if len(self.track_history[track_id]) > TRACK_HISTORY_LENGTH:
                self.track_history[track_id].pop(0)
            
            # Check for line crossing
            if len(self.track_history[track_id]) >= 2 and track_id not in self.counted_ids:
                prev_y = self.track_history[track_id][-2][1]
                curr_y = self.track_history[track_id][-1][1]
                
                # Check if crossed from top to bottom (Entry)
                if prev_y < counting_line_y and curr_y >= counting_line_y:
                    self.entry_count += 1
                    self.counted_ids.add(track_id)
                
                # Check if crossed from bottom to top (Exit)
                elif prev_y > counting_line_y and curr_y <= counting_line_y:
                    self.exit_count += 1
                    self.counted_ids.add(track_id)
    
    def draw_visualizations(self, frame, detections):
        """Draw bounding boxes, trajectories, and counting line"""
        frame_height, frame_width = frame.shape[:2]
        counting_line_y = int(frame_height * self.line_position)
        
        # Draw counting line
        cv2.line(frame, (0, counting_line_y), (frame_width, counting_line_y), 
                (0, 255, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (10, counting_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            centroid = detection['centroid']
            track_id = detection['track_id']
            
            # Draw bounding box
            color = (0, 255, 0) if track_id in self.counted_ids else (255, 0, 0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw track ID
            cv2.putText(frame, f"ID: {track_id}", (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw centroid
            cv2.circle(frame, centroid, 4, color, -1)
            
            # Draw trajectory
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
        
        return frame
    
    def draw_counts(self, frame):
        """Draw entry, exit, and total counts on frame"""
        # Create semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw counts
        cv2.putText(frame, f"Entries: {self.entry_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Exits: {self.exit_count}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Total: {self.entry_count - self.exit_count}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        return frame

def main():
    """Main function to run the footfall counter"""
    # Initialize counter
    counter = FootfallCounter(line_position=LINE_POSITION)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video loaded: {frame_width}x{frame_height} @ {fps} FPS")
    print("Processing... Press 'ESC' to quit, 'SPACE' to pause")
    
    # Optional: Save output video
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video reached")
                break
            
            frame_count += 1
            
            # Detect and track people
            detections = counter.detect_and_track(frame)
            
            # Update counts based on line crossing
            counter.update_counts(detections, frame_height)
            
            # Draw visualizations
            frame = counter.draw_visualizations(frame, detections)
            frame = counter.draw_counts(frame)
            
            # Optional: Write to output video
            # out.write(frame)
            
            # Display progress every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: Entries={counter.entry_count}, "
                      f"Exits={counter.exit_count}, "
                      f"Current={counter.entry_count - counter.exit_count}")
        
        # Display frame
        cv2.imshow("Footfall Counter", frame)
        
        # Handle key presses
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    # Cleanup
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL FOOTFALL COUNT")
    print("="*50)
    print(f"Total Entries: {counter.entry_count}")
    print(f"Total Exits: {counter.exit_count}")
    print(f"Current Occupancy: {counter.entry_count - counter.exit_count}")
    print(f"Total Frames Processed: {frame_count}")
    print("="*50)

if __name__ == "__main__":
    main()