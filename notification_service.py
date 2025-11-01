import datetime
import winsound
import os
import time
from win10toast import ToastNotifier
import numpy as np

class NotificationService:
    def __init__(self):
        self.log_file = "emergency_log.txt"
        self.last_notification_time = 0
        self.notification_cooldown = 5
        self.toaster = ToastNotifier()
        
    def send_emergency_notifications(self, confidence_score):
        current_time = time.time()
        
        if current_time - self.last_notification_time < self.notification_cooldown:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Windows notification
            try:
                self.toaster.show_toast(
                    "⚠️ EMERGENCY - Scream Detected!",
                    f'Confidence Score: {confidence_score:.2f}\nTime: {timestamp}',
                    duration=10,
                    threaded=True
                )
            except Exception as e:
                print(f"Notification failed: {str(e)}")
            
            # Play alarm sound with error handling
            try:
                winsound.Beep(1000, 500)  # Reduced duration to 500ms
                time.sleep(0.1)
                winsound.Beep(1200, 500)  # Second beep at different frequency
            except Exception as e:
                print(f"Sound alert failed: {str(e)}")
            
            # Console output
            alert_message = f"\n{'!'*50}\nSCREAM DETECTED at {timestamp}\nConfidence: {confidence_score:.2f}\n{'!'*50}\n"
            print(alert_message)
            
            # Log to file
            self._write_to_log(alert_message)
            
            # Update last notification time
            self.last_notification_time = current_time
            
        except Exception as e:
            print(f"Error in notification service: {str(e)}")
    
    def _write_to_log(self, message):
        try:
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            with open(self.log_file, "a") as f:
                f.write(message)
        except Exception as e:
            print(f"Failed to write to log file: {str(e)}")