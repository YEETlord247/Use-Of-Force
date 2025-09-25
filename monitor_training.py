import time
import os
import psutil
import datetime

def monitor_training():
    last_metrics_time = 0
    last_confusion_time = 0
    
    while True:
        # Clear screen
        os.system('clear')
        
        # Print current time
        print(f"\n=== Training Monitor === {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if training process is running
        training_running = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'train_model.py' in ' '.join(proc.info['cmdline'] or []):
                    training_running = True
                    cpu_percent = proc.cpu_percent()
                    memory_info = proc.memory_info()
                    print(f"\nTraining Process Status:")
                    print(f"CPU Usage: {cpu_percent:.1f}%")
                    print(f"Memory Usage: {memory_info.rss / 1024 / 1024:.1f} MB")
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if not training_running:
            print("\nTraining process not found!")
            break
        
        # Check metrics file
        if os.path.exists('training_metrics.png'):
            current_metrics_time = os.path.getmtime('training_metrics.png')
            if current_metrics_time > last_metrics_time:
                print("\nMetrics updated!")
                last_metrics_time = current_metrics_time
        
        # Check confusion matrix
        if os.path.exists('confusion_matrix.png'):
            current_confusion_time = os.path.getmtime('confusion_matrix.png')
            if current_confusion_time > last_confusion_time:
                print("Confusion matrix updated!")
                last_confusion_time = current_confusion_time
        
        # Sleep for a bit
        time.sleep(5)

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user") 