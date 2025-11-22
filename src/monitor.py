import pygetwindow as gw
import time
import threading

class WindowMonitor:
    """
    Monitors active window for keywords to trigger proactive assistance.
    """
    def __init__(self, callback, keywords=None):
        self.callback = callback
        self.keywords = keywords or ["News", "Article", "PDF", "Paper", "Report", "BBC", "CNN", "Kompas", "Detik"]
        self.running = False
        self.last_window_title = ""
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _monitor_loop(self):
        while self.running:
            try:
                window = gw.getActiveWindow()
                if window:
                    title = window.title
                    if title != self.last_window_title:
                        self.last_window_title = title
                        # Check for keywords
                        if any(keyword.lower() in title.lower() for keyword in self.keywords):
                            self.callback(title)
            except Exception:
                pass
            time.sleep(2.0) # Check every 2 seconds
