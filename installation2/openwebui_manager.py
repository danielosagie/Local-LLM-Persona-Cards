import sys
import subprocess
import webbrowser
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QTimer

class OpenWebUIManager(QSystemTrayIcon):
    def __init__(self):
        super().__init__()
        self.setIcon(QIcon('custom_logo.ico'))
        self.setVisible(True)
        
        self.menu = QMenu()
        self.setContextMenu(self.menu)
        
        self.start_action = QAction("Start OpenWebUI", self)
        self.start_action.triggered.connect(self.start_openwebui)
        self.menu.addAction(self.start_action)
        
        self.stop_action = QAction("Stop OpenWebUI", self)
        self.stop_action.triggered.connect(self.stop_openwebui)
        self.menu.addAction(self.stop_action)
        
        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.exit_app)
        self.menu.addAction(self.exit_action)
        
        self.docker_running = False
        self.check_docker_status()
        
        # Set up a timer to periodically check Docker status
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_docker_status)
        self.timer.start(5000)  # Check every 5 seconds
    
    def check_docker_status(self):
        try:
            subprocess.run(["docker", "info"], check=True, capture_output=True)
            self.docker_running = True
            self.start_action.setEnabled(True)
            self.stop_action.setEnabled(True)
        except subprocess.CalledProcessError:
            self.docker_running = False
            self.start_action.setEnabled(False)
            self.stop_action.setEnabled(False)
    
    def start_openwebui(self):
        if not self.docker_running:
            subprocess.Popen(["docker", "start"])
        subprocess.Popen(["docker", "start", "open-webui"])
        webbrowser.open('http://localhost:3000')
    
    def stop_openwebui(self):
        subprocess.run(["docker", "stop", "open-webui"])
        result = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True)
        if not result.stdout.strip():
            # No containers running, stop Docker
            subprocess.run(["docker", "stop"])
    
    def exit_app(self):
        self.stop_openwebui()
        QApplication.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    manager = OpenWebUIManager()
    sys.exit(app.exec_())